# this code is inspired by the DeepSpeed library and implemented with our own design from scratch
import math
import warnings
from enum import Enum
from typing import Any, Dict, Set, Tuple

import torch
import torch.distributed as dist
from torch.nn import Parameter
from torch.optim import Optimizer

from oslo.torch.utils.logging import DistributedLogger

from oslo.torch.nn.parallel.data_parallel.zero.sharded_optim._base_optim import (
    BaseOptimizerWrapper,
)


from oslo.torch.nn.parallel.data_parallel.zero.utils import get_current_device

from oslo.torch.nn.parallel.data_parallel._utils import (
    is_ddp_ignored,
)

from oslo.torch.nn.parallel.data_parallel.zero.chunk import Chunk, ChunkManager
from oslo.torch.nn.parallel.data_parallel.zero.fully_sharded_data_parallel import (
    _FullyShardedDataParallel,
)

import functools

from typing import Callable


def _disposable(func: Callable) -> Callable:
    executed = False

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal executed
        if not executed:
            executed = True
            return func(*args, **kwargs)

    return wrapper


class _HeterogeneousZeroOptimizer(BaseOptimizerWrapper):
    """A wrapper for optimizer. ``_FullyShardedDataParallel`` and ``_HeterogeneousZeroOptimizer`` implement Zero Redundancy Optimizer (ZeRO state-3).

    Note:
        You must use ``_FullyShardedDataParallel`` with ``_HeterogeneousZeroOptimizer``.

    Note:
        Make sure you set ``placement_policy`` of ``heterogeneousManager`` to `"auto"`,
        if you set ``gpu_margin_mem_ratio > 0``.

    Args:
        optim (Optimizer): An Optimizer instance.
        module (ZeroDDP): A ``ZeroDDP`` instance.
        gpu_margin_mem_ratio (float, optional): The ratio of GPU remaining memory (after the first forward-backward)
            which will be used when using hybrid CPU optimizer.
            This argument is meaningless when `placement_policy` of `heterogeneousManager` is not "auto".
            Defaults to 0.0.
        clipping_norm (float, optional): The norm value used to clip gradient. Defaults to 0.0.
        norm_type (float, optional): The type of norm used for gradient clipping. Currently, only L2-norm (norm_type=2.0)
            is supported in ZeroOptimizer. Defaults to 2.0.
        num_fp32_shards_per_param (int, optional): The number of fp32 shards per param. Defaults to 0.
        verbose (bool, optional): Whether to print verbose information, including grad overflow info. Defaults to False.
    """

    def __init__(
        self,
        optim: Optimizer,
        module: _FullyShardedDataParallel,
        gpu_margin_mem_ratio: float = 0.0,
        clipping_norm: float = 0.0,
        norm_type: float = 2.0,
        num_fp32_shards_per_param: int = 0,
        verbose: bool = False,
        **kwargs: Any,
    ):
        super().__init__(optim)
        assert isinstance(module, _FullyShardedDataParallel)
        self.module = module
        self.heterogeneous_manager = module.heterogeneous_manager
        self.chunk_manager: ChunkManager = self.heterogeneous_manager.chunk_manager
        self.param_to_range: Dict[Parameter, Tuple[int, int]] = dict()
        self.param_to_chunk32: Dict[Parameter, Chunk] = dict()
        self.chunk16_set: Set[Chunk] = set()
        self.clipping_flag = clipping_norm > 0.0
        self.max_norm = clipping_norm
        self.verbose = verbose

        if self.clipping_flag:
            assert norm_type == 2.0, "ZeroOptimizer only supports L2 norm now"

        ddp_param_list = []
        for name, param in module.named_parameters():
            if is_ddp_ignored(param):
                if param.requires_grad:
                    warnings.warn(
                        f"Parameter `{name}` is ignored by DDP but requires gradient! "
                        "You should handle its optimizer update by yourself!"
                    )
            else:
                ddp_param_list.append(param)

        for p, _ in zip(ddp_param_list, module.fp32_params):
            chunk_16 = self.chunk_manager.get_chunk(p)
            if chunk_16 not in self.chunk16_set:
                chunk_16.l2_norm_flag = self.clipping_flag
                self.chunk16_set.add(chunk_16)

        self.__init__optimizer()

        self._found_overflow: torch.Tensor = torch.zeros(
            1, dtype=torch.int64, device=get_current_device()
        )
        self._logger = DistributedLogger.get_instance(__name__)

        self.gpu_margin_mem_ratio: float = float(gpu_margin_mem_ratio)
        assert (
            0.0 <= self.gpu_margin_mem_ratio <= 1.0
        ), f"gpu_margin_mem_ratio must >=0.0 and <=1.0"

        self._should_move_fp32_params_h2d: bool = (
            self.heterogeneous_manager.is_cuda_margin_mem_avail
            and self.gpu_margin_mem_ratio > 0.0
            and num_fp32_shards_per_param >= 2
        )
        if (
            self.gpu_margin_mem_ratio > 0.0
            and not self.heterogeneous_manager.is_cuda_margin_mem_avail
        ):
            self._logger.warning(
                f'gpu_margin_mem_ratio is meaningless when placement_policy is not "auto"'
            )

        self._register_states = _disposable(self._register_states_)

    def _set_grad_ptr(self):
        for group in self.param_groups:
            for fake_param in group["params"]:
                chunk32 = self.param_to_chunk32[fake_param]
                begin, end = self.param_to_range[fake_param]
                chunk16 = chunk32.paired_chunk

                fake_param.data = chunk32.payload[begin:end]
                fake_param.grad = chunk16.payload[begin:end].to(chunk32.dtype)

    def _update_fp16_params(self):
        none_tensor = torch.empty([0])
        for group in self.param_groups:
            for fake_param in group["params"]:
                assert fake_param.grad is None
                fake_param.data = none_tensor.to(fake_param.device)

        for chunk16 in self.chunk16_set:
            chunk16.optim_update()

    def _check_overflow(self):
        # clear previous overflow record
        self._found_overflow.fill_(self.module.overflow_counter)

        # all-reduce across global group
        dist.all_reduce(self._found_overflow)

        return self._found_overflow.item() > 0

    def _clear_global_norm(self) -> None:
        for c16 in self.chunk16_set:
            c16.l2_norm = None

    def _calc_global_norm(self) -> float:
        norm_sqr: float = 0.0
        group_to_norm = dict()
        for c16 in self.chunk16_set:
            assert c16.l2_norm is not None

            if c16.is_gathered:
                norm_sqr += c16.l2_norm
            else:
                # this chunk is sharded, use communication to collect total norm
                if c16.torch_pg not in group_to_norm:
                    group_to_norm[c16.torch_pg] = 0.0
                group_to_norm[c16.torch_pg] += c16.l2_norm

            c16.l2_norm = None  # clear l2 norm

        comm_buffer = torch.zeros(1, dtype=torch.float, device=get_current_device())
        for group, part_norm in group_to_norm.items():
            comm_buffer.fill_(part_norm)
            dist.all_reduce(comm_buffer, group=group)
            norm_sqr += comm_buffer.item()

        global_norm = math.sqrt(norm_sqr)
        return global_norm

    def _clip_grads(self):
        total_norm = self._calc_global_norm()
        clip_coef = self.max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for group in self.param_groups:
                for param in group["params"]:
                    param.grad.data.mul_(clip_coef)

    def zero_grad(self, *args, **kwargs):
        self.module.overflow_counter = 0
        return self.optim.zero_grad(set_to_none=True)

    def step(self, *args, **kwargs):
        self._maybe_move_fp32_params()
        self._set_grad_ptr()

        found_inf = self._check_overflow()
        if found_inf:
            if self.verbose:
                self._logger.info(f"Found overflow. Skip step")
            self._clear_global_norm()  # clear recorded norm
            self.zero_grad()  # reset all gradients
            self._update_fp16_params()
            return

        if self.clipping_flag:
            self._clip_grads()
        ret = self.optim.step(*args, **kwargs)
        self._register_states()
        self.zero_grad()
        self._update_fp16_params()
        return ret

    def clip_grad_norm(self, *args, **kwargs):
        raise NotImplementedError

    def _maybe_move_fp32_params(self):
        if self._should_move_fp32_params_h2d:
            self._should_move_fp32_params_h2d = False
            available_cuda_margin_mem = (
                self.heterogeneous_manager.cuda_margin_mem * self.gpu_margin_mem_ratio
            )
            fp32_params_available_cuda_margin_mem = (
                available_cuda_margin_mem / self.optim.num_fp32_shards_per_param
            )
            fp32_params_used_cuda_margin_mem = 0

            for group in self.param_groups:
                for fake_param in group["params"]:
                    chunk32 = self.param_to_chunk32[fake_param]
                    chunk16 = chunk32.paired_chunk

                    if chunk32.device_type == "cuda":
                        continue

                    if (
                        fp32_params_used_cuda_margin_mem + chunk32.payload_mem
                        < fp32_params_available_cuda_margin_mem
                    ):
                        self.chunk_manager.move_chunk(chunk32, get_current_device())
                        # stores grad now
                        self.chunk_manager.move_chunk(chunk16, get_current_device())
                        self.module.set_chunk_grad_device(chunk16, get_current_device())
                        fp32_params_used_cuda_margin_mem += chunk32.payload_mem

            for group in self.param_groups:
                for fake_param in group["params"]:
                    chunk32 = self.param_to_chunk32[fake_param]
                    if chunk32.device_type == "cuda":
                        state = self.optim.state[fake_param]
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(get_current_device())

    def _register_states_(self):
        for group in self.optim.param_groups:
            for p in group["params"]:
                state = self.optim.state[p]
                for val in state.values():
                    if isinstance(val, torch.Tensor):
                        self.chunk_manager.add_extern_static_tensor(val)

    def __init__optimizer(self):
        def get_range_pair(local_chunk: Chunk, local_param: Parameter):
            param_info = local_chunk.tensors_info[local_param]
            if local_chunk.keep_gathered:
                return param_info.offset, param_info.end
            begin = max(0, param_info.offset - local_chunk.shard_begin)
            end = min(local_chunk.shard_size, param_info.end - local_chunk.shard_begin)
            return begin, end

        for group in self.optim.param_groups:
            fake_params_list = list()

            for param in group["params"]:
                if is_ddp_ignored(param):
                    continue
                chunk16 = self.chunk_manager.get_chunk(param)
                range_pair = get_range_pair(chunk16, param)
                if range_pair[0] >= range_pair[1]:
                    continue

                grad_device = self.module.grads_device[param]
                fake_param = torch.nn.Parameter(torch.empty([0], device=grad_device))
                self.param_to_chunk32[fake_param] = chunk16.paired_chunk
                self.param_to_range[fake_param] = range_pair

                fake_params_list.append(fake_param)

            group["params"] = fake_params_list
