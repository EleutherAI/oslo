import copy
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist

from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.distributed.parallel_mode import ParallelMode
from oslo.torch.nn.parallel.utils import (
    get_parallel_context,
    add_wrapper,
    OsloParallelWrapper,
)
from oslo.torch.nn.parallel.data_parallel._reducer import Reducer
from oslo.torch.nn.parallel.data_parallel._utils import (
    is_ddp_ignored,
    DistributedBackwardFunction,
)

from enum import Enum, auto
from typing import Optional, Tuple, Dict, Any


class ShardingStrategy(Enum):
    SHARD_PARAM = auto()
    SHARD_GRAD_PARAM = auto()
    HETERO_SHARD = auto()


def DistributedDataParallel(
    module: nn.Module,
    parallel_context: ParallelContext,
    model_wrapper_config: Optional[Dict[str, Any]] = None,
    optimizer_wrapper_config: Optional[Dict[str, Any]] = None,
    sharding_strategy: Optional[ShardingStrategy] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer]]:
    """
    This function wraps a PyTorch module with a distributed data parallel wrapper for OSLO.

    This wrapper allows the module to be trained across multiple GPUs or machines, with optional sharding strategies
    to reduce memory footprint. The function supports different sharding strategies that determines how the model
    parameters and optimizer states are partitioned across the GPUs.

    Supported sharding strategies are:
    - None: No sharding is used. This is the default strategy, where each GPU maintains a full replica of the model.
    - SHARD_PARAM: Shards the model parameters across GPUs.
    - SHARD_GRAD_PARAM: Shards the gradient as well as the model parameters across GPUs.
    - HETERO_SHARD: Use the CPU-GPU heterogeneous memory space to store the model data, inspired from PatrickStar.

    For the SHARD_PARAM, SHARD_GRAD_PARAM, and HETERO_SHARD strategies, it is mandatory to provide an optimizer.

    Args:
        module (nn.Module): PyTorch module object to be wrapped.
        parallel_context (ParallelContext): Process group object for distributed training.
        model_wrapper_config (Optional[Dict[str, Any]]): Additional configuration parameters for the model wrapper.
        optimizer_wrapper_config (Optional[Dict[str, Any]]): Additional configuration parameters for the optimizer wrapper.
        sharding_strategy (Optional[ShardingStrategy]): The strategy for sharding. Options include None, SHARD_PARAM, SHARD_GRAD_PARAM, and HETERO_SHARD.
        optimizer (Optional[torch.optim.Optimizer]): PyTorch optimizer object to be wrapped if a sharding strategy is specified.

    Returns:
        Tuple[nn.Module, Optional[torch.optim.Optimizer]]: The wrapped module and optimizer (if applicable).

    Raises:
        AssertionError: If a sharding strategy other than None is selected, but no optimizer is provided.
    """
    if sharding_strategy is not None:
        assert (
            optimizer is not None
        ), "optimizer must be provided when sharding_strategy is not None"
        from oslo.torch.nn.parallel.data_parallel import zero

    model_wrapper_config = model_wrapper_config or {}
    optimizer_wrapper_config = optimizer_wrapper_config or {}

    def default_strategy():
        ddp = _DistributedDataParallel(
            module=module, parallel_context=parallel_context, **model_wrapper_config
        )
        add_wrapper(
            module,
            mode=ParallelMode.DATA,
            wrapper=ddp,
            parallel_context=parallel_context,
        )
        return module

    def shard_param_strategy():
        optimizer_wrapper_config.pop("partition_grad", None)
        return module, zero.ZeroRedundancyOptimizer(
            optimizer,
            parallel_context=parallel_context,
            partition_grad=False,
            **optimizer_wrapper_config,
        )

    def shard_grad_param_strategy():
        optimizer_wrapper_config.pop("partition_grad", None)
        return module, zero.ZeroRedundancyOptimizer(
            optimizer,
            parallel_context=parallel_context,
            partition_grad=True,
            **optimizer_wrapper_config,
        )

    def hetero_shard_strategy():
        fsdp = zero._HeteroDataParallel(
            module=module,
            device=torch.device("cuda"),
            parallel_context=parallel_context,
            force_outputs_fp32=True,
            **model_wrapper_config,
        )
        opt = zero._HeteroOptimizer(
            optimizer,
            module=fsdp,
            **optimizer_wrapper_config,
        )
        add_wrapper(
            module,
            mode=ParallelMode.DATA,
            wrapper=fsdp,
            parallel_context=parallel_context,
        )
        return module, opt

    strategy_map = {
        None: default_strategy,
        ShardingStrategy.SHARD_PARAM: shard_param_strategy,
        ShardingStrategy.SHARD_GRAD_PARAM: shard_grad_param_strategy,
        ShardingStrategy.HETERO_SHARD: hetero_shard_strategy,
    }

    strategy = strategy_map.get(sharding_strategy)

    return strategy()


class _DistributedDataParallel(OsloParallelWrapper):
    """Distributed data parallel wrapper for OSLO.
    Example:
        >>> from oslo.torch.nn.parallel import DistributedDataParallel as DDP
        >>> model = torch.nn.Linear(20, 1)
        >>> model = DDP(model, parallel_context)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> oslo.ready(model, parallel_context)
        >>> model.zero_grad()
        >>> logits = model(x)
        >>> loss = criterion(logits, labels)
        >>> loss.backward()
        >>> optimizer.step()
    Args:
        module (nn.Module): PyTorch module object
        parallel_context (ParallelContext): process group object
    """

    def __init__(
        self,
        module: torch.nn.Module,
        parallel_context: ParallelContext = None,
        bucket_cap_mb: int = 25,
        rebuild_bucket: bool = True,
    ) -> None:
        super().__init__(parallelism_priority=99)
        self.module = module

        self.comm_stream: torch.cuda.Stream = torch.cuda.Stream()
        assert parallel_context
        self.parallel_context = get_parallel_context(module, parallel_context)
        self.dp_world_size = self.parallel_context.get_world_size(ParallelMode.DATA)

        self.reducer = Reducer(bucket_cap_mb)
        self.rebuild_bucket = rebuild_bucket

        for p in module.parameters():
            if is_ddp_ignored(p):
                continue
            if p.requires_grad:
                p.register_hook(partial(self.grad_handle, p))

    def parallelize(self):
        self._forward = copy.copy(self.module.forward)
        self.module.zero_grad = self.zero_grad

    def forward(self, *args, **kwargs):
        # inputs must be `torch.Tensor` or collections that contain `torch.Tensor`
        inputs = self._forward(*args, **kwargs)
        if isinstance(inputs, dict):
            return type(inputs)(
                {
                    k: v
                    for k, v in zip(
                        inputs.keys(),
                        DistributedBackwardFunction.apply(self, *inputs.values()),
                    )
                }
            )

        single_output = isinstance(inputs, torch.Tensor)
        if single_output:
            inputs = (inputs,)

        outputs = DistributedBackwardFunction.apply(self, *inputs)
        return outputs[0] if single_output else outputs

    def _pre_backward(self):
        pass

    def _post_backward(self):
        with torch.cuda.stream(self.comm_stream):
            self.reducer.flush()
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        if self.rebuild_bucket:
            self.reducer.free()
        for p in self.module.parameters():
            if is_ddp_ignored(p):
                continue
            p.grad = p._saved_grad

    def grad_handle(self, p, grad):
        if grad.device.type != "cpu":
            empty_grad = torch.empty_like(grad)
            if self.dp_world_size > 1:
                grad = grad / self.dp_world_size
                self.comm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.comm_stream):
                    self.reducer.all_reduce_async(
                        grad,
                        group=self.parallel_context.get_group(ParallelMode.DATA),
                        callback_fn=partial(self._save_grad, p),
                    )
                grad.record_stream(self.comm_stream)
            else:
                _DistributedDataParallel._save_grad(p, grad)

            return empty_grad

        else:
            # You must assign the model to CPU after invoking ``oslo.ready()``.
            dist.all_reduce(
                grad, group=self.parallel_context.get_cpu_group(ParallelMode.DATA)
            )
            return grad

    @staticmethod
    def _save_grad(p, grad):
        if hasattr(p, "_saved_grad"):
            p._saved_grad.add_(grad)
        else:
            p._saved_grad = grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        super().zero_grad(set_to_none=True)
        for p in self.module.parameters():
            if getattr(p, "_saved_grad", None) is not None:
                if set_to_none:
                    p._saved_grad = None
                else:
                    if p._saved_grad.grad_fn is not None:
                        p._saved_grad.detach_()
                    else:
                        p._saved_grad.requires_grad_(False)
                    p._saved_grad.zero_()
