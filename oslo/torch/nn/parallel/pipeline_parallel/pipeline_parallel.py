import concurrent.futures
import copy
import time
from threading import Lock

import torch
import torch.nn as nn
from torch.distributed import rpc

from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.distributed.parallel_mode import ParallelMode
from oslo.torch.nn.parallel.pipeline_parallel._buffers import save_activation
from oslo.torch.nn.parallel.pipeline_parallel._functional import (
    apply_backward_redirection,
    len_forward_marker,
)
from oslo.torch.nn.parallel.pipeline_parallel._messages import assemble_args
from oslo.torch.nn.parallel.pipeline_parallel._model_partitioner import ModelPartitioner
from oslo.torch.nn.parallel.pipeline_parallel._server import (
    _ORIGINAL_FORWARDS,
    _MODULE_DEVICE_LOCATIONS,
    remote_module_forward,
    _RESULT_DICT,
    get_result,
    reset_result,
    increment_done,
    get_done,
    reset_done,
    _FORWARD_COUNTER,
    get_forward_counter,
    increment_forward_counter,
    reset_forward_counter,
)
from oslo.torch.nn.parallel.utils import get_parallel_context, add_wrapper


def PipelineParallel(
    module: nn.Module,
    parallel_context: ParallelContext,
    memory_computation_balance: float = 1.0,
    num_micro_batches: int = 1,
):
    pp = _PipelineParallel(
        module=module,
        parallel_context=parallel_context,
        memory_computation_balance=memory_computation_balance,
        num_micro_batches=num_micro_batches,
    )
    add_wrapper(module, ParallelMode.PIPELINE, pp)
    setattr(module, "forward", pp.forward)
    return module


class _PipelineParallel(nn.Module):
    """
    Pipeline parallel module

    Args:
        module (nn.Module): PyTorch module object
        parallel_context (ParallelContext): process group object
        memory_computation_balance (float): memory computation balance factor

    Notes:
        1. Similar design with `torch.nn.parallel.DistributedDataParallel`.
        2. Support inter-module partitioning described in Sagemaker Model Parallelism.

    Examples:
        >>> from oslo.torch.nn.parallel import PipelineParallel
        >>>
        >>> model = TransformersModel()
        >>> optimizer = AnyOptimizer(model.parameters(), lr=3e-5)
        >>> pp_wrapper = PipelineParallel(model, ...)

        >>> output = pp_wrapper(input_data)
        >>> output.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        module: nn.Module,
        parallel_context: ParallelContext,
        memory_computation_balance: float = 1.0,
        num_micro_batches: int = 1,
    ):
        super().__init__()
        self.module = module
        self.module_forward = copy.copy(module.forward)
        self.parallel_context = get_parallel_context(module, parallel_context)
        self.partitioner = ModelPartitioner(
            module=module,
            process_group=parallel_context.get_group(ParallelMode.PIPELINE),
            memory_computation_balance=memory_computation_balance,
        )
        self.partitioner.partition()
        self.oslo_parallel = self.module.oslo_parallel

        self._recursive_wrap(self, "")
        self._lock = Lock()
        self.rank = self.parallel_context.get_local_rank(ParallelMode.PIPELINE)
        self.num_micro_batches = num_micro_batches

        # set up worker for inputs
        self.producer = None
        if self.rank == 0:
            self.producer = concurrent.futures.ThreadPoolExecutor()

    def forward(self, *args, **kwargs):
        # set forward counter to zero
        reset_forward_counter()

        # to ensure optimizer's step is done for all processes
        # TODO; barrier for only PP
        torch.distributed.barrier()

        if self.rank == 0:
            # TODO;
            new_args = [list() for _ in range(self.num_micro_batches)]
            for x in args:
                x = x.chunk(self.num_micro_batches)
                for i, x_chunk in enumerate(x):
                    new_args[i].append(x_chunk)

            new_kwargs = [dict() for _ in range(self.num_micro_batches)]
            for k, v in kwargs.items():
                v = v.chunk(self.num_micro_batches)
                for i, v_chunk in enumerate(v):
                    new_kwargs[i][k] = v_chunk

            futures = []
            # TODO; implement warm-up phase?
            # warm-up
            # ind = 0
            # while ind < self.num_micro_batches:
            #     args_ = new_args[ind]
            #     kwargs_ = new_kwargs[ind]
            #     future = self.producer.submit(self.module, *args_, **kwargs_)
            #     futures.append(future)
            #     ind += 1

            for ind, (args_, kwargs_) in enumerate(zip(new_args, new_kwargs)):
                future = self.producer.submit(self.module_forward, *args_, **kwargs_)
                futures.append(future)

            for i, done in enumerate(concurrent.futures.as_completed(futures)):
                result = done.result()
                _RESULT_DICT[i] = result

                # print(f'{i=}, {result.loss=}, {dist.get_rank()=}')

                yield result

        else:
            # forward pass end, wait results from master
            for i in range(self.num_micro_batches):
                # result = FINAL_RESULT_QUEUE.get()
                rpc_dst = self.parallel_context.get_pipeline_rpc_worker_name(0)
                result = rpc.rpc_sync(
                    to=rpc_dst,
                    func=get_result,
                    args=(i,),
                )

                yield result  # has no gradient

        # barrier ?
        # TODO; check the reason why we need this code block
        for other in self.parallel_context.get_ranks_in_group(ParallelMode.PIPELINE):
            if other == self.rank:
                increment_done()
            else:
                rpc_dst = self.parallel_context.get_pipeline_rpc_worker_name(other)
                rpc.rpc_sync(
                    to=rpc_dst,
                    func=increment_done,
                )

        while get_done() < self.parallel_context.get_world_size(ParallelMode.PIPELINE):
            time.sleep(0.0)

        reset_done()
        reset_result()

        while len_forward_marker() != 0:
            time.sleep(0.0)

        torch.cuda.empty_cache()

    def _recursive_wrap(self, module, prefix):
        if not hasattr(module, "oslo_pp_location"):  # prevent infinite loop
            setattr(module, "oslo_pp_location", prefix)
            if prefix != "":  # wrapper's forward function should not be wrapped
                self._wrap_forward(module)

        for name, m in module.named_children():
            new_prefix = f"{prefix}.{name}" if prefix != "" else name
            self._recursive_wrap(m, new_prefix)

    def _wrap_forward(self, module):
        if module is self.module:
            orig_forward = self.module_forward
        else:
            orig_forward = module.forward
        loc = module.oslo_pp_location
        device = module.oslo_parallel[ParallelMode.PIPELINE]

        _ORIGINAL_FORWARDS[loc] = orig_forward
        _MODULE_DEVICE_LOCATIONS[loc] = device
        _FORWARD_COUNTER[loc] = 0

        def new_forward(*args, **kwargs):
            location = module.oslo_pp_location
            module_device = _MODULE_DEVICE_LOCATIONS[location]
            module_device = torch.device("cuda", module_device)
            current_device = self.parallel_context.get_local_rank(ParallelMode.PIPELINE)
            current_device = torch.device("cuda", current_device)
            is_same = module_device == current_device

            if is_same:
                forward_fn = _ORIGINAL_FORWARDS[location]
                result = forward_fn(*args, **kwargs)

            else:
                arg_keys, new_args = assemble_args(args, kwargs)

                with self._lock:
                    cnt = get_forward_counter(location)
                    unique_key = (location, cnt)
                    increment_forward_counter(location)

                caller = self.parallel_context.get_pipeline_rpc_worker_name(
                    current_device.index
                )
                callee = self.parallel_context.get_pipeline_rpc_worker_name(
                    module_device.index
                )

                # prepare backward
                save_activation(unique_key, new_args)
                forward_fn = _ORIGINAL_FORWARDS[location]

                # request forward
                fut = rpc.rpc_async(
                    to=callee,
                    func=remote_module_forward,
                    args=(caller, location, unique_key, arg_keys) + new_args,
                )
                result = fut.wait()

                # TODO; does result always be an args?
                result = apply_backward_redirection(
                    callee,
                    unique_key,
                    *result,
                )

            return result

        module.forward = new_forward
