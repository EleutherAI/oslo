import concurrent.futures
import copy
from threading import Lock

import torch
import torch.nn as nn
from torch.distributed import rpc
from transformers.utils import ModelOutput

from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.distributed.parallel_mode import ParallelMode
from oslo.torch.nn.parallel.pipeline_parallel._buffers import (
    register_original_forward_function,
    get_original_forward_function,
    get_module_device_location,
    save_activation,
)
from oslo.torch.nn.parallel.pipeline_parallel._functional import (
    remote_module_forward,
    apply_backward_redirection,
)
from oslo.torch.nn.parallel.pipeline_parallel._messages import (
    pack_tensor_stub,
    unpack_tensor_stub,
)
from oslo.torch.nn.parallel.pipeline_parallel._model_partitioner import ModelPartitioner
from oslo.torch.nn.parallel.pipeline_parallel._sync import (
    wait_other_ranks,
    make_unique_key,
    reset_forward_used_counter,
    set_result,
    get_result,
)
from oslo.torch.nn.parallel.utils import (
    get_parallel_context,
    add_wrapper,
    OsloParallelWrapper,
)
from oslo.transformers.constants import BATCH_DIMENSIONS_PP


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

    add_wrapper(
        module,
        mode=ParallelMode.PIPELINE,
        wrapper=pp,
        parallel_context=parallel_context,
    )
    return module


# function to launch self.module. needs this
# function because torch.set_grad_enabled() is
# thread local.
def launch(fn, is_grad_enabled, *args, **kwargs):
    with torch.set_grad_enabled(is_grad_enabled):
        result = fn(*args, **kwargs)
    return result


class _PipelineParallel(OsloParallelWrapper):
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
        super().__init__(parallelism_priority=99)
        self.module = module
        self.module_forward = copy.copy(module.forward)
        self.parallel_context = get_parallel_context(module, parallel_context)
        self.memory_computation_balance = memory_computation_balance
        self.num_micro_batches = num_micro_batches

    @torch.no_grad()
    def parallelize(self):
        self.partitioner = ModelPartitioner(
            module=self.module,
            process_group=self.parallel_context.get_group(ParallelMode.PIPELINE),
            actual_ranks=self.parallel_context.get_ranks_in_group(
                ParallelMode.PIPELINE
            ),
            memory_computation_balance=self.memory_computation_balance,
        )
        self.partitioner.partition()
        self.oslo_parallel = self.module.oslo_parallel

        self._recursive_wrap(self, "")
        self._lock = Lock()
        self.local_rank = self.parallel_context.get_local_rank(ParallelMode.PIPELINE)

        # set up worker for inputs
        self.producer = None
        if self.local_rank == 0:
            self.producer = concurrent.futures.ThreadPoolExecutor()

    def forward(self, *args, **kwargs):
        assert len(args) == 0, (
            "Pipeline parallel model only supports ``**kwargs`` input (keyword arguments). "
            "If you wrote code like ``model(input_ids, labels)``, "
            "please modify your code like ``model(input_ids=input_ids, labels=labels)``."
        )

        if "return_dict" in kwargs and kwargs["return_dict"] is False:
            raise ValueError(
                "Pipeline parallel model does not support ``return_dict=False``. "
                "Please set ``return_dict=True``."
            )

        # set forward counter to zero
        reset_forward_used_counter()

        # to ensure optimizer's step is done for all processes
        torch.distributed.barrier(
            self.parallel_context.get_group(ParallelMode.PIPELINE)
        )

        if self.local_rank == 0:
            new_kwargs = [dict() for _ in range(self.num_micro_batches)]
            for k, v in kwargs.items():
                if k in BATCH_DIMENSIONS_PP:
                    # splittable
                    v = v.chunk(
                        self.num_micro_batches,
                        dim=BATCH_DIMENSIONS_PP[k],
                    )
                    for i, v_chunk in enumerate(v):
                        new_kwargs[i][k] = v_chunk
                else:
                    # not splittable
                    for i in range(self.num_micro_batches):
                        new_kwargs[i][k] = v

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

            is_grad_enabled = torch.is_grad_enabled()
            for ind, kwargs_ in enumerate(new_kwargs):
                future = self.producer.submit(
                    launch, self.module_forward, is_grad_enabled, **kwargs_
                )
                futures.append(future)

            for i, done in enumerate(concurrent.futures.as_completed(futures)):
                result = done.result()
                set_result(i, result)

                if (
                    isinstance(result, ModelOutput)
                    and result.get("loss", None) is not None
                ):
                    result.loss = result.loss / self.num_micro_batches

                yield result

        else:
            # TODO; the code block below does not make
            #  same number with rank 0. However, since
            #  this result is a dummy, it does not cause
            #  an error.
            # forward pass end, wait results from master
            for i in range(self.num_micro_batches):
                rpc_dst = self.parallel_context.get_pipeline_rpc_worker_name(
                    self.parallel_context.pipeline_local_master_rank
                )
                result = rpc.rpc_sync(
                    to=rpc_dst,
                    func=get_result,
                    args=(i,),
                )

                # remove gradient of non-master result.
                # without this, the users need to consider rank
                # when calling loss.backward()
                result, tensors = pack_tensor_stub(result, [])
                for i_tensor in range(len(tensors)):
                    tensors[i_tensor].grad = None
                result, _ = unpack_tensor_stub(result, tensors)

                yield result

        # barrier; wait for all rank
        wait_other_ranks(self.local_rank, self.parallel_context)

        # TODO; seems like this is not necessary?
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
        orig_forward = module.forward
        loc = module.oslo_pp_location
        # device = module.oslo_parallel[ParallelMode.PIPELINE]
        device = module.oslo_actual_pp_rank

        register_original_forward_function(loc, orig_forward, device)

        def new_forward(*args, **kwargs):
            location = module.oslo_pp_location
            module_device = get_module_device_location(location)
            module_device = torch.device("cuda", module_device)
            current_device = self.parallel_context.get_local_rank(ParallelMode.GLOBAL)
            current_device = torch.device("cuda", current_device)
            is_same = module_device == current_device

            if is_same:
                forward_fn = get_original_forward_function(location)
                result = forward_fn(*args, **kwargs)

            else:
                (args_stub, kwargs_stub), tensors = pack_tensor_stub([args, kwargs], [])
                tensors = tuple(tensors)

                # does not save activation if the module is in eval mode
                is_grad_enabled = torch.is_grad_enabled()
                is_training = self.training
                need_activation_save = any([t.requires_grad for t in tensors])
                with self._lock:
                    unique_key = make_unique_key(location, self.local_rank)

                if need_activation_save and is_training and is_grad_enabled:
                    # prepare backward
                    save_activation(unique_key, tensors)

                caller = self.parallel_context.get_pipeline_rpc_worker_name(
                    current_device.index
                )
                callee = self.parallel_context.get_pipeline_rpc_worker_name(
                    module_device.index
                )

                # request forward
                fut = rpc.rpc_async(
                    to=callee,
                    func=remote_module_forward,
                    args=(
                        caller,
                        location,
                        unique_key,
                        args_stub,
                        kwargs_stub,
                        need_activation_save,
                        is_training,
                        is_grad_enabled,
                    )
                    + tensors,
                )
                # receive result as stub
                result_stub, tensors, requires_redirection = fut.wait()

                if requires_redirection and is_training and is_grad_enabled:
                    tensors = apply_backward_redirection(
                        callee,
                        unique_key,
                        *tensors,
                    )

                result, _ = unpack_tensor_stub(result_stub, tensors)

            return result

        module.forward = new_forward

    @torch.no_grad()
    def deparallelize(self):
        # collect in global rank 0 first
        if 0 in self.parallel_context.get_ranks_in_group(ParallelMode.PIPELINE):
            pg = self.parallel_context.get_group(ParallelMode.PIPELINE)

            for name, param in self.module.named_parameters():
                src = param.oslo_parallel[ParallelMode.PIPELINE]

                from_cpu = False
                if param.device == torch.device("cpu"):
                    from_cpu = True
                    param.data = param.cuda()

                torch.distributed.broadcast(
                    param.data,
                    src=src,
                    group=pg,
                )

                if from_cpu:
                    param.data = param.cpu()

        # broadcast to all
        for name, param in self.module.named_parameters():
            from_cpu = False
            if param.device == torch.device("cpu"):
                from_cpu = True
                param.data = param.cuda()

            torch.distributed.broadcast(
                param.data,
                src=0,
                group=None,
            )

            if from_cpu:
                param.data = param.cpu()
