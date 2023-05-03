import concurrent.futures
import copy
import time
from threading import RLock
from queue import Queue
from functools import partial

import torch
import torch.nn as nn
from torch.distributed import rpc
from transformers.utils import ModelOutput
from transformers.activations import ACT2FN

from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.distributed.parallel_mode import ParallelMode
from oslo.torch.nn.parallel.pipeline_parallel._buffers import (
    register_original_forward_function,
    get_original_forward_function,
    get_module_device_location,
    save_activation,
)
from oslo.torch.nn.parallel.pipeline_parallel._functional import (
    remote_request, send_results,
    apply_backward_job_enqueue,
    funcs,
)
from oslo.torch.nn.parallel.pipeline_parallel._messages import (
    pack_tensor_stub,
    unpack_tensor_stub,
)
from oslo.torch.nn.parallel.pipeline_parallel._model_partitioner import ModelPartitioner
from oslo.torch.nn.parallel.pipeline_parallel._sync import (
    wait_other_ranks,
    make_unique_key,
    initialize_job,
    select_job,
    reset_job_queue,
    set_result,
    get_result,
    _JOBS,
)
from oslo.torch.nn.parallel.pipeline_parallel._workers import (
    workers,
)
from oslo.torch.nn.parallel.pipeline_parallel._comm import (
    infos, enqueue_result,
    enqueue_batch_finished_notice,
    enqueue_forward_finished_notice,
    enqueue_forward_ready_notice,
    enqueue_forward_start_notice,
)
from oslo.torch.nn.parallel.pipeline_parallel._job import Metadata
from oslo.torch.nn.parallel.utils import (
    get_parallel_context,
    add_wrapper,
    OsloParallelWrapper,
)
from oslo.transformers.constants import BATCH_DIMENSIONS_PP


_DEBUG = False


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

        # TODO;
        funcs["request"] = partial(remote_request, parallel_context=parallel_context)
        funcs["send"] = partial(send_results, parallel_context=parallel_context)
        infos["PC"] = parallel_context

    @torch.no_grad()
    def parallelize(self):
        # copy activations for partitioning
        self._recursive_replace_act(self.module)

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
        self._lock = RLock()

        infos["LOCK"] = self._lock

        self.pipe_rank = self.parallel_context.get_local_rank(ParallelMode.PIPELINE)
        self.tensor_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR)

        # set up workers
        self.worker = None
        if self.parallel_context.is_first_rank(ParallelMode.PIPELINE):
            self.worker = workers

        # communication queues
        self.out_queue = Queue()
        infos["OUT_QUEUE"] = self.out_queue

        self.last_backward_notice = Queue()
        infos["LAST_BACKWARD_NOTICE"] = self.last_backward_notice

        self.batch_finished_notice = Queue()
        infos["BATCH_FINISHED_NOTICE"] = self.batch_finished_notice

        self.forward_finished_notice = Queue()
        infos["FORWARD_FINISHED_NOTICE"] = self.forward_finished_notice

        self.forward_ready_notice = Queue()
        infos["FORWARD_READY_NOTICE"] = self.forward_ready_notice

        self.forward_start_notice = Queue()
        infos["FORWARD_START_NOTICE"] = self.forward_start_notice

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

        if torch.distributed.get_rank() == 0:
            ranks_ready = set()

            # torch.cuda.synchronize(
            #     torch.distributed.get_rank()
            # )

            enqueue_forward_ready_notice(0)

            while True:
                while self.forward_ready_notice.empty():
                    time.sleep(0.05)

                r = self.forward_ready_notice.get()
                ranks_ready.add(r)

                if len(ranks_ready) == self.parallel_context.get_world_size(ParallelMode.GLOBAL):
                    break

            for dst_rank in range(1, self.parallel_context.get_world_size(ParallelMode.GLOBAL)):
                rpc.rpc_sync(
                    to=f"RPC_WORKER_{dst_rank}",
                    func=enqueue_forward_start_notice,
                )

        else:
            # torch.cuda.synchronize(
            #     torch.distributed.get_rank()
            # )

            # notice to master
            src_rank = torch.distributed.get_rank()
            dst_rank = 0
            rpc.rpc_sync(
                to=f"RPC_WORKER_{dst_rank}",
                func=enqueue_forward_ready_notice,
                args=(src_rank, ),
            )

            # wait for a sign
            while self.forward_start_notice.empty():
                time.sleep(0.05)

            _ = self.forward_start_notice.get()
        # end sync

        # TODO; seems like this is not necessary?
        torch.cuda.empty_cache()

        if self.parallel_context.is_first_rank(ParallelMode.PIPELINE):
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

            # enqueue jobs
            is_grad_enabled = torch.is_grad_enabled()

            num_concurrent = 16

            for ind, kwargs_ in enumerate(new_kwargs[:num_concurrent]):
                initialize_job(
                    fn=self.module_forward,
                    is_grad_enabled=is_grad_enabled,
                    unique_key=ind,
                    out_queue=self.out_queue,
                    **kwargs_,
                )

            for ri in range(self.num_micro_batches):
                while self.out_queue.empty():
                    time.sleep(0.05)

                # TODO; order?
                ind, result = self.out_queue.get()

                # scale loss
                if (
                    isinstance(result, ModelOutput)
                    and result.get("loss", None) is not None
                ):
                    result.loss = result.loss / self.num_micro_batches

                yield ind, result   # TODO;

                # TODO; how to find dst_rank?
                # TODO; do this like tree
                for dst_rank in range(1, self.parallel_context.get_world_size(ParallelMode.GLOBAL)):
                    rpc.rpc_sync(
                        to=f"RPC_WORKER_{dst_rank}",
                        func=enqueue_result,
                        args=(ind, result, ),
                    )

                if ri < self.num_micro_batches - num_concurrent:
                    ind = ri + num_concurrent
                    kwargs_ = new_kwargs[ind]

                    initialize_job(
                        fn=self.module_forward,
                        is_grad_enabled=is_grad_enabled,
                        unique_key=ind,
                        out_queue=self.out_queue,
                        **kwargs_,
                    )

            # barrier for last backward
            while self.last_backward_notice.empty():
                time.sleep(0.05)

            _ = self.last_backward_notice.get()

            ranks_done = set()
            ranks_done.add(0)

            while True:
                while self.batch_finished_notice.empty():
                    time.sleep(0.05)

                r = self.batch_finished_notice.get()
                ranks_done.add(r)

                if len(ranks_done) == self.parallel_context.get_world_size(ParallelMode.GLOBAL):
                    break

            # notice other ranks
            # TODO; how to find dst_rank?
            # TODO; do this like tree?
            for dst_rank in range(1, self.parallel_context.get_world_size(ParallelMode.GLOBAL)):
                rpc.rpc_sync(
                    to=f"RPC_WORKER_{dst_rank}",
                    func=enqueue_forward_finished_notice,
                )

        else:
            for _ in range(self.num_micro_batches):
                while self.out_queue.empty():
                    time.sleep(0.05)

                ind, result = self.out_queue.get()

                yield ind, result

            # barrier for last backward
            while self.last_backward_notice.empty():
                time.sleep(0.05)

            _ = self.last_backward_notice.get()

            # notice to master
            src_rank = torch.distributed.get_rank()
            dst_rank = 0
            rpc.rpc_sync(
                to=f"RPC_WORKER_{dst_rank}",
                func=enqueue_batch_finished_notice,
                args=(src_rank, ),
            )

            # barrier
            while self.forward_finished_notice.empty():
                time.sleep(0.05)

            _ = self.forward_finished_notice.get()

            # torch.cuda.synchronize(
            #     torch.distributed.get_rank()
            # )

        print(f"RANK {torch.distributed.get_rank()} | len jobs -> {len(_JOBS)}")

    def _recursive_replace_act(self, module):
        for name, m in module.named_children():
            if m in ACT2FN.values():
                setattr(module, name, copy.deepcopy(m))

            self._recursive_replace_act(m)

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
                # protect from communication overlap
                # infos["LOCK"].acquire()

                forward_fn = get_original_forward_function(location)
                result = forward_fn(*args, **kwargs)

                # infos["LOCK"].release()

            else:

                (args_stub, kwargs_stub), tensors = pack_tensor_stub([args, kwargs], [])

                tensors = tuple(tensors)

                # does not save activation if the module is in eval mode
                is_grad_enabled = torch.is_grad_enabled()
                is_training = self.training
                need_activation_save = any([t.requires_grad for t in tensors])

                # infos["LOCK"].acquire()

                unique_key = make_unique_key(location, current_device.index, module_device.index)

                if need_activation_save and is_training and is_grad_enabled:
                    # prepare backward
                    save_activation(unique_key, tensors)

                out_queue = Queue()

                request_fn = funcs["request"]
                request_fn(
                    src=current_device.index,
                    dst=module_device.index,
                    unique_key=unique_key,
                    queue=out_queue,
                    args_stub=args_stub,
                    kwargs_stub=kwargs_stub,
                    is_forward=True,
                    is_grad_enabled=is_grad_enabled,
                    is_training=is_training,
                    is_fp16=False,  # TODO;
                    func_name=location,
                    tensors=tensors,
                    parallel_context=self.parallel_context,
                )

                # infos["LOCK"].release()

                # wait for return
                while out_queue.empty():
                    time.sleep(0.05)

                result_stub, tensors = out_queue.get()
                del out_queue

                need_activation_save = any([t.requires_grad for t in tensors])
                if need_activation_save and is_training and is_grad_enabled:
                    meta = Metadata(
                        is_request=True,
                        is_first=False,
                        is_forward=False,
                        is_training=is_training,
                        is_grad_enabled=is_grad_enabled,
                        is_fp16=False,  # TODO;
                        func_name="",   # dummy
                        src=current_device.index,
                        dst=module_device.index,
                    )

                    tensors = apply_backward_job_enqueue(meta, unique_key, *tensors)

                    if _DEBUG:
                        print(f"RANK {torch.distributed.get_rank()} | apply backward redirection {unique_key}")

                result, _ = unpack_tensor_stub(result_stub, tensors)

                if _DEBUG:
                    print(f"RANK {torch.distributed.get_rank()} | got result {unique_key}")

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
