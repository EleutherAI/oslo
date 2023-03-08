from functools import partial
from typing import Optional, List, Callable

import torch
import torch.distributed as dist
from torch.optim import Optimizer

from oslo.torch.distributed.parallel_mode import ParallelMode
from oslo.torch.distributed.parallel_context import ParallelContext

from oslo.torch.nn.parallel.data_parallel.zero.sharded_optim._utils import (
    calculate_global_norm_from_list,
    compute_norm,
    flatten,
    has_inf_or_nan,
    reduce_tensor_dp_group,
    release_param_grad,
    split_half_float_double,
    sync_param,
)

from ._base_optim import BaseOptimizerWrapper
from .bookkeeping import BucketStore, GradientStore, ParameterStore, TensorBucket


class ZeroRedundancyOptimizer(BaseOptimizerWrapper):
    """
    Optimizer used for ZeRO-1 and ZeRO-2.

    Args:
        optimizer (Optimizer):
            The optimizer to wrap.
        parallel_context (ParallelContext):
            The parallel context object.
        clip_grad_norm (float, optional):
            Clipping norm for gradients. Defaults to 0.0.
        reduce_bucket_size (int, optional):
            Size of the buckets for communication reduction. Defaults to 1024 * 1024.
        communication_dtype (torch.dtype, optional):
            Dtype for communication. Defaults to None.
        overlap_communication (bool, optional):
            Enables overlapping of communication and computation. Defaults to False.
        partition_grad (bool, optional):
            Flag for partitioning gradients in stage 2. Defaults to False.
        cpu_offload (bool, optional):
            Flag for offloading computations to CPU. Defaults to False.
        forced_dtype (torch.dtype, optional):
            Forced dtype for parameters. Defaults to None.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        parallel_context: ParallelContext,
        clip_grad_norm: float = 0.0,
        reduce_bucket_size: int = 1024 * 1024,  # communication
        communication_dtype: Optional[torch.dtype] = None,
        overlap_communication: bool = False,
        partition_grad: bool = False,  # stage 2 flag
        cpu_offload: bool = False,  # cpu offload
        forced_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the ZeroRedundancyOptimizer class.
        """

        # TODO: add support for
        # 1. fp16 master weights
        # 2. contiguous gradients
        # 3. support when some parameters requires_grad = False
        super(ZeroRedundancyOptimizer, self).__init__(optim=optimizer)
        self._dtype = self.optim.param_groups[0]["params"][0].dtype

        # stage 2
        self._partition_grads = partition_grad

        self._cpu_offload = cpu_offload

        self._dp_torch_group = parallel_context.get_group(ParallelMode.DATA)
        self._local_rank = parallel_context.get_local_rank(ParallelMode.DATA)
        self._world_size = parallel_context.get_world_size(ParallelMode.DATA)
        self._dp_global_ranks = parallel_context.get_ranks_in_group(ParallelMode.DATA)
        self._dp_torch_group = parallel_context.get_group(ParallelMode.DATA)
        self._mp_torch_group: Optional[dist.ProcessGroup]
        if parallel_context.get_world_size(ParallelMode.TENSOR) > 1:
            self._mp_torch_group = parallel_context.get_group(ParallelMode.TENSOR)
        else:
            self._mp_torch_group = None

        # fp16 and fp32 params for mixed precision training
        self._fp16_param_groups = dict()
        self._fp32_flat_param_groups_of_current_rank = dict()

        # communication params
        self._overlap_communication = overlap_communication
        self._reduce_bucket_size = reduce_bucket_size
        self._communication_dtype = communication_dtype

        self._found_overflow = torch.FloatTensor([0]).to(torch.cuda.current_device())

        # gradient clipping
        self._clip_grad_norm = clip_grad_norm

        if forced_dtype:
            for group in self.optim.param_groups:
                group_params = group["params"]
                for param in group_params:
                    param.data = param.data.to(forced_dtype)
            self._dtype = forced_dtype

        # check argument conflict
        self._sanity_checks()

        # ParameterStore will manage the tensor buffers used for zero
        # it will not manage the tensors used by mixed precision training
        self._param_store = ParameterStore(self._dp_torch_group)
        self._grad_store = GradientStore(self._dp_torch_group)
        self._bucket_store = BucketStore(self._dp_torch_group)

        # iterate over the param group in the optimizer
        # partition these param groups for data parallel training
        # and add buffers to parameter store for future access
        for group_id, param_group in enumerate(self.optim.param_groups):
            group_params = param_group["params"]

            # add the fp16 params to fp16_param_groups for bookkeeping
            self._fp16_param_groups[group_id] = group_params

            # assign parameters to ranks
            # the params in the list are sorted
            params_per_rank = self._partition_param_list(group_params)

            # store the mapping between param to rank
            # each param should belong to only one rank
            for rank, params in enumerate(params_per_rank):
                self._param_store.add_fp16_param_list_by_rank_group(
                    rank, group_id, params
                )
                for param in params:
                    self._param_store.set_param_to_rank(param, rank)

            # move to cpu to make room to create the flat tensor
            # move_tensor(params, device='cpu')
            for param in group_params:
                param.data = param.data.cpu()

            # flatten the reordered tensors
            for rank in range(self._world_size):
                tensor_list = self._param_store.get_fp16_params_by_rank_group(
                    rank, group_id
                )
                with torch.no_grad():
                    flat_tensor = flatten(tensor_list)
                flat_tensor = flat_tensor.data.cuda()
                self._param_store.add_flat_fp16_param_by_rank_group(
                    rank, group_id, flat_tensor
                )

            # sync parameters
            for rank in range(self._world_size):
                flat_tensor = self._param_store.get_flat_fp16_param_by_rank_group(
                    rank, group_id
                )
                tensor_list = self._param_store.get_fp16_params_by_rank_group(
                    rank, group_id
                )
                sync_param(flat_tensor=flat_tensor, tensor_list=tensor_list)

            # create a copy of fp32 weights of the parameters for which this rank is responsible
            fp16_flat_current_rank = (
                self._param_store.get_flat_fp16_param_by_rank_group(
                    self._local_rank, group_id
                )
            )
            fp32_flat_current_rank = fp16_flat_current_rank.float()
            device = "cpu" if self._cpu_offload else torch.cuda.current_device()
            fp32_flat_current_rank = fp32_flat_current_rank.to(device)
            fp32_flat_current_rank.requires_grad = True
            self._fp32_flat_param_groups_of_current_rank[
                group_id
            ] = fp32_flat_current_rank

            # need to replace the params in the `params` field in the optimizer
            # so that when the optimizer calls step(), it only updates the tensors
            # managed by this data parallel rank
            param_group["params"] = [fp32_flat_current_rank]

            # set reduction state
            for param in self._fp16_param_groups[group_id]:
                self._param_store.set_param_reduction_state(param, False)

        # initialize communication stream for
        # communication-compuation overlapping
        if self._overlap_communication:
            self._comm_stream = torch.cuda.Stream()

        # reduction hook is only used if overlapping communication
        # or stage 2 is used
        # if it is stage 1 without overlapping, no hook will be attached
        if self._overlap_communication or self._partition_grads:
            self._attach_reduction_bucket_hook()

        self._attach_terminate_reduction_hook()

    @property
    def dtype(self) -> torch.dtype:
        """Returns the dtype of the optimizer."""
        return self._dtype

    @property
    def num_param_groups(self) -> int:
        """Returns the number of parameter groups in the optimizer."""
        return len(self._fp16_param_groups)

    def _sanity_checks(self):
        """Performs sanity checks for the optimizer.

        Raises:
            AssertionError: If CUDA is not available.
            AssertionError: If parameters have different dtypes than the optimizer dtype.
        """
        assert torch.cuda.is_available(), "CUDA is required"
        for param_group in self.optim.param_groups:
            group_params = param_group["params"]
            for param in group_params:
                assert (
                    param.dtype == self._dtype
                ), f"Parameters are expected to have the same dtype `{self._dtype}`, but got `{param.dtype}`"

    def _partition_param_list(self, param_list: List) -> List:
        """Partitions a list of parameters into sublists for each rank in a greedy fashion.

        The parameters are sorted by number of elements in descending order and are assigned to the rank
        with the smallest number of elements.

        Args:
            param_list (list): List of parameters to partition.

        Returns:
            list: List of lists of parameters, where each sublist contains parameters for a single rank.

        """
        params_per_rank = [[] for _ in range(self._world_size)]
        numel_per_rank = [0 for _ in range(self._world_size)]

        # partititon the parameters in a greedy fashion
        sorted_params = sorted(param_list, key=lambda x: x.numel(), reverse=True)
        for param in sorted_params:
            # allocate this parameter to the rank with
            # the smallest numel for load balancing purpose
            rank_to_go = numel_per_rank.index(min(numel_per_rank))
            params_per_rank[rank_to_go].append(param)
            numel_per_rank[rank_to_go] += param.numel()

        return params_per_rank

    ###########################
    # Backward Reduction Hook #
    ###########################

    def _grad_handler(
        self, param: torch.Tensor, grad: torch.Tensor, reduce_rank: Optional[int]
    ) -> torch.Tensor:
        """Handles the gradient for the given parameter.

        Args:
            param (torch.Tensor): Tensor representing the model parameter.
            grad (torch.Tensor): Tensor representing the gradient of the parameter.
            reduce_rank (Optional[int]): Rank that should be used for reduction.

        Returns:
            torch.Tensor: The gradient tensor.
        """

        self._add_to_reduction_bucket(param, reduce_rank)
        return grad

    def _attach_reduction_bucket_hook(self):
        """Attaches gradient reduction hook to each model parameter."""
        # we iterate over the fp16 params
        # on each param, we register a hook to its AccumulateGrad object
        for group_id in range(self.num_param_groups):
            param_group = self._fp16_param_groups[group_id]
            for param in param_group:
                if param.requires_grad:
                    # determines the reduction destination rank
                    # this is only valid for stage 2
                    # dst_rank = None means using all-reduce
                    # else using reduce
                    if self._partition_grads:
                        reduce_rank = self._param_store.get_param_rank(param)
                    else:
                        reduce_rank = None

                    param.register_hook(
                        partial(self._grad_handler, param, reduce_rank=reduce_rank)
                    )

    def _attach_terminate_reduction_hook(self):
        """Binds the backward reduction hook.

        This hook is responsible for performing the reduction of the gradients and synchronizing them
        between different processes in a distributed setup.
        """
        hook_count = 0
        hook_count_target = sum(
            1
            for group_id in range(self.num_param_groups)
            for param in self._fp16_param_groups[group_id]
            if param.requires_grad
        )
        assert (
            hook_count_target > 0
        ), "At least one parameter is required to apply the reduction hook."

        def hook(param: torch.Tensor, grad, sync_grad=True):
            nonlocal hook_count
            hook_count += 1
            if hook_count == hook_count_target:
                param.grad = grad
                # finish gradient reduction
                if not self._partition_grads:
                    self._reduce_grad_stage1()
                else:
                    # TODO: support async comm in reduce
                    self._reduce_grad_stage2()

                # clear reduced grads
                if self._overlap_communication:
                    torch.cuda.synchronize()
                    self._param_store.clear_grads_of_previous_reduced_params()

                # gradient synchronization
                if sync_grad:
                    self._sync_grad()

                hook_count = 0
                return param.grad
            else:
                return grad

        for group_id in range(self.num_param_groups):
            for param in self._fp16_param_groups[group_id]:
                if param.requires_grad:
                    param.register_hook(partial(hook, param))

    def _reduce_tensor_bucket(self, bucket: TensorBucket, reduce_rank: Optional[int]):
        """Reduces the given tensor bucket.

        Args:
            bucket (TensorBucket): Bucket of tensors to reduce.
            reduce_rank (Optional[int]): Rank that should be used for reduction.

        Returns:
            None
        """
        if self._overlap_communication:
            torch.cuda.synchronize()
            self._param_store.clear_grads_of_previous_reduced_params()
            stream = self._comm_stream
        else:
            stream = torch.cuda.current_stream()

        with torch.cuda.stream(stream):
            flat = bucket.flatten()
            reduce_global_rank = None
            if reduce_rank is not None:
                reduce_global_rank = self._dp_global_ranks[reduce_rank]
            reduced_flat = reduce_tensor_dp_group(
                tensor=flat,
                dtype=self._communication_dtype,
                dst_local_rank=reduce_rank,
                dst_global_rank=reduce_global_rank,
                group=self._dp_torch_group,
            )

            # update the reduced tensor
            if reduce_rank is None or reduce_rank == self._local_rank:
                bucket.unflatten_and_copy(reduced_flat)

    def _reduce_tensor_list_with_one_dtype(
        self, tensor_list: List, bucket_size: int, reduce_rank: Optional[int]
    ):
        """
        Reduces a list of tensors with the same dtype into one tensor.

        Args:
            tensor_list (List[torch.Tensor]): List of tensors to be reduced.
            bucket_size (int): Maximum size of the tensor bucket.
            reduce_rank (int, optional): Rank to perform reduction on. If None, the tensor will be reduced on all ranks.
        """
        param_bucket = TensorBucket(bucket_size)

        for tensor in tensor_list:
            param_bucket.add_to_bucket(tensor, allow_oversize=True)

            if param_bucket.is_full_or_oversized():
                self._reduce_tensor_bucket(bucket=param_bucket, reduce_rank=reduce_rank)
                param_bucket.empty()

        if not param_bucket.is_empty():
            self._reduce_tensor_bucket(bucket=param_bucket, reduce_rank=reduce_rank)

    def _reduce_grads(
        self, reduce_rank: Optional[int], grads: List[torch.Tensor], bucket_size: int
    ):
        """Reduces the gradients.

        Args:
            reduce_rank (Optional[int]): Rank that should be used for reduction.
            grads (List[Tensor]): List of gradients to reduce.
            bucket_size (int): Size of the bucket.
        """
        grad_buckets_by_dtype = split_half_float_double(grads)

        for tensor_list in grad_buckets_by_dtype:
            self._reduce_tensor_list_with_one_dtype(
                tensor_list=tensor_list,
                bucket_size=bucket_size,
                reduce_rank=reduce_rank,
            )

    #######################
    # Reduction Functions #
    #######################

    def _run_reduction(self, reduce_rank: Optional[int] = None):
        """Reduces the gradients.

        Args:
            reduce_rank (Optional[int], optional): Rank that should be used for reduction. Defaults to None.
        """
        # reduce grads
        self._reduce_grads(
            reduce_rank=reduce_rank,
            grads=self._bucket_store.get_grad(reduce_rank=reduce_rank),
            bucket_size=self._bucket_store.num_elements_in_bucket(reduce_rank),
        )

        # use communication stream if overlapping
        # communication with computation
        if self._overlap_communication:
            stream = self._comm_stream
        else:
            stream = torch.cuda.current_stream()

        with torch.cuda.stream(stream):
            params_in_bucket = self._bucket_store.get_param(reduce_rank=reduce_rank)

            for param in params_in_bucket:
                # the is_param_reduced flag should be False showing that
                # this param is not reduced before calling self._reduce_grads_by_rank
                is_param_reduced = self._param_store.is_param_reduced(param)

                if is_param_reduced:
                    msg = (
                        f"Parameter of size ({param.size()}) has been reduced, "
                        + "duplicate reduction will lead to arithmetic incorrectness"
                    )
                    raise RuntimeError(msg)

                # update the flag
                self._param_store.set_param_reduction_state(param, True)

                # if partition grads = True
                # we do not keep the gradient after reduction
                if (
                    self._partition_grads
                    and not self._param_store.belongs_to_current_rank(param)
                ):
                    if self._overlap_communication:
                        # we need to keep this gradient for now as reduction may
                        # be completed yet since it is using a different cuda stream
                        self._param_store.add_previous_reduced_param(param)
                    else:
                        param.grad = None

        self._bucket_store.reset_by_rank(reduce_rank)

    def _add_to_reduction_bucket(
        self, param: torch.Tensor, reduce_rank: Optional[int] = None
    ):
        """add a given parameter tensor to a "reduction bucket" for reduction.

        Args:
            param (torch.Tensor): Tensor representing the model parameter.
            reduce_rank (Optional[int]): Rank that should be used for reduction. Defaults to None.

        Raises:
            RuntimeError: The parameter has already been reduced
        """
        param_size = param.numel()

        # check if the bucket is full
        # if full, will reduce the grads already in the bucket
        # after reduction, the bucket will be empty
        if (
            self._bucket_store.num_elements_in_bucket(reduce_rank) + param_size
            > self._reduce_bucket_size
        ):
            self._run_reduction(reduce_rank)

        # the param must not be reduced to ensure correctness
        is_param_reduced = self._param_store.is_param_reduced(param)
        if is_param_reduced:
            msg = (
                f"Parameter of size ({param.size()}) has already been reduced, "
                + "duplicate reduction will lead to arithmetic incorrectness"
            )
            raise RuntimeError(msg)

        self._bucket_store.add_num_elements_in_bucket(param_size, reduce_rank)
        self._bucket_store.add_param(param, reduce_rank)

    ################################
    # torch.optim.Optimizer methods
    ################################

    def zero_grad(self, set_to_none: bool = True):
        """
        Set parameter gradients to zero. If set_to_none = True, gradient
        will be set to None to save memory.

        Args:
            set_to_none (bool): Whether set the gradient to None. Default value is True.
        """
        for _, param_group in self._fp16_param_groups.items():
            for param in param_group:
                if set_to_none:
                    param.grad = None
                else:
                    if param.grad is not None:
                        param.grad.detach()
                        param.grad.zero_()

    ####################
    # Update Parameter #
    ####################

    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.

        Args:
            closure: not used by step() method, but is mandatory for compatibility
                with PyTorch optimizer interface.
        """
        assert closure is None, "closure is not supported by step()"

        # check for overflow
        found_inf = self._check_overflow()

        # update loss scale if overflow occurs
        if found_inf:
            self._grad_store.reset_all_average_gradients()
            self.zero_grad()
            return

        # copy the grad of fp16 param to fp32 param
        single_grad_partition_groups = []
        norm_groups = []

        for group_id in range(self.num_param_groups):
            # compute norm
            norm_group = compute_norm(
                gradients=self._grad_store.get_averaged_gradients_by_group(group_id),
                params=self._param_store.get_fp16_params_by_rank_group(
                    group_id=group_id, rank=self._local_rank
                ),
                dp_group=self._dp_torch_group,
                mp_group=self._mp_torch_group,
            )
            norm_groups.append(norm_group)

            # create flat gradient for the flat fp32 params
            fp16_avg_grads = self._grad_store.get_averaged_gradients_by_group(group_id)
            flat_fp16_avg_grads = flatten(fp16_avg_grads)

            dtype = self._fp32_flat_param_groups_of_current_rank[group_id].dtype
            flat_fp32_avg_grads = flat_fp16_avg_grads.to(dtype)

            param_shape = self._fp32_flat_param_groups_of_current_rank[group_id].shape
            assert (
                param_shape == flat_fp32_avg_grads.shape
            ), f"fp32 param and grad have different shape {param_shape} vs {flat_fp32_avg_grads.shape}"

            single_grad_partition_groups.append(flat_fp32_avg_grads)
            device = self._fp32_flat_param_groups_of_current_rank[group_id].device
            self._fp32_flat_param_groups_of_current_rank[
                group_id
            ].grad = flat_fp32_avg_grads.to(device)
            self._grad_store.reset_average_gradients_by_group(group_id)

        # unscale and clip grads
        global_norm = calculate_global_norm_from_list(norm_list=norm_groups)
        self._unscale_and_clip_grads(single_grad_partition_groups, global_norm)

        # update the parameters
        self.optim.step()
        # release the fp32 grad
        release_param_grad(self._fp32_flat_param_groups_of_current_rank.values())

        # update fp16 partition updated by the current rank
        for group_id in range(len(self._fp16_param_groups)):
            fp16_param = self._param_store.get_flat_fp16_param_by_rank_group(
                rank=self._local_rank, group_id=group_id
            )
            fp32_param = self._fp32_flat_param_groups_of_current_rank[group_id]
            fp16_param.data.copy_(fp32_param)

        # broadcast the updated model weights
        handles = []
        for group_id in range(self.num_param_groups):
            for index in range(self._world_size):
                rank = self._dp_global_ranks[index]
                fp16_param = self._param_store.get_flat_fp16_param_by_rank_group(
                    rank=index, group_id=group_id
                )
                handle = dist.broadcast(
                    fp16_param, src=rank, group=self._dp_torch_group, async_op=True
                )
                handles.append(handle)

        for handle in handles:
            handle.wait()

    ##################
    # FP16 Utilities #
    ##################

    def _check_overflow(self):
        """check for the presence of numerical overflow
        (infinite or NaN values) in the averaged gradient values of a set of model parameters.

        Returns:
            bool: The presence of numerical overflow.
        """
        # clear previous overflow record
        self._found_overflow.fill_(0.0)

        # check for overflow
        for group_id in range(len(self._fp16_param_groups)):
            for avg_grad in self._grad_store.get_averaged_gradients_by_group(group_id):
                if avg_grad is not None and has_inf_or_nan(avg_grad):
                    self._found_overflow.fill_(1.0)
                    break

        # all-reduce across dp group
        dist.all_reduce(
            self._found_overflow, op=dist.ReduceOp.MAX, group=self._dp_torch_group
        )

        # all-reduce over model parallel group
        if self._mp_torch_group:
            dist.all_reduce(
                self._found_overflow, op=dist.ReduceOp.MAX, group=self._mp_torch_group
            )

        if self._found_overflow.item() > 0:
            return True
        else:
            return False

    def _unscale_and_clip_grads(self, grad_groups_flat: List, total_norm: float):
        """
        Unscale the gradient norms and clip them if the `clip_grad_norm` argument is provided.

        Args:
            grad_groups_flat (List): List of PyTorch tensors representing the gradient.
            total_norm (float): Combined normalization factor for the gradients.
        """
        if self._clip_grad_norm > 0.0:
            clip = (total_norm + 1e-6) / self._clip_grad_norm
            if clip > 1:
                for grad in grad_groups_flat:
                    grad.data.mul_(1.0 / clip)

    ############################
    # Gradient Synchronization #
    ############################

    def _sync_grad(self):
        """
        Synchronizes the gradients of all parameters in the model by accumulating them from multiple devices/processes.
        """
        # update param already reduced flag
        reduction_states = self._param_store.get_param_reduction_states()
        for tensor, _ in reduction_states.items():
            reduction_states[tensor] = False

        # accumulate gradient
        for group_id in range(self.num_param_groups):
            param_group = self._param_store.get_fp16_params_by_rank_group(
                self._local_rank, group_id
            )

            group_avg_gradients = self._grad_store.get_averaged_gradients_by_group(
                group_id
            )

            param_idx = 0
            for param in param_group:
                if param.grad is not None:
                    if len(group_avg_gradients) == param_idx:
                        self._grad_store.append_average_gradient_by_group(
                            group_id, param.grad
                        )
                    else:
                        self._grad_store.add_average_gradient_by_group(
                            group_id, param_idx, param.grad
                        )
                    param_idx += 1

        # the gradients needed are stored in the avg_gradients buffer
        # thus, can clear this
        self.zero_grad()

    def _reduce_grad_stage1(self):
        """
        Reduces the gradients to the model parameters stage1.
        """
        # if not overlapping communication (no reduction hook is attached)
        # we need to manually reduce these gradients
        if not self._overlap_communication:
            for group_id in range(len(self._fp16_param_groups)):
                param_group = self._fp16_param_groups[group_id]
                for param in param_group:
                    if param.grad is not None:
                        self._add_to_reduction_bucket(param)

        # we need to reduce the gradients
        # left in the communication bucket
        self._run_reduction()

    def _reduce_grad_stage2(self):
        """
        Reduces the gradients to the model parameters stage2.
        """
        # when partition_grads is True, reduction hooks
        # are attached in the __init__ function, so we
        # only need to reduce the gradients
        # left in the communication bucket
        for reduce_rank in range(self._world_size):
            self._run_reduction(reduce_rank)
