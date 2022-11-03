from typing import Any, Optional

import torch
from torch.distributed import ReduceOp
from torch.nn.functional import cross_entropy
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn.modules.loss import _Loss
from oslo.torch.distributed.nn.functional import all_reduce
from oslo.torch.distributed import ParallelContext, ParallelMode


class _VocabParallelCrossEntropy1D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx: Any,
        vocab_parallel_logits: Tensor,
        targets: Tensor,
        parallel_context: ParallelContext,
    ):
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        logits_max = all_reduce(
            logits_max,
            op=ReduceOp.MAX,
            parallel_context=parallel_context,
            parallel_mode=ParallelMode.TENSOR_1D,
        )
        # Subtract the maximum value.
        vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

        # Get the partition's vocab indices
        partition_vocab_size = vocab_parallel_logits.size(-1)
        rank = parallel_context.get_local_rank(ParallelMode.TENSOR_1D)
        vocab_start_index = partition_vocab_size * rank
        vocab_end_index = vocab_start_index + partition_vocab_size

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (targets < vocab_start_index) | (targets >= vocab_end_index)
        masked_target = targets.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(
            start=0, end=logits_2d.size(0), device=logits_2d.device
        )

        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(targets)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        predicted_logits = all_reduce(
            predicted_logits,
            parallel_context=parallel_context,
            parallel_mode=ParallelMode.TENSOR_1D,
        )

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = torch.exp(vocab_parallel_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        sum_exp_logits = all_reduce(
            sum_exp_logits,
            parallel_context=parallel_context,
            parallel_mode=ParallelMode.TENSOR_1D,
        )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits
        # Store softmax, target-mask and masked-target for backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)
        return loss

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):

        # Retrieve tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as their gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size(-1)
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size(0), device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).float()

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None


class VocabParallelCrossEntropyLoss1D(_Loss):
    """Vocab parallel cross entropy loss for 1D parallelism.
    Args:
        reduction (bool, optional): whether to average the loss, defaults to True.
    """

    def __init__(
        self,
        reduce_mean: bool = True,
        ignore_index: int = -100,
        parallel_context: Optional[ParallelContext] = None,
    ):
        super().__init__()
        self.reduce_mean = reduce_mean
        self.ignore_index = ignore_index
        self.parallel_context = parallel_context

    def forward(self, logits: Tensor, targets: Tensor):
        """Calculate loss between logits and targets.
        Args:
            logits (:class:`torch.tensor`): Predicted unnormalized scores (often referred to as logits).
            targets (:class:`torch.tensor`): Ground truth class indices or class probabilities.
        """
        loss = _VocabParallelCrossEntropy1D.apply(
            logits, targets, self.parallel_context
        )
        loss[targets == self.ignore_index] = 0.0
        if self.reduce_mean:
            loss = loss.sum() / (targets != self.ignore_index).sum()
        return loss


class CrossEntropyLoss2D(_Loss):
    r"""Cross entropy loss for 2D parallelism
    Args:
        reduction (bool, optional): whether to average the loss, defaults to True.
    The ``args`` and ``kwargs`` should include parameters below:
    ::
        weight (Tensor, optional)
        size_average (bool, optional)
        ignore_index (int, optional)
        reduce (bool, optional)
        label_smoothing (float, optional)
    More details about ``args``, ``kwargs`` and ``torch.nn.functional.cross_entropy`` could be found in
    `Cross_entropy <https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy>`_.
    """

    def __init__(
        self,
        reduce_mean=True,
        parallel_context: Optional[ParallelContext] = None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.reduce_mean = reduce_mean
        self.parallel_context = parallel_context
        self.loss_args = args
        self.loss_kwargs = kwargs

    def forward(self, logits: Tensor, targets: Tensor):
        from oslo.torch.nn.parallel.tensor_parallel._2d._ops import (
            split_batch_2d,
            reduce_by_batch_2d,
        )

        """Calculate loss between logits and targets.
        Args:
            logits (:class:`torch.tensor`): Predicted unnormalized scores (often referred to as logits).
            targets (:class:`torch.tensor`): Ground truth class indices or class probabilities.
        Returns:
            float: the loss between logits and targets.
        """
        targets = split_batch_2d(targets, dim=0, parallel_context=self.parallel_context)
        loss = cross_entropy(
            logits, targets, reduction="none", *self.loss_args, **self.loss_kwargs
        )
        if self.reduce_mean:
            loss = loss.mean()
            loss = reduce_by_batch_2d(
                loss,
                reduce_mean=True,
                parallel_context=self.parallel_context,
            )
        return loss


class _VocabParallelCrossEntropy2D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx,
        vocab_parallel_logits: Tensor,
        targets: Tensor,
        parallel_context: ParallelContext,
    ):
        # logits: [b/q, h/q]
        # labels: [b/q]
        # loss: [b/q]
        # vocab_parallel_logits: [b/q, s, v/q]
        # target: [b/q, s]
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        logits_max = all_reduce(
            logits_max,
            op=ReduceOp.MAX,
            parallel_context=parallel_context,
            parallel_mode=ParallelMode.TENSOR_2D,
        )
        # Subtract the maximum value.
        vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

        # Get the partition's vocab indices
        partition_vocab_size = vocab_parallel_logits.size(-1)
        rank = parallel_context.get_local_rank(ParallelMode.TENSOR_2D_ROW)
        vocab_start = rank * partition_vocab_size
        vocab_end = (rank + 1) * partition_vocab_size - 1

        target_mask = (targets < vocab_start) | (targets > vocab_end)
        masked_target = targets.clone() - vocab_start
        masked_target[target_mask] = 0

        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(
            start=0,
            end=logits_2d.size(0),
            device=logits_2d.device,
        )

        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(targets)
        predicted_logits[target_mask] = 0.0

        predicted_logits = all_reduce(
            predicted_logits,
            parallel_context=parallel_context,
            parallel_mode=ParallelMode.TENSOR_2D_ROW,
        )

        exp_logits = torch.exp(vocab_parallel_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        sum_exp_logits = all_reduce(
            sum_exp_logits,
            parallel_context=parallel_context,
            parallel_mode=ParallelMode.TENSOR_2D_ROW,
        )

        loss = torch.log(sum_exp_logits) - predicted_logits

        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad):
        # Retrieve tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as their gradient.
        grad_input = softmax

        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size(-1)
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size(0), device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).float()

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(output_grad.unsqueeze(dim=-1))

        return grad_input, None, None


class VocabParallelCrossEntropyLoss2D(_Loss):
    """Vocab parallel cross entropy loss for 2D parallelism.
    Args:
        reduce_mean (bool, optional): whether to average the loss, defaults to True.
    """

    def __init__(
        self,
        reduce_mean: bool = True,
        ignore_index: int = -100,
        parallel_context: Optional[ParallelContext] = None,
    ):
        super().__init__()
        self.reduce_mean = reduce_mean
        self.ignore_index = ignore_index
        self.parallel_context = parallel_context

    def forward(self, logits: Tensor, targets: Tensor):
        from oslo.torch.nn.parallel.tensor_parallel._2d._ops import (
            split_batch_2d,
            reduce_by_batch_2d,
        )

        """Calculate loss between logits and targets.
        Args:
            logits (:class:`torch.tensor`): Predicted unnormalized scores (often referred to as logits).
            targets (:class:`torch.tensor`): Ground truth class indices or class probabilities.
        """
        targets = split_batch_2d(targets, dim=0, parallel_context=self.parallel_context)
        loss = _VocabParallelCrossEntropy2D.apply(
            logits,
            targets,
            self.parallel_context,
        )
        loss[targets == self.ignore_index] = 0.0
        if self.reduce_mean:
            loss = loss.sum() / (targets != self.ignore_index).sum()
            loss = reduce_by_batch_2d(
                loss, reduce_mean=True, parallel_context=self.parallel_context
            )
        return loss


class CrossEntropyLoss2p5D(_Loss):
    r"""Cross entropy loss for 2.5D parallelism
    Args:
        reduce_mean (bool, optional): whether to average the loss, defaults to True.
    The ``args`` and ``kwargs`` should include parameters below:
    ::
        weight (Tensor, optional)
        size_average (bool, optional)
        ignore_index (int, optional)
        reduce (bool, optional)
        label_smoothing (float, optional)
    More details about ``args``, ``kwargs`` and ``torch.nn.functional.cross_entropy`` could be found in
    `Cross_entropy <https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy>`_.
    """

    def __init__(
        self,
        reduce_mean=True,
        parallel_context: Optional[ParallelContext] = None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.reduce_mean = reduce_mean
        self.parallel_context = parallel_context
        self.loss_args = args
        self.loss_kwargs = kwargs

    def forward(self, logits: Tensor, targets: Tensor):
        from oslo.torch.nn.parallel.tensor_parallel._2p5d._ops import (
            split_batch_2p5d,
            reduce_by_batch_2p5d,
        )

        """Calculate loss between logits and targets.
        Args:
            logits (:class:`torch.tensor`): Predicted unnormalized scores (often referred to as logits).
            targets (:class:`torch.tensor`): Ground truth class indices or class probabilities.
        """
        targets = split_batch_2p5d(
            targets, dim=0, parallel_context=self.parallel_context
        )
        loss = cross_entropy(logits, targets, *self.loss_args, **self.loss_kwargs)
        if self.reduce_mean:
            loss = loss.mean()
            loss = reduce_by_batch_2p5d(
                loss, reduce_mean=True, parallel_context=self.parallel_context
            )
        return loss


class _VocabParallelCrossEntropy2p5D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx,
        vocab_parallel_logits: Tensor,
        targets: Tensor,
        parallel_context: ParallelContext,
    ):
        # logits: [b/dq, h/q]
        # loss: [b/dq]
        # targets: [b/dq, h/q]
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        logits_max = all_reduce(
            logits_max,
            op=ReduceOp.MAX,
            parallel_context=parallel_context,
            parallel_mode=ParallelMode.TENSOR_2P5D_ROW,
        )
        # Subtract the maximum value.
        vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

        partition_vocab_size = vocab_parallel_logits.size(-1)
        rank = parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
        vocab_start = rank * partition_vocab_size
        vocab_end = (rank + 1) * partition_vocab_size - 1

        target_mask = (targets < vocab_start) | (targets > vocab_end)
        masked_target = targets.clone() - vocab_start
        masked_target[target_mask] = 0

        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(
            start=0,
            end=logits_2d.size(0),
            device=logits_2d.device,
        )

        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(targets)
        predicted_logits[target_mask] = 0.0

        predicted_logits = all_reduce(
            predicted_logits,
            parallel_context=parallel_context,
            parallel_mode=ParallelMode.TENSOR_2P5D_ROW,
        )

        exp_logits = torch.exp(vocab_parallel_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        sum_exp_logits = all_reduce(
            sum_exp_logits,
            parallel_context=parallel_context,
            parallel_mode=ParallelMode.TENSOR_2P5D_ROW,
        )

        loss = torch.log(sum_exp_logits) - predicted_logits

        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad):
        # Retrieve tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as their gradient.
        grad_input = softmax

        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size(-1)
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size(0), device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).float()

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(output_grad.unsqueeze(dim=-1))

        return grad_input, None, None


class VocabParallelCrossEntropyLoss2p5D(_Loss):
    """
    Vocab parallel cross entropy loss for 2.5D parallelism
    Args:
        reduction (bool, optional): whether to average the loss, defaults to True.
    """

    def __init__(
        self,
        reduce_mean: bool = True,
        ignore_index: int = -100,
        parallel_context: Optional[ParallelContext] = None,
    ):
        super().__init__()
        self.reduce_mean = reduce_mean
        self.ignore_index = ignore_index
        self.parallel_context = parallel_context

    def forward(self, logits: Tensor, targets: Tensor):
        from oslo.torch.nn.parallel.tensor_parallel._2p5d._ops import (
            split_batch_2p5d,
            reduce_by_batch_2p5d,
        )

        """Calculate loss between logits and targets.
        Args:
            logits (:class:`torch.tensor`): Predicted unnormalized scores (often referred to as logits).
            targets (:class:`torch.tensor`): Ground truth class indices or class probabilities.
        """
        targets = split_batch_2p5d(
            targets, dim=0, parallel_context=self.parallel_context
        )
        loss = _VocabParallelCrossEntropy2p5D.apply(
            logits, targets, self.parallel_context
        )
        loss[targets == self.ignore_index] = 0.0
        if self.reduce_mean:
            loss = loss.sum() / (targets != self.ignore_index).sum()
            loss = reduce_by_batch_2p5d(
                loss, reduce_mean=True, parallel_context=self.parallel_context
            )
        return loss


class CrossEntropyLoss3D(_Loss):
    r"""Cross entropy loss for 3D parallelism.
    Args:
        reduction (bool, optional): whether to average the loss, defaults to True.
    The ``args`` and ``kwargs`` should include parameters below:
    ::
        weight (Tensor, optional)
        size_average (bool, optional)
        ignore_index (int, optional)
        reduce (bool, optional)
        label_smoothing (float, optional)
    More details about ``args``, ``kwargs`` and ``torch.nn.functional.cross_entropy`` could be found in
    `Cross_entropy <https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy>`_.
    """

    def __init__(
        self,
        reduce_mean=True,
        parallel_context: Optional[ParallelContext] = None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.reduce_mean = reduce_mean
        self.parallel_context = parallel_context
        self.loss_args = args
        self.loss_kwargs = kwargs

    def forward(self, logits: Tensor, targets: Tensor):
        from oslo.torch.nn.parallel.tensor_parallel._3d._ops import (
            split_tensor_3d,
            reduce_by_batch_3d,
        )

        """Calculate loss between logits and targets.
        Args:
            logits (:class:`torch.tensor`): Predicted unnormalized scores (often referred to as logits).
            targets (:class:`torch.tensor`): Ground truth class indices or class probabilities.
        """
        targets = split_tensor_3d(
            targets,
            dim=0,
            parallel_context=self.parallel_context,
            parallel_mode=ParallelMode.TENSOR_3D_WEIGHT,
        )
        targets = split_tensor_3d(
            targets,
            dim=0,
            parallel_context=self.parallel_context,
            parallel_mode=ParallelMode.TENSOR_3D_INPUT,
        )
        loss = cross_entropy(
            logits, targets, reduction="none", *self.loss_args, **self.loss_kwargs
        )
        if self.reduce_mean:
            loss = loss.mean()
            loss = reduce_by_batch_3d(
                loss,
                reduce_mean=True,
                parallel_context=self.parallel_context,
                input_parallel_mode=ParallelMode.TENSOR_3D_INPUT,
                weight_parallel_mode=ParallelMode.TENSOR_3D_WEIGHT,
            )
        return loss


class _VocabParallelCrossEntropy3D(torch.autograd.Function):
    # Adapted from megatron.mpu.cross_entropy
    # loss[i] = -logits[i][targets] + log(sum(exp(logits[i])))

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx,
        vocab_parallel_logits: Tensor,
        targets: Tensor,
        parallel_context: ParallelContext,
    ):
        # logits: [b/q^2, c/q]
        # labels: [b/q^2]
        # loss: [b/q^2]
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        logits_max = all_reduce(
            logits_max,
            op=ReduceOp.MAX,
            parallel_context=parallel_context,
            parallel_mode=ParallelMode.TENSOR_3D_OUTPUT,
        )
        # Subtract the maximum value.
        vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

        partition_vocab_size = vocab_parallel_logits.size(-1)
        rank = parallel_context.get_local_rank(ParallelMode.TENSOR_3D_OUTPUT)
        vocab_start = rank * partition_vocab_size
        vocab_end = (rank + 1) * partition_vocab_size - 1

        target_mask = (targets < vocab_start) | (targets > vocab_end)
        masked_target = targets.clone() - vocab_start
        masked_target[target_mask] = 0

        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(
            start=0,
            end=logits_2d.size(0),
            device=logits_2d.device,
        )

        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(targets)
        predicted_logits[target_mask] = 0.0

        predicted_logits = all_reduce(
            predicted_logits,
            parallel_context=parallel_context,
            parallel_mode=ParallelMode.TENSOR_3D_OUTPUT,
        )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        exp_logits = torch.exp(vocab_parallel_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        sum_exp_logits = all_reduce(
            sum_exp_logits,
            parallel_context=parallel_context,
            parallel_mode=ParallelMode.TENSOR_3D_OUTPUT,
        )

        loss = torch.log(sum_exp_logits) - predicted_logits

        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad):
        # Retrieve tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as their gradient.
        input_grad = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size(-1)
        grad_2d = input_grad.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size(0), device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).float()
        input_grad.mul_(output_grad.unsqueeze(dim=-1))

        return input_grad, None, None


class VocabParallelCrossEntropyLoss3D(_Loss):
    """Vocab parallel cross entropy loss for 2D parallelism.
    Args:
        reduction (bool, optional): whether to average the loss, defaults to True.
    """

    def __init__(
        self,
        reduce_mean: bool = True,
        ignore_index: int = -100,
        parallel_context: Optional[ParallelContext] = None,
    ):
        super().__init__()
        self.reduce_mean = reduce_mean
        self.ignore_index = ignore_index
        self.parallel_context = parallel_context

    def forward(self, logits: Tensor, targets: Tensor):
        from oslo.torch.nn.parallel.tensor_parallel._3d._ops import (
            split_tensor_3d,
            reduce_by_batch_3d,
        )

        """Calculate loss between logits and targets.
        Args:
            logits (:class:`torch.tensor`): Predicted unnormalized scores (often referred to as logits).
            targets (:class:`torch.tensor`): Ground truth class indices or class probabilities.
        """
        targets = split_tensor_3d(
            targets,
            dim=0,
            parallel_context=self.parallel_context,
            parallel_mode=ParallelMode.TENSOR_3D_WEIGHT,
        )
        targets = split_tensor_3d(
            targets,
            dim=0,
            parallel_context=self.parallel_context,
            parallel_mode=ParallelMode.TENSOR_3D_INPUT,
        )
        loss = _VocabParallelCrossEntropy3D.apply(
            logits, targets, self.parallel_context
        )
        loss[targets == self.ignore_index] = 0.0
        if self.reduce_mean:
            loss = loss.sum() / (targets != self.ignore_index).sum()
            loss = reduce_by_batch_3d(
                loss,
                reduce_mean=True,
                parallel_context=self.parallel_context,
                input_parallel_mode=ParallelMode.TENSOR_3D_INPUT,
                weight_parallel_mode=ParallelMode.TENSOR_3D_WEIGHT,
            )
        return loss
