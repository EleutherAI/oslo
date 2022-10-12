from typing import Any, Tuple, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from oslo.torch._C import get_expert_parallel_kernel


class AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        context: Any,
        group: ProcessGroup,
        inputs: Tensor,
    ) -> Tensor:
        context.comm_group = group
        inputs = inputs.contiguous()
        output = torch.empty_like(inputs)
        dist.all_to_all_single(output, inputs, group=group)

        return output

    @staticmethod
    def backward(context: Any, *grad_outputs: Tensor) -> Tuple[Tensor, None]:
        return (None, AllToAll.apply(context.comm_group, *grad_outputs))


class EPDispatch(torch.autograd.Function):
    @staticmethod
    def forward(context, tokens, mask, dest_idx, ec):
        s = tokens.size(0)
        h = tokens.size(1)

        expert_input = get_expert_parallel_kernel().dispatch_forward(
            s, ec, h, tokens, mask, dest_idx
        )
        context.save_for_backward(mask, dest_idx)
        context.s, context.ec, context.h = s, ec, h

        return expert_input

    @staticmethod
    def backward(context, output_grad):
        mask, dest_idx = context.saved_tensors
        d_tokens = get_expert_parallel_kernel().dispatch_forward(
            context.s, context.ec, context.h, output_grad, mask, dest_idx
        )

        return d_tokens, None, None, None


class EPCombine(torch.autograd.Function):
    @staticmethod
    def forward(context, expert_tokens, logits, mask, dest_idx, ec):
        assert logits.dtype == torch.float32

        s = logits.size(0)
        e = logits.size(1)
        c = ec // e
        h = expert_tokens.size(-1)

        fp16_flag = expert_tokens.dtype == torch.float16
        combine_inp = expert_tokens.to(torch.float32) if fp16_flag else expert_tokens
        ctokens = get_expert_parallel_kernel().combine_forward(
            s, e, c, h, combine_inp, logits, mask, dest_idx
        )
        output = ctokens.to(torch.float16) if fp16_flag else ctokens

        context.save_for_backward(expert_tokens, logits, mask, dest_idx)
        context.s, context.e, context.s, context.h = s, e, c, h
        context.fp16_flag = fp16_flag

        return output

    @staticmethod
    def backward(context, tokens_grad):
        expert_tokens, logits, mask, dest_idx = context.saved_tensors

        combine_grad = (
            tokens_grad.to(torch.float32)
            if tokens_grad.type is torch.float16
            else tokens_grad
        )

        combine_inp = (
            expert_tokens.to(torch.float32) if context.fp16_flag else expert_tokens
        )
        d_expert, d_logits = get_expert_parallel_kernel().combine_backward(
            context.s,
            context.e,
            context.s,
            context.h,
            combine_grad,
            combine_inp,
            logits,
            mask,
            dest_idx,
        )
        d_expert = d_expert.to(torch.float16) if context.fp16_flag else d_expert

        return d_expert, d_logits, None, None, None


class AllReduce:
    def __init__(self, ep_group, ep_size, para_name):
        self.ep_group = ep_group
        self.ep_size = ep_size
        self.para_name = para_name

    def __call__(self, grad):
        grad.mul_(1.0 / self.ep_size)
        dist.all_reduce(grad, group=self.ep_group)
