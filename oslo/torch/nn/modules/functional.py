import torch
from torch import Tensor
from torch.nn import functional as F

from oslo.torch._C import (
    get_softmax_kernel,
    get_layernorm_kernel,
    get_ngram_repeat_block_kernel,
)

"""
Autograd Functions
"""


# Utils from apex
def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        return torch.cuda.amp.autocast_mode._cast(args, torch.get_autocast_gpu_dtype())


@torch.jit.script
def _fused_gelu_fwb(x):
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


@torch.jit.script
def _fused_gelu_bwd(g, x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    return ff * g


@torch.jit.script
def _fused_bias_gelu_fwb(y, bias):
    x = y + bias
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


@torch.jit.script
def _fused_bias_gelu_bwd(g, y, bias):
    x = y + bias
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    return ff * g


class _FusedGeLUFunction(torch.autograd.Function):
    """
    Kernel fusion function: GeLU
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return _fused_gelu_fwb(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        tmp = _fused_gelu_bwd(grad_output, input)
        return tmp, tmp


class _FusedBiasGeLUFunction(torch.autograd.Function):
    """
    Kernel fusion function: Bias + GeLU
    """

    @staticmethod
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return _fused_bias_gelu_fwb(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = _fused_bias_gelu_bwd(grad_output, input, bias)
        return tmp, tmp


class _FusedLayerNormAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        bias_ = bias.contiguous()
        output, mean, invvar = get_layernorm_kernel().layer_norm_forward_affine(
            input_, ctx.normalized_shape, weight_, bias_, ctx.eps
        )
        ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight_, bias_, mean, invvar = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        (
            grad_input,
            grad_weight,
            grad_bias,
        ) = get_layernorm_kernel().layer_norm_backward_affine(
            grad_output.contiguous(),
            mean,
            invvar,
            input_,
            ctx.normalized_shape,
            weight_,
            bias_,
            ctx.eps,
        )
        return grad_input, grad_weight, grad_bias, None, None


class _FusedRMSNormAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, normalized_shape, eps):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        output, invvar = get_layernorm_kernel().rms_norm_forward_affine(
            input_, ctx.normalized_shape, weight_, ctx.eps
        )
        ctx.save_for_backward(input_, weight_, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight_, invvar = ctx.saved_tensors
        grad_input = grad_weight = None
        grad_input, grad_weight = get_layernorm_kernel().rms_norm_backward_affine(
            grad_output.contiguous(),
            invvar,
            input_,
            ctx.normalized_shape,
            weight_,
            ctx.eps,
        )
        return grad_input, grad_weight, None, None


class _FusedLayerNormAffineMixedDtypesFunction(_FusedLayerNormAffineFunction):
    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        bias_ = bias.contiguous()
        (
            output,
            mean,
            invvar,
        ) = get_layernorm_kernel().layer_norm_forward_affine_mixed_dtypes(
            input_, ctx.normalized_shape, weight_, bias_, ctx.eps
        )
        ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
        return output


class _FusedRMSNormAffineMixedDtypesFunction(_FusedRMSNormAffineFunction):
    @staticmethod
    def forward(ctx, input, weight, normalized_shape, eps):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        output, invvar = get_layernorm_kernel().rms_norm_forward_affine_mixed_dtypes(
            input_, ctx.normalized_shape, weight_, ctx.eps
        )

        ctx.save_for_backward(input_, weight_, invvar)
        return output


class _FusedLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, eps):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        output, mean, invvar = get_layernorm_kernel().layer_norm_forward(
            input_, ctx.normalized_shape, ctx.eps
        )
        ctx.save_for_backward(input_, mean, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, mean, invvar = ctx.saved_tensors
        grad_input = None
        grad_input = get_layernorm_kernel().layer_norm_backward(
            grad_output.contiguous(),
            mean,
            invvar,
            input_,
            ctx.normalized_shape,
            ctx.eps,
        )
        return grad_input, None, None


class _FusedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, eps):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        output, invvar = get_layernorm_kernel().rms_norm_forward(
            input_, ctx.normalized_shape, ctx.eps
        )
        ctx.save_for_backward(input_, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, invvar = ctx.saved_tensors
        grad_input = None
        grad_input = get_layernorm_kernel().rms_norm_backward(
            grad_output.contiguous(), invvar, input_, ctx.normalized_shape, ctx.eps
        )
        return grad_input, None, None


class _FusedScaleUpperTriangMaskSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, scale):
        scale_t = torch.tensor([scale])
        softmax_results = (
            get_softmax_kernel().fused_scaled_upper_triang_masked_softmax_forward(
                inputs, scale_t[0]
            )
        )

        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors
        input_grads = (
            get_softmax_kernel().fused_scaled_upper_triang_masked_softmax_backward(
                output_grads, softmax_results, scale_t[0]
            )
        )

        return input_grads, None


class _FusedScaleMaskSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, mask, scale):
        scale_t = torch.tensor([scale])

        softmax_results = get_softmax_kernel().fused_scaled_masked_softmax_forward(
            inputs, mask, scale_t[0]
        )
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors

        input_grads = get_softmax_kernel().fused_scaled_masked_softmax_backward(
            output_grads, softmax_results, scale_t[0]
        )
        return input_grads, None, None


class _NGramRepeatBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, lprobs, bsz, step, beam_size, no_repeat_ngram_size):
        return get_ngram_repeat_block_kernel().forward(
            tokens, lprobs, bsz, step, beam_size, no_repeat_ngram_size
        )

    def backward(*args):
        raise NotImplementedError


"""
User Functions
"""


def fused_layer_norm_affine(input, weight, bias, normalized_shape, eps=1e-6):
    args = _cast_if_autocast_enabled(input, weight, bias, normalized_shape, eps)
    with torch.cuda.amp.autocast(enabled=False):
        return _FusedLayerNormAffineFunction.apply(*args)


def fused_layer_norm(input, normalized_shape, eps=1e-6):
    args = _cast_if_autocast_enabled(input, normalized_shape, eps)
    with torch.cuda.amp.autocast(enabled=False):
        return _FusedLayerNormFunction.apply(*args)


def mixed_dtype_fused_layer_norm_affine(
    input, weight, bias, normalized_shape, eps=1e-6
):
    args = _cast_if_autocast_enabled(input, weight, bias, normalized_shape, eps)
    with torch.cuda.amp.autocast(enabled=False):
        return _FusedLayerNormAffineMixedDtypesFunction.apply(*args)


def fused_rms_norm_affine(input, weight, normalized_shape, eps=1e-6):
    args = _cast_if_autocast_enabled(input, weight, normalized_shape, eps)
    with torch.cuda.amp.autocast(enabled=False):
        return _FusedRMSNormAffineFunction.apply(*args)


def fused_rms_norm(input, normalized_shape, eps=1e-6):
    args = _cast_if_autocast_enabled(input, normalized_shape, eps)
    with torch.cuda.amp.autocast(enabled=False):
        return _FusedRMSNormFunction.apply(*args)


def mixed_dtype_fused_rms_norm_affine(input, weight, normalized_shape, eps=1e-6):
    args = _cast_if_autocast_enabled(input, weight, normalized_shape, eps)
    with torch.cuda.amp.autocast(enabled=False):
        return _FusedRMSNormAffineMixedDtypesFunction.apply(*args)


def fused_gelu(x):
    return _FusedGeLUFunction.apply(x)


def fused_bias_gelu(x, bias):
    return _FusedBiasGeLUFunction.apply(x, bias)


@torch.jit.script
def fused_bias_dropout(x, bias, p, training, inplace):
    # type: (Tensor, Tensor, float, bool, bool) -> Tensor
    return F.dropout(x + bias, p=p, training=training, inplace=inplace)


def _fused_scale_mask_softmax_sanity_check(input, scale, softmax_in_fp32):
    assert input.dim() == 4, "input must be be `(batch, nhead, len_q, len_k)`."
    assert scale is not None, "scale must not be None."
    assert scale == 1.0 or softmax_in_fp32, "softmax should be in fp32 when scaled"


def _is_fused_scale_mask_softmax_available(
    input, scale, softmax_in_fp32, use_triang_mask
):
    bsz, np, sq, sk = input.size()
    dtype = input.dtype
    _fused_scale_mask_softmax_sanity_check(input, scale, softmax_in_fp32)

    if dtype != torch.float16 and dtype != torch.bfloat16:
        return False

    if sk > 4096 or sk <= 0:
        return False

    if softmax_in_fp32 is True:
        return False

    bsz_per_block = get_softmax_kernel().get_batch_per_block(sq, sk, bsz, np)

    if use_triang_mask:
        if sq == sk and (sk <= 64 or sk % 4 == 0) and (bsz * np) % bsz_per_block == 0:
            return True
    else:
        if sq > 1 and sq % bsz_per_block == 0:
            return True

    return False


def _fused_scale_mask_softmax_torch(input, scale, softmax_in_fp32, mask):
    original_input_dtype = input.dtype
    _fused_scale_mask_softmax_sanity_check(input, scale, softmax_in_fp32)

    if softmax_in_fp32 and original_input_dtype != torch.float32:
        input = input.float()

    input = input * scale
    mask_output = input.masked_fill_(mask, -10000.0) if mask is not None else input
    probs = torch.nn.Softmax(dim=-1)(mask_output)

    if softmax_in_fp32 and original_input_dtype != torch.float32:
        if original_input_dtype == torch.float16:
            probs = probs.half()
        else:
            probs = probs.bfloat16()

    return probs


def _fused_scale_mask_softmax_cuda(input, scale, use_triang_mask, pad_mask):
    bsz, np, sq, sk = input.size()
    _fused_scale_mask_softmax_sanity_check(input, scale, softmax_in_fp32=False)

    if use_triang_mask:
        if pad_mask is not None:
            input += pad_mask
        output = _FusedScaleUpperTriangMaskSoftmaxFunction.apply(
            input.view(-1, sq, sk),
            scale,
        )
        return output.view(bsz, np, sq, sk)
    else:
        if pad_mask is not None:
            if pad_mask.size(2) == 1:
                pad_mask = pad_mask.repeat(1, 1, sq, 1)
            return _FusedScaleMaskSoftmaxFunction.apply(
                input,
                pad_mask.bool(),
                scale,
            )
        else:
            pad_mask = torch.zeros(1, 1, sq, sk, device=input.device, dtype=input.dtype)
            return _FusedScaleMaskSoftmaxFunction.apply(
                input,
                pad_mask.bool(),
                scale,
            )


def fused_scale_mask_softmax(
    input, scale, use_triang_mask, softmax_in_fp32, pad_mask=None
):
    scale = scale if scale is not None else 1.0
    if _is_fused_scale_mask_softmax_available(
        input, scale, softmax_in_fp32, use_triang_mask
    ):
        return _fused_scale_mask_softmax_cuda(input, scale, use_triang_mask, pad_mask)
    else:
        return _fused_scale_mask_softmax_torch(input, scale, softmax_in_fp32, pad_mask)
