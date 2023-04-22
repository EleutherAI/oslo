from dataclasses import dataclass
from itertools import zip_longest
import math
import copy

import torch
from torch import nn
from torch.autograd import Function

from oslo.extension.training.ops.pytorch.builder import LayerBuilder
from oslo.extension.training.ops.pytorch.quantization import (
    weight_quant_config,
    act_quant_config,
    relu_quant_config,
)
from oslo.extension.training.ops.pytorch.util import (
    copy_para,
    state_dict,
    calc_offset,
)

from transformers import PretrainedConfig



layer_cuda_module = None


_all_layer_grads = dict()


class LSMultiheadAttentionFunc(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        input_mask,
        parameters,
        config,
    ):
        cuda_module = layer_cuda_module
        forward_func = (
            cuda_module.multihead_attention_layer_fw_fp16
            if config.fp16
            else cuda_module.multihead_attention_layer_fw_fp32
        )
        if config.fp16:
            input = input.to(torch.half)
            input_mask = input_mask.to(torch.half)

        (output,) = forward_func(
            config.layer_id,
            input,
            input_mask,
            config.training)

        if config.is_grad_enabled and config.training:
            ctx.save_for_backward(output, input, input_mask)
            ctx.config = config
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert ctx.config.training

        cuda_module = layer_cuda_module
        backward_func = (
            cuda_module.multihead_attention_layer_bw_fp16
            if ctx.config.fp16
            else cuda_module.multihead_attention_layer_bw_fp32
        )

        output, input, input_mask = ctx.saved_tensors
        if ctx.config.fp16:
            grad_output = grad_output.to(torch.half)
            output = output.to(torch.half)
            input = input.to(torch.half)
            input_mask = input_mask.to(torch.half)
        (grad_input,) = backward_func(
            ctx.config.layer_id, grad_output, output, input, input_mask
        )

        grad = _all_layer_grads[ctx.config.layer_id]

        return (grad_input, None, grad, None)


class LSMultiheadAttentionLayer(nn.Module):
    """Initialize the Lightseq Cross Entropy Layer.

    Static variable:
        layer_id: The layer-index counter starting from 0 and incrementing by 1 every time a layer object is instantiated,
    Arguments:
        config: An object of LSMultiheadAttentionLayer config, see get_config
    """

    layer_id = 0

    def __init__(
        self,
        config,
        initial_weights=None, 
        initial_biases=None
    ):
        super(LSMultiheadAttentionLayer, self).__init__()

        self.config = copy.deepcopy(config)
        if isinstance(self.config, PretrainedConfig):
            self.config.max_batch_tokens = 4096 # 일단 default batch size로 128로 설정
            self.config.activation_dropout_ratio = self.config.hidden_dropout_prob  # 이 둘이 같다고 가정
            self.config.attention_probs_dropout_prob = self.config.hidden_dropout_prob
            self.config.fp16 = False
            self.config.local_rank = 0
            self.config.pre_or_postLayerNorm = False
            self.config.mask_future_tokens = False
            self.config.is_post_ln = False

        self.config.layer_id = LSMultiheadAttentionLayer.layer_id
        LSMultiheadAttentionLayer.layer_id += 1

        self.quant_mode = False

        if self.config.local_rank is not None and self.config.local_rank >= 0:
            torch.cuda.set_device(self.config.local_rank)

        # Load cuda modules if needed
        global layer_cuda_module
        if layer_cuda_module is None:
            layer_cuda_module = LayerBuilder().load()

        # create the layer in cuda kernels.
        cuda_module = layer_cuda_module
        create_layer_func = (
            cuda_module.create_multihead_attention_layer_new_fp16
            if self.config.fp16
            else cuda_module.create_multihead_attention_layer_new_fp32
        )

        # int layer_id, int max_batch_tokens, int max_position_embeddings, int hidden_size,
        # int num_heads, int intermediate_size, float attention_probs_dropout_prob,
        # float activation_dropout_ratio, float hidden_dropout_prob,
        # bool pre_or_postLayerNorm, std::string hidden_act,
        # bool mask_future_tokens, bool is_post_ln
        create_layer_func(
            self.config.layer_id,
            self.config.max_batch_tokens,
            self.config.max_position_embeddings,
            self.config.hidden_size,
            self.config.num_attention_heads,
            self.config.intermediate_size,
            self.config.attention_probs_dropout_prob,
            self.config.activation_dropout_ratio,
            self.config.hidden_dropout_prob,
            self.config.pre_or_postLayerNorm,
            self.config.hidden_act,
            self.config.mask_future_tokens,
            self.config.is_post_ln,
        )
        self.assigned_layer_weight_grad = False

        hs = self.config.hidden_size
        ims = self.config.intermediate_size
        self.hs = hs
        self.ims = ims
        self.para_offset = LSMultiheadAttentionLayer.gen_offset(hs, ims)
        self.para = nn.Parameter(torch.Tensor(self.para_offset[-1]))

        if initial_weights is None or initial_biases is None:
            self.init_transformer_weights()
            return

        # For testing only.
        qkv_w = [ele.detach().clone() for ele in initial_weights[:3]]
        qkv_w = torch.cat(qkv_w, dim=0)
        weights = [qkv_w] + [copy_para(ele) for ele in initial_weights[3:]]

        qkv_b = [ele.detach().clone() for ele in initial_biases[:3]]
        qkv_b = torch.cat(qkv_b, dim=0)
        biases = [qkv_b] + [copy_para(ele) for ele in initial_biases[3:]]

        idx = 0
        for w, b in zip_longest(weights, biases):
            cur_para = self._get_weights(idx)
            assert cur_para.numel() == w.numel()
            cur_para.copy_(w.view(-1))
            idx += 1

            cur_para = self._get_weights(idx)
            assert cur_para.numel() == b.numel()
            cur_para.copy_(b.view(-1))
            idx += 1

    @staticmethod
    def get_config(**kwargs):
        @dataclass
        class Config:
            max_batch_tokens: int  # max batch token numbers
            max_position_embeddings: int  # max sequence length
            hidden_size: int  # size of transformer hidden layers
            num_attention_heads: int  # number of heads in attention
            intermediate_size: int # size of intermediate layer
            attention_probs_dropout_prob: float  # attention score dropout ratio
            activation_dropout_ratio: float # activation dropout ratio
            hidden_dropout_prob: float  # dropout ration before residual
            pre_or_postLayerNorm: bool  # pre layer norm or post
            hidden_act: str  # relu or gelu
            mask_future_tokens: bool  # mask future tokens
            is_post_ln: bool  # post layer norm
            fp16: bool  # fp16 presion
            local_rank: int = 0 # rank in local node
            quant_mode: bool = False
            training: bool = True
            is_grad_enabled: bool = True

        return Config(**kwargs)

    @staticmethod
    def gen_offset(hidden_size, intermediate_size):
        hs, ims = hidden_size, intermediate_size
        sizes = [
            hs * hs * 3, # attn_qkvw
            hs * 3,     # attn_qkvb
            hs * hs,    # attn_ow
            hs,     # attn_ob
            hs,     # attn_nw
            hs,     # attn_nb
        ]
        offsets = calc_offset(sizes)
        return offsets

    def _get_weights(self, i):
        return self.para.data.narrow(
            0, self.para_offset[i], self.para_offset[i + 1] - self.para_offset[i]
        )

    def calc_bound(self, w):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1.0 / math.sqrt(fan_in)
        return bound

    def init_transformer_weights(self):
        hs = self.config.hidden_size
        attn_qkvw = self._get_weights(0).view(3, hs, hs)
        nn.init.normal_(attn_qkvw[0,:,:], mean=0.0, std=self.config.initializer_range)
        nn.init.normal_(attn_qkvw[1,:,:], mean=0.0, std=self.config.initializer_range)
        nn.init.normal_(attn_qkvw[2,:,:], mean=0.0, std=self.config.initializer_range)

        nn.init.zeros_(self._get_weights(1))

        nn.init.normal_(self._get_weights(2).view(-1, hs), mean=0.0, std=self.config.initializer_range)
        nn.init.zeros_(self._get_weights(3))

        nn.init.ones_(self._get_weights(4))
        nn.init.zeros_(self._get_weights(5))

    def params_dict(self):
        """
        Returns:
            weight: dict
            bias: dict
        """

        def copy_and_view(m, shape=None):
            if shape is None:
                shape = (-1,)
            return m.data.clone().view(*shape)

        self_attn_qkvw = self._get_weights(0)
        self_attn_qw, self_attn_kw, self_attn_vw = self_attn_qkvw.split(
            self.hs * self.hs, 0
        )
        self_attn_qkvb = self._get_weights(1)
        self_attn_qb, self_attn_kb, self_attn_vb = self_attn_qkvb.split(self.hs, 0)

        weight = {
            "self_attn.q_proj": copy_and_view(self_attn_qw, (self.hs, self.hs)),
            "self_attn.k_proj": copy_and_view(self_attn_kw, (self.hs, self.hs)),
            "self_attn.v_proj": copy_and_view(self_attn_vw, (self.hs, self.hs)),
            "self_attn.out_proj": copy_and_view(
                self._get_weights(2), (self.hs, self.hs)
            ),
            "self_attn_layer_norm": copy_and_view(self._get_weights(4), (self.hs,)),
        }
        bias = {
            "self_attn.q_proj": copy_and_view(self_attn_qb),
            "self_attn.k_proj": copy_and_view(self_attn_kb),
            "self_attn.v_proj": copy_and_view(self_attn_vb),
            "self_attn.out_proj": copy_and_view(self._get_weights(3)),
            "self_attn_layer_norm": copy_and_view(self._get_weights(5)),
        }
        return weight, bias

    def assign_layer_weight_grad(self):
        if self.assigned_layer_weight_grad == True:
            return
        self.assigned_layer_weight_grad = True
        param = (
            self.para_16
            if self.config.fp16 and self.para.dtype != torch.half
            else self.para
        )
        if self.config.layer_id in _all_layer_grads:
            return
        global layer_cuda_module
        cuda_module = layer_cuda_module
        if self.config.fp16:
            func = cuda_module.assign_layer_weight_grad_fp16
        else:
            func = cuda_module.assign_layer_weight_grad_fp32
        grad = torch.zeros_like(param)
        func(param, grad, "MultiheadAttentionLayer", self.config.layer_id)
        _all_layer_grads[self.config.layer_id] = grad

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        destination = state_dict(
            self, destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        return destination

    def forward(self, hidden_states, encoder_padding_mask, **kwargs):
        # encoder_padding_mask is a mask for the input sequence
        # sizes are [batch_size, seq_len] or [seq_len] when batch_size = 1
        # masked value should be 1.0, unmasked value should be 0.0

        self.config.training = self.training
        self.config.is_grad_enabled = torch.is_grad_enabled()
        hidden_states = hidden_states.contiguous()
        encoder_padding_mask = (
            (encoder_padding_mask * -1e8).type_as(hidden_states).contiguous()
        )
        if self.config.fp16 and self.para.dtype != torch.half:
            if hasattr(self, "para_16"):
                self.para_16.copy_(self.para.to(torch.half))
            else:
                self.register_buffer("para_16", self.para.clone().detach().half())

        self.assign_layer_weight_grad()

        bs, sl, dim = hidden_states.size()
        if bs * sl > self.config.max_batch_tokens:
            raise ValueError(
                f"Batch token numbers {bs * sl} exceeds the limit"
                f" {self.config.max_batch_tokens}."
            )
        if sl > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {sl} exceeds the limit {self.config.max_position_embeddings}."
            )
        if len(encoder_padding_mask.size()) == 1:
            assert bs == 1 and sl == encoder_padding_mask.size(0)
        else:
            encoder_padding_mask = encoder_padding_mask.squeeze()

            assert bs == encoder_padding_mask.size(
                0
            ) and sl == encoder_padding_mask.size(1)
        encoder_padding_mask = torch.nan_to_num(encoder_padding_mask)
        output = LSMultiheadAttentionFunc.apply(
            hidden_states,
            encoder_padding_mask,
            self.para,
            self.config,
        )

        return output.to(self.para)