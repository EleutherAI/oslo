import torch

from oslo.extension.training.ops.pytorch.multihead_attention_layer import LSMultiheadAttentionLayer

import time
import pytest


def test_multihead_attention_layer():
    """Test the multihead attention layer layer."""
    # Create a multihead attention layer layer.
    config = LSMultiheadAttentionLayer.get_config(
            max_batch_tokens=3,
            max_seq_len=16,
            hidden_size=128,  # size of transformer hidden layers
            nhead=8,  # number of heads in attention
            intermediate_size=1, # size of intermediate layer
            attn_prob_dropout_ratio=0.2,  # attention score dropout ratio
            activation_dropout_ratio=0.2,
            hidden_dropout_ratio=0.2,  # dropout ration before residual
            pre_or_postLayerNorm=False,  # pre layer norm or post
            activation_fn='relu',  # relu or gelu
            mask_future_tokens=False, # mask future tokens
            is_post_ln=False,  # post layer norm
            fp16=False,  # fp16 presion
            local_rank=0,  # rank in local node
            quant_mode=False
        )
    
    ls_multihead_attention_layer = LSMultiheadAttentionLayer(config)
    ls_multihead_attention_layer.cuda()

    # Create some dummy data.
    y_pred = torch.tensor([[[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]]).cuda()
    y_true = torch.tensor([[2, 1]]).cuda()

    output = ls_multihead_attention_layer(y_pred, y_true)
    print(output)


if __name__ == "__main__":
    print("Test tests/extension/test_multihead_attention_layer.py", end="")
    test_multihead_attention_layer()
    print("OK")