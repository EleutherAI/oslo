import torch

from oslo.lightseq2.training.ops.pytorch.multihead_attention_layer import LSMultiheadAttentionLayer

import time
import pytest


def test_multihead_attention_layer():
    """Test the multihead attention layer layer."""
    # Create a multihead attention layer layer.
    config = LSMultiheadAttentionLayer.get_config(
            max_batch_tokens=4096,
            max_seq_len=512,
            max_position_embeddings=512,
            num_attention_heads=1,
            attention_probs_dropout_prob=0.1,
            hidden_dropout_prob=0.1,
            hidden_act='gelu',
            hidden_size=8,  # size of transformer hidden layers
            nhead=1,  # number of heads in attention
            intermediate_size=1, # size of intermediate layer
            attn_prob_dropout_ratio=0.1,  # attention score dropout ratio
            activation_dropout_ratio=0.1,
            hidden_dropout_ratio=0.1,  # dropout ration before residual
            pre_or_postLayerNorm=False,  # pre layer norm or post
            activation_fn='gelu',  # relu or gelu
            initializer_range=0.02,
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
    y_true = torch.tensor([[0, 1]]).cuda()

    output = ls_multihead_attention_layer(y_pred, y_true)
    print(output)


if __name__ == "__main__":
    print("Test tests/extension/test_multihead_attention_layer.py", end="")
    test_multihead_attention_layer()
    print("OK")