import torch

from oslo.extension.training.ops.pytorch.multihead_attention_layer import LSMultiheadAttentionLayer

import time
import pytest


def test_multihead_attention_layer():
    """Test the multihead attention layer layer."""
    # Create a multihead attention layer layer.
    config = LSMultiheadAttentionLayer.get_config(
            max_batch_tokens=3,
            padding_idx=10,
            epsilon=0.0,
            fp16=False,
            local_rank=0,
        )
    
    ls_multihead_attention_layer = LSMultiheadAttentionLayer(config)

    # Create some dummy data.
    y_pred = torch.tensor([[[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]]).cuda()
    y_true = torch.tensor([[2, 1]]).cuda()

    output = ls_multihead_attention_layer(y_pred, y_true)
    print(output)


if __name__ == "__main__":
    print("Test tests/extension/test_multihead_attention_layer.py", end="")
    test_multihead_attention_layer()
    print("OK")