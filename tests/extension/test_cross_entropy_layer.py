import torch

from oslo.extension.training.ops.pytorch.cross_entropy_layer import LSCrossEntropyLayer

import time
import pytest


def test_cross_entropy_layer():
    """Test the cross entropy layer."""
    # Create a cross entropy layer.
    config = LSCrossEntropyLayer.get_config(
            max_batch_tokens=3,
            padding_idx=10,
            epsilon=0.0,
            fp16=False,
            local_rank="cuda:0",
        )
    
    ls_cross_entropy = LSCrossEntropyLayer(config)

    # Create some dummy data.
    y_pred = torch.tensor([[[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]]).cuda()
    y_true = torch.tensor([[2, 1]]).cuda()

    # Compute the cross entropy loss. 
    start = time.time()
    loss, _ = ls_cross_entropy(y_pred, y_true)  # sum
    print("Evaluated in {} seconds at ls_cross_entropy".format(time.time() - start))

    # Assert that the loss is correct.
    assert loss.item() == pytest.approx(0.9539*2.0, 1e-4)

    start = time.time()
    loss = torch.nn.functional.cross_entropy(y_pred.squeeze(), y_true.squeeze()) # mean
    print("Evaluated in {} seconds at torch.nn.functional.cross_entropy".format(time.time() - start))

    # Assert that the loss is correct.
    assert loss.item() == pytest.approx(0.9539, 1e-4)


if __name__ == "__main__":
    print("Test tests/extension/test_cross_entropy_layer.py", end="")
    test_cross_entropy_layer()
    print("OK")