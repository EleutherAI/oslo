"""
python3 test_dropout.py
"""
import torch

import oslo.torch.nn as onn


def test_fused_bias_dropout():
    # create linear layers
    onn_linear = onn.Linear(10, 10, skip_bias_add=True).cuda()
    torch_linear = torch.nn.Linear(10, 10).cuda()

    # make sure the parameters are the same
    torch_linear.load_state_dict(onn_linear.state_dict())

    # create dropout layers
    onn_dropout = onn.FusedBiasDropout(0.2).cuda()
    torch_dropout = torch.nn.Dropout(0.2).cuda()

    # create input
    input_tensor = torch.randn(1, 10, requires_grad=True).cuda()

    # forward pass
    onn_output = onn_dropout(*onn_linear(input_tensor)).squeeze()
    torch_output = torch_dropout(torch_linear(input_tensor)).squeeze()

    print("> Test forward...", end="")
    for o1, o2 in zip(onn_output, torch_output):
        if o1 != 0 and o2 != 0:
            assert torch.allclose(o1, o2)
    print("OK")


if __name__ == "__main__":
    print("Test tests/torch/nn/modules/dropout/test_dropout.py")
    test_fused_bias_dropout()
