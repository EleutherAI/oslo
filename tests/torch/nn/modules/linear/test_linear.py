"""
python3 test_linear.py
"""
import torch
from torch.nn import Linear

import oslo.torch.nn as onn


def test_linear():
    torch_linear = Linear(10, 10).cuda()
    onn_linear = onn.Linear(10, 10).cuda()
    onn_linear_skip = onn.Linear(10, 10, skip_bias_add=True).cuda()

    # make sure the parameters are the same
    onn_linear.load_state_dict(torch_linear.state_dict())
    onn_linear_skip.load_state_dict(torch_linear.state_dict())

    print("> Test weight shape...", end="")
    assert torch_linear.weight.shape == onn_linear.weight.shape
    print("OK")

    print("> Test bias shape...", end="")
    assert torch_linear.bias.shape == onn_linear.bias.shape
    print("OK")

    print("> Test forward...", end="")
    input_tensor = torch.randn(1, 10, 10).cuda()
    assert torch.allclose(torch_linear(input_tensor), onn_linear(input_tensor))
    print("OK")

    print("> Test forward skip bias add...", end="")
    input_tensor = torch.randn(1, 10, 10).cuda()
    torch_output = torch_linear(input_tensor)
    onn_output, bias = onn_linear_skip(input_tensor)
    onn_output += bias
    assert torch.allclose(torch_output, onn_output)
    print("OK")


if __name__ == "__main__":
    print("Test tests/torch/nn/modules/linear/test_linear.py")
    test_linear()
