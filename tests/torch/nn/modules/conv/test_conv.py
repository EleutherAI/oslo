"""
python3 test_conv.py
"""
import torch
from transformers.modeling_utils import Conv1D

import oslo.torch.nn as onn


def test_conv1d():
    transformers_conv = Conv1D(10, 10).cuda()
    onn_conv = onn.Conv1D(10, 10).cuda()
    onn_conv_skip = onn.Conv1D(10, 10, skip_bias_add=True).cuda()

    # make sure the parameters are the same
    onn_conv.load_state_dict(transformers_conv.state_dict())
    onn_conv_skip.load_state_dict(transformers_conv.state_dict())

    print("> Test weight shape...", end="")
    assert transformers_conv.weight.shape == onn_conv.weight.shape
    print("OK")

    print("> Test bias shape...", end="")
    assert transformers_conv.bias.shape == onn_conv.bias.shape
    print("OK")

    print("> Test forward...", end="")
    input_tensor = torch.randn(1, 10, 10).cuda()
    assert torch.allclose(transformers_conv(input_tensor), onn_conv(input_tensor))
    print("OK")

    print("> Test forward skip bias add...", end="")
    input_tensor = torch.randn(1, 10, 10).cuda()
    transformers_output = transformers_conv(input_tensor)
    onn_output, bias = onn_conv_skip(input_tensor)
    onn_output += bias
    assert torch.allclose(transformers_output, onn_output)
    print("OK")


if __name__ == "__main__":
    print("Test tests/torch/nn/modules/conv/test_conv.py")
    test_conv1d()
