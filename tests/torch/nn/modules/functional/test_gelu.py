"""
python3 test_gelu.py
"""
import torch
from torch.nn import functional as F
from oslo.torch.nn import fused_gelu, fused_bias_gelu


def test_gelu():
    input_fused = torch.randn(4, 4, requires_grad=True).cuda()
    input_fused.retain_grad()
    input_non_fused = input_fused.clone().detach().requires_grad_(True)
    input_non_fused.retain_grad()

    output_fused = fused_gelu(input_fused)
    output_non_fused = F.gelu(input_non_fused)

    print("> Test forward...", end="")
    assert torch.allclose(output_fused, output_non_fused, atol=1e-2)
    print("OK")

    print("> Test backward...", end="")
    output_non_fused.sum().backward()
    output_fused.sum().backward()
    assert torch.allclose(input_fused.grad, input_non_fused.grad, atol=1e-2)
    print("OK")

    input_fused = torch.randn(4, 4, requires_grad=True).cuda()
    input_fused.retain_grad()
    input_bias_fused = torch.randn(4, 4, requires_grad=True).cuda()
    input_bias_fused.retain_grad()

    input_non_fused = input_fused.clone().detach().requires_grad_(True)
    input_non_fused.retain_grad()
    input_bias_non_fused = input_bias_fused.clone().detach().requires_grad_(True)
    input_bias_non_fused.retain_grad()

    output_fused = fused_bias_gelu(input_fused, input_bias_fused)
    output_non_fused = F.gelu(input_non_fused + input_bias_non_fused)

    print("> Test forward with bias...", end="")
    assert torch.allclose(output_fused, output_non_fused, atol=1e-2)
    print("OK")

    print("> Test backward with bias...", end="")
    output_non_fused.sum().backward()
    output_fused.sum().backward()
    assert torch.allclose(input_fused.grad, input_non_fused.grad, atol=1e-2)
    assert torch.allclose(input_bias_fused.grad, input_bias_non_fused.grad, atol=1e-2)
    print("OK")


if __name__ == "__main__":
    print("Test tests/torch/nn/modules/functional/test_gelu.py")
    test_gelu()
