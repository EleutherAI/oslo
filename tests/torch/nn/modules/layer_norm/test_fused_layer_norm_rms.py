# from oslo.torch._C import FusedLayerNormBinder
import unittest

import torch

import oslo.torch.nn as onn


class TestFusedRMSNorm(unittest.TestCase):
    dtype = torch.float
    elementwise_affine = False
    normalized_shape = [32, 16]
    rtol, atol = None, None
    fwd_thresholds = dict(rtol=1e-2, atol=1e-2)
    bwd_thresholds = dict(rtol=1e-2, atol=1e-2)
    mixed_fused = False

    def setUp(self):
        # bias and weight are set to 0 and 1 respectively, so no need to copy parameters from cpu module to the gpu one
        if not self.mixed_fused:
            self.module_cpu_ = onn.FusedRMSNorm(
                normalized_shape=self.normalized_shape,
                elementwise_affine=self.elementwise_affine,
            ).cpu()
            self.module_cuda_ = onn.FusedRMSNorm(
                normalized_shape=self.normalized_shape,
                elementwise_affine=self.elementwise_affine,
            ).to(device="cuda", dtype=self.dtype)
        else:
            assert self.elementwise_affine
            self.module_cpu_ = onn.MixedFusedRMSNorm(
                normalized_shape=self.normalized_shape
            ).cpu()
            self.module_cuda_ = onn.MixedFusedRMSNorm(
                normalized_shape=self.normalized_shape
            ).to(device="cuda", dtype=self.dtype)

    def _check_same_output(self, batch_size, contiguous):
        torch.cuda.manual_seed(42)
        if contiguous:
            input_shape = [batch_size] + self.normalized_shape
            input_ = torch.randn(input_shape, device="cpu").requires_grad_(True)
            input_cuda_ = (
                input_.to(device="cuda", dtype=self.dtype).detach().requires_grad_(True)
            )
            self.assertTrue(input_.is_contiguous())
            self.assertTrue(input_cuda_.is_contiguous())
        else:
            input_shape = [batch_size * 3] + [
                self.normalized_shape[0] * 5,
                self.normalized_shape[1] * 3,
            ]
            input_src_ = torch.randn(input_shape, device="cpu")
            input_ = input_src_[::3, ::5, ::3].detach().requires_grad_(True)
            input_cuda_ = (
                input_src_.to(device="cuda", dtype=self.dtype)[::3, ::5, ::3]
                .detach()
                .requires_grad_(True)
            )
            # make sure that tensors are NOT contiguous.
            self.assertFalse(input_.is_contiguous())
            self.assertFalse(input_cuda_.is_contiguous())
        out_cpu_ = self.module_cpu_(input_)
        gO = torch.rand_like(out_cpu_)
        out_cpu_.backward(gO)
        out_cuda_ = self.module_cuda_(input_cuda_)
        # TODO (mkozuki): `torch.testing.assert_close` is deprecated.
        # Use `torch.testing.assert_close`.
        # See https://github.com/pytorch/pytorch/issues/61844
        torch.testing.assert_close(
            out_cpu_.to(device="cuda", dtype=self.dtype),
            out_cuda_.clone().detach(),
            **self.fwd_thresholds,
        )
        gO = gO.to(device="cuda", dtype=self.dtype)
        out_cuda_.backward(gO)
        self.assertFalse(out_cpu_.is_cuda)
        self.assertTrue(out_cuda_.is_cuda)
        torch.testing.assert_close(
            input_.grad.to(device="cuda", dtype=self.dtype),
            input_cuda_.grad,
            **self.bwd_thresholds,
        )
        if self.elementwise_affine:
            torch.testing.assert_close(
                self.module_cpu_.weight.grad.to(device="cuda", dtype=self.dtype),
                self.module_cuda_.weight.grad,
                **self.bwd_thresholds,
            )

    def _test_same_output(self, batch_size):
        for contiguous in (True, False):
            with self.subTest(contiguous=contiguous):
                self._check_same_output(batch_size, contiguous)

    def test_layer_norm(self):
        self._test_same_output(16)

    def test_large_batch(self):
        self._test_same_output(65536)


class TestFusedRMSNormElemWise(TestFusedRMSNorm):
    elementwise_affine = True


class TestMixedFusedRMSNormElemWise(TestFusedRMSNorm):
    elementwise_affine = True
    mixed_fused = True


if __name__ == "__main__":
    unittest.main(verbosity=True)
