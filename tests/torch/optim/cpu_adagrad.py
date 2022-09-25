import numpy as np
import pytest
import torch

# import deepspeed
from oslo.torch.optim.cpu_adagrad import CPUAdagrad


# from deepspeed.ops.op_builder import CPUAdagradBuilder

# if not deepspeed.ops.__compatible_ops__[CPUAdagradBuilder.NAME]:
#     pytest.skip("cpu-adagrad is not compatible")


def check_equal(first, second, atol=1e-2, verbose=False):
    x = first.detach().numpy()
    y = second.detach().numpy()
    if verbose:
        print("x = {}".format(x.flatten()))
        print("y = {}".format(y.flatten()))
        print("-" * 80)
    np.testing.assert_allclose(x, y, err_msg="param-update mismatch!", atol=atol)


@pytest.mark.parametrize(
    "model_size",
    [
        (64),
        (22),
        (55),
        (127),
        (1024),
        (1048576),
        (30000000),
    ],
)  # yapf: disable
def test_cpu_adagrad_opt(model_size):
    device = "cpu"
    rng_state = torch.get_rng_state()
    param = torch.nn.Parameter(torch.randn(model_size, device=device))
    torch.set_rng_state(rng_state)
    param1 = torch.nn.Parameter(torch.randn(model_size, device=device))
    torch.set_rng_state(rng_state)

    optimizer = CPUAdagrad([param])
    optimizer1 = torch.optim.Adagrad([param1])

    for i in range(10):
        rng_state = torch.get_rng_state()
        param.grad = torch.randn(model_size, device=device)
        torch.set_rng_state(rng_state)
        param1.grad = torch.randn(model_size, device=device)
        optimizer.step()
        optimizer1.step()

    check_equal(param, param1, atol=1e-2, verbose=True)


@pytest.mark.parametrize(
    "model_size,vocabulary_size,dim",
    [
        (16 * 2, 16 * 4, 16),
        (16 * 32, 16 * 256, 16),
        (16 * 256, 16 * 16384, 16),
    ],
)  # yapf: disable
def test_cpu_adagrad_opt_sparse_embedding(model_size, vocabulary_size, dim):
    device = "cpu"
    rng_state = torch.get_rng_state()

    def gen_sparse_grad(vocabulary_size, dim, num_indices, dtype, device):
        i = torch.randint(
            vocabulary_size, size=(1, num_indices), dtype=torch.int64, device=device
        )
        v = torch.randn(num_indices, dim, dtype=dtype, device=device)
        t = torch.sparse_coo_tensor(i, v, (vocabulary_size, dim), device=device)
        t = t.coalesce()
        new_i = (
            (t.indices().view(-1, 1).repeat(1, dim) * dim + torch.tensor(range(dim)))
            .flatten()
            .unsqueeze(0)
        )
        new_v = t.values().flatten()
        new_t = torch.sparse_coo_tensor(
            new_i, new_v, (vocabulary_size * dim,), device=device
        )
        new_t = new_t.coalesce()
        new_t.requires_grad = False
        return new_t

    voc_size = vocabulary_size
    dim = dim
    num_indices = int(model_size // dim)
    dtype = torch.float32

    param = torch.nn.Parameter(
        torch.randn((voc_size * dim,), dtype=dtype, device=device), requires_grad=True
    )
    torch.set_rng_state(rng_state)
    param1 = torch.nn.Parameter(
        torch.randn((voc_size * dim,), dtype=dtype, device=device), requires_grad=True
    )
    torch.set_rng_state(rng_state)

    optimizer = CPUAdagrad([param])
    optimizer1 = torch.optim.Adagrad([param1])

    for i in range(10):
        torch.set_rng_state(rng_state)
        param.grad = gen_sparse_grad(
            voc_size, dim, num_indices, dtype=dtype, device=device
        )
        torch.set_rng_state(rng_state)
        param1.grad = gen_sparse_grad(
            voc_size, dim, num_indices, dtype=dtype, device=device
        )
        optimizer.step()
        optimizer1.step()

    check_equal(param, param1, atol=1e-2, verbose=True)


def test_cpu_adam_gpu_error():
    model_size = 64
    device = "cuda:0"
    param = torch.nn.Parameter(torch.randn(model_size, device=device))
    optimizer = CPUAdagrad([param])

    param.grad = torch.randn(model_size, device=device)
    with pytest.raises(AssertionError):
        optimizer.step()
