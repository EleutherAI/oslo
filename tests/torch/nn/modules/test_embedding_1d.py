import torch
import torch.distributed as dist

import oslo
import oslo.torch.nn as onn
from oslo import ParallelMode
from utils import split_1d


def test_embedding_1d(pc):
    torch_embedding = torch.nn.Embedding(16, 16).cuda()
    onn_embedding = onn.Embedding1D(16, 16, parallel_context=pc).cuda()
    vp_embedding = onn.VocabParallelEmbedding1D(16, 16, parallel_context=pc).cuda()

    tp_size = pc.get_world_size(ParallelMode.TENSOR)
    onn_embedding.weight.data = split_1d(
        torch_embedding.weight.data, tp_size, dim=1, parallel_context=pc
    )
    vp_embedding.weight.data = split_1d(
        torch_embedding.weight.data, tp_size, dim=0, parallel_context=pc
    )

    # test forward
    if dist.get_rank() == 0:
        print("> Test forward...", end="")
    input_tensor = torch.randint(0, 16, (16,)).cuda()

    torch_output = torch_embedding(input_tensor)
    onn_output = onn_embedding(input_tensor)
    vp_output = vp_embedding(input_tensor)

    assert torch.allclose(torch_output, onn_output)
    assert torch.allclose(torch_output, vp_output)

    if dist.get_rank() == 0:
        print("OK")


if __name__ == "__main__":
    pc = oslo.ParallelContext.from_torch(tensor_parallel_size=4)
    if dist.get_rank() == 0:
        print("Test tests/torch/nn/modules/test_embedding_1d.py")
    test_embedding_1d(pc=pc)
