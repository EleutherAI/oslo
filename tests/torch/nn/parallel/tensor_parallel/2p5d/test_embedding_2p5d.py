from copy import deepcopy

import torch
import torch.distributed as dist

from _utils import split_batch_2p5d, split_2p5d, split_embedding_2p5d, gather_2p5d
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import Embedding2p5D

tp_size = 8
tp_depth = 2

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode=ParallelMode.TENSOR_2P5D,
    tensor_parallel_depth=tp_depth,
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)

batch_size = 2
seq_len = 5
num_embeddings = 16
embedding_dim = 8
tesseract_dim = parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)
input_ = torch.LongTensor([[0, 1, 6, 3, 8], [5, 2, 7, 4, 9]]).cuda()
target = torch.randn((batch_size, seq_len, embedding_dim)).cuda()
dist.broadcast(input_, src=0)
dist.broadcast(target, src=0)

embedding = torch.nn.Embedding(num_embeddings, embedding_dim).cuda()
w = deepcopy(embedding.weight.data)

out = embedding(input_)
optimizer = torch.optim.Adam(embedding.parameters(), lr=1e-3)
loss = torch.nn.MSELoss()(out, target)
loss.backward()
optimizer.step()

out_update = embedding(input_)

if parallel_context.get_global_rank() == 0:
    print(f"original output: \n{out}\n")
    print(f"original update output: \n{out_update}\n")

input_ = split_batch_2p5d(input_, tesseract_dim, parallel_context=parallel_context)
target = split_2p5d(target, tesseract_dim, parallel_context=parallel_context)
w = split_embedding_2p5d(w, tesseract_dim, dim=-1, parallel_context=parallel_context)

embedding_2p5d = Embedding2p5D(
    num_embeddings, embedding_dim, parallel_context=parallel_context
)
embedding_2p5d.weight.data.copy_(w)

pout = embedding_2p5d(input_)
optimizer = torch.optim.Adam(embedding_2p5d.parameters(), lr=1e-3)
loss = torch.nn.MSELoss()(pout, target)
loss.backward()
optimizer.step()

pout_update = embedding_2p5d(input_)

pout = gather_2p5d(pout, tesseract_dim, parallel_context=parallel_context)
pout_update = gather_2p5d(pout_update, tesseract_dim, parallel_context=parallel_context)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{pout}\n")
    print(f"parallel update output: \n{pout_update}\n")

if parallel_context.get_global_rank() == 0:
    sse = torch.sum((out - pout) ** 2).item()
    sse_update = torch.sum((out_update - pout_update) ** 2).item()
    print(f"output sse: \n{sse}\n")
    print(f"next output sse: \n{sse_update}\n")
