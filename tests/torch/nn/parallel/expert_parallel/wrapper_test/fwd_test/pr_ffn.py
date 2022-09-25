import random
from functools import partial
import os

import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from oslo.torch.nn.parallel.data_parallel.distributed_data_parallel import (
    DistributedDataParallel,
)
from oslo.torch.nn.parallel.expert_parallel.expert_parallel import ExpertParallel

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.expert_parallel.mapping import Front, Behind

from fwd_utils import TestFFNBlock, fix_seed, sequence_dataloader

from oslo.torch.nn.parallel.expert_parallel._ops import AllReduce

torch.set_printoptions(threshold=10_000)

total_samples = 50

batch_size = 2
sent_len = 4

hidden_dim = 2
in_features = hidden_dim
out_features = 4
n_layers = 2

world_size = 4
num_experts = {
    (0,): world_size,
    (1,): 2 * world_size,
}
top_k = 1

use_residual = True


class TestMoE(torch.nn.Module):
    def __init__(self, ffns):
        super().__init__()

        self.ffns = torch.nn.ModuleList(ffns)

    def forward(self, x):
        out = x
        for cur_layer in self.ffns:
            out = cur_layer(out)
        return out


class SimplePRMoEModel(torch.nn.Module):
    def __init__(self, linear, moe):
        super().__init__()

        # self.linear = linear
        self.moe = moe
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        # linear_out = self.linear(x)
        # moe_out = self.moe(linear_out)
        moe_out = self.moe(x)

        # resid_out = linear_out + moe_out
        resid_out = x + moe_out
        sent_emb = resid_out.mean(1)

        return self.cross_entropy_loss(sent_emb, y)


# Class for Mapping information of Entire Model to expert parallelize
class ExpertParallelMappingForTest(object):
    __MAPPING__ = {
        "TestMoE": [
            Front("fc1", enc_name="ffns", layer="ffns"),
            Behind("fc2", enc_name="ffns", layer="ffns"),
        ]
    }

    def __init__(self):
        cache_mapping = {}
        import sys

        for cls_name, mapping in self.__MAPPING__.items():
            cls = globals()[cls_name]
            if cls is not None:
                cache_mapping[cls] = mapping

        self.__MAPPING__ = cache_mapping

    def get_mapping(self, model):
        mapping_by_model = None
        for cls, mapping in self.__MAPPING__.items():
            if isinstance(model, cls):
                mapping_by_model = {cls: mapping}

        assert mapping_by_model is not None, (
            f"Currently, {model.__class__.__qualname__} is not supported. "
            f"The current supported models are {list(self.__MAPPING__.keys())}"
        )
        return mapping_by_model


def run_test(rank, port):
    # 1. Configure for Parallelization
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = os.environ["WORLD_SIZE"]
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # 2. Set Parallel Context
    parallel_context = ParallelContext.from_torch(
        data_parallel_size=2,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        expert_parallel_size=world_size // 2,
    )

    # if rank in [0, 1]:
    #    dist.new_group([0, 1], backend="gloo", timeout=datetime.timedelta(seconds=1))
    #    print(f'Global Rank #{rank} : {[0, 1]} Made')
    # elif rank in [2, 3]:
    #    dist.new_group([2, 3], backend="gloo", timeout=datetime.timedelta(seconds=1))
    #    print(f'Global Rank #{rank} : {[2, 3]} Made')

    fix_seed(rank)

    linear = torch.nn.Linear(in_features, in_features).to(rank)
    ep_group = parallel_context.get_group(ParallelMode.EXPERT)
    src_rank = parallel_context.get_ranks_in_group(ParallelMode.EXPERT)[0]
    torch.distributed.broadcast(linear.weight, src_rank, group=ep_group)
    torch.distributed.broadcast(linear.bias, src_rank, group=ep_group)

    mapping = ExpertParallelMappingForTest()
    ffns = [TestFFNBlock(in_features, out_features) for i in range(n_layers)]
    resid_mix_w1 = torch.nn.Linear(in_features, in_features).to(rank)
    resid_mix_w2 = torch.nn.Linear(in_features, in_features).to(rank)
    torch.distributed.broadcast(resid_mix_w1.weight, src_rank, group=ep_group)
    torch.distributed.broadcast(resid_mix_w1.bias, src_rank, group=ep_group)
    torch.distributed.broadcast(resid_mix_w2.weight, src_rank, group=ep_group)
    torch.distributed.broadcast(resid_mix_w2.bias, src_rank, group=ep_group)
    data_loader = sequence_dataloader(
        batch_size,
        total_samples,
        hidden_dim=hidden_dim,
        device=rank,
        seq_len=sent_len,
        dtype=torch.float32,
    )
    batches = [(n, batch) for n, batch in enumerate(data_loader)]
    moe = TestMoE(ffns)
    moe = ExpertParallel(
        moe,
        parallel_context,
        num_enc_experts=num_experts,
        top_k=top_k,
        use_kernel_optim=False,
        use_residual=use_residual,
        mapping=mapping,
        noisy_policy="Not Use",
        use_rts=False,
    )

    ep_group = parallel_context.get_group(ParallelMode.EXPERT)
    resid_mix_w1.weight.register_hook(
        AllReduce(ep_group, world_size // 2, "expert_parallel_residual_mix")
    )
    resid_mix_w1.bias.register_hook(
        AllReduce(ep_group, world_size // 2, "expert_parallel_residual_mix")
    )
    resid_mix_w2.weight.register_hook(
        AllReduce(ep_group, world_size // 2, "expert_parallel_residual_mix")
    )
    resid_mix_w2.bias.register_hook(
        AllReduce(ep_group, world_size // 2, "expert_parallel_residual_mix")
    )

    moe.model.ffns[0].fc1.expert_parallel_residual_mix = resid_mix_w1
    moe.model.ffns[1].fc1.expert_parallel_residual_mix = resid_mix_w2

    model_ep = SimplePRMoEModel(linear, moe).to(rank)

    model_ep = DistributedDataParallel(model_ep, parallel_context)

    # 6. Forward Propagation
    optimizer = torch.optim.AdamW(params=model_ep.parameters())

    # dp_group_ranks = parallel_context.get_ranks_in_group(ParallelMode.DATA)
    # ep_group_ranks = parallel_context.get_ranks_in_group(ParallelMode.EXPERT)
    # print(f'Data Parallel Process Group : {dp_group_ranks}')
    # print(f'Expert Parallel Process Group : {ep_group_ranks}')
    # for param_name, module in model_ep.named_parameters():
    #   print(
    #       f"Worker #{rank} - param_name : {param_name}, param_size : {module.size()}"
    #   )
    #   print(f"Worker #{rank} - param  : {module}")
    # return

    # for n, batch in enumerate(data_loader):
    for n, batch in batches:
        loss = model_ep(batch[0], batch[1])
        print(f"Worker # {rank} Instance #{n} loss : {loss}")
        loss.backward()
        # for param_name, module in model_ep.named_parameters():
        #    print(
        #        f"Worker #{rank} - param_name : {param_name}, param_size : {module.size()}"
        #    )
        #    print(f"Worker #{rank} - grad  : {module.grad}")
        optimizer.step()

    return


def test_expert_parallel_block():
    run_func = partial(run_test, port=29500)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == "__main__":
    # Set Random Seed for Reproducibility
    # fix_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_expert_parallel_block()
