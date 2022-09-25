import os
import random
import json
import argparse

import numpy as np

import torch
import torch.backends.cudnn as cudnn

import deepspeed.comm as dist


class TestFFNBlock(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.fc1 = torch.nn.Linear(in_features, out_features)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(out_features, in_features)

    def forward(self, inp):
        front_out = self.fc1(inp)
        inter = self.act(front_out)
        behind_out = self.fc2(inter)

        return behind_out


def fix_seed(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def sequence_dataloader(
    batch_size, total_samples, hidden_dim, device, seq_len: int = 32, dtype=torch.half
):
    train_data = torch.randn(
        total_samples, seq_len, hidden_dim, device=device, dtype=dtype
    )
    train_label = torch.empty(total_samples, dtype=torch.long, device=device).random_(
        hidden_dim
    )

    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    return train_loader


def create_config_from_dict(tmpdir, config_dict):
    config_path = os.path.join(tmpdir, "temp_config.json")
    with open(config_path, "w") as fd:
        json.dump(config_dict, fd)
    return config_path


def create_deepspeed_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args="")
    args.deepspeed = True
    if dist.is_initialized():
        # We assume up to one full node executing unit tests
        assert dist.get_world_size() <= torch.cuda.device_count()
        args.local_rank = dist.get_rank()
    return args


def args_from_dict(tmpdir, config_dict):
    args = create_deepspeed_args()
    config_path = create_config_from_dict(tmpdir, config_dict)
    args.deepspeed_config = config_path
    return args
