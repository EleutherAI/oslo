import os
import json
import argparse

import torch
import torch.backends.cudnn as cudnn

import deepspeed.comm as dist


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
