import argparse
import os
import time

import torch
import torch.distributed as dist
from torch.distributed import rpc


QUEUE = list()


def send(src, dst, tensor):
    fut = rpc.rpc_async(
        to=f"rank_{dst}",
        func=prepare_to_receive,
        args=(src, tensor.shape),
    )

    dist.send(tensor, dst)
    print("Okay!!")

    fut.wait()


def prepare_to_receive(src, shape):
    device = f"cuda:{1}"
    tensor = torch.Tensor(shape).to(device)

    print("Ready to recv!")
    dist.recv(tensor, src)

    QUEUE.append(tensor)


def parse_args():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])

    # init NCCL
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )
    torch.cuda.set_device(args.local_rank)
    device = f"cuda:{args.local_rank}"

    # init rpc
    rpc.init_rpc(
        name=f"rank_{args.local_rank}",
        rank=args.local_rank,
        world_size=args.world_size,
    )

    if args.local_rank == 0:
        tensor = torch.rand((128, 128, 128)).to(device)
        send(0, 1, tensor)

    else:
        while len(QUEUE) < 1:
            time.sleep(0.1)

        tensor = QUEUE.pop()
        print(f"GPU {args.local_rank} received a tensor of shape {tensor.shape}")

    rpc.shutdown()


if __name__ == "__main__":
    main()
