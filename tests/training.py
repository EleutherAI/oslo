import os
import random
import numpy as np
import torch
import torch.distributed as dist
import transformers
import oslo
import wandb

from copy import deepcopy
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tqdm import tqdm
from tests.tasks.model_task import ModelTask
from oslo import ParallelContext, ParallelMode
from oslo.torch.nn.parallel import TensorParallel, PipelineParallel
from oslo.torch.nn.parallel.data_parallel import DistributedDataParallel as DDP


from tests.util.arg_parser import get_args

# Define tensor parallel mode
tensor_parallel_mode_map = {
    "1D": ParallelMode.TENSOR_1D,
    "2D": ParallelMode.TENSOR_2D,
    "2D_ROW": ParallelMode.TENSOR_2D_ROW,
    "2D_COL": ParallelMode.TENSOR_2D_COL,
    "2P5D": ParallelMode.TENSOR_2P5D,
    "2P5D_ROW": ParallelMode.TENSOR_2P5D_ROW,
    "2P5D_COL": ParallelMode.TENSOR_2P5D_COL,
    "2P5D_DEP": ParallelMode.TENSOR_2P5D_DEP,
    "2P5D_XZ": ParallelMode.TENSOR_2P5D_XZ,
    "3D": ParallelMode.TENSOR_3D,
    "3D_INPUT": ParallelMode.TENSOR_3D_INPUT,
    "3D_WEIGHT": ParallelMode.TENSOR_3D_WEIGHT,
    "3D_OUTPUT": ParallelMode.TENSOR_3D_OUTPUT,
}


def torch_ddp_dataloader(dataset, batch_size, parallel_context, args):
    """DDP func"""
    num_workers = 1
    if args.tensor_parallel_size > 1:
        rank_group = tensor_parallel_mode_map[args.tensor_parallel_mode]
    else:
        rank_group = ParallelMode.DATA

    num_replicas = parallel_context.get_world_size(rank_group)

    rank = parallel_context.get_local_rank(rank_group)

    batch_sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank)

    d_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size // num_replicas,
        pin_memory=True,
        shuffle=False,
        num_workers=num_workers,
        sampler=batch_sampler,
    )
    return d_loader


def main():

    args = get_args()
    name = (
        f"{args.model}-{args.task}-"
        f"bsz={args.batch_size}-"
        f"len={args.sequence_length}"
    )

    # 1. set tokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. set seed
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    args.local_rank = int(os.environ["LOCAL_RANK"])
    print(args.local_rank)

    # 3. Create parallelized model and optimizer
    model_tasks = ModelTask()
    model_tasks_config = model_tasks.get_model_task(args.task)

    model_oslo = model_tasks_config["class"](args.model)
    optimizer_oslo = AdamW(model_oslo.parameters(), lr=3e-5)

    model_no_oslo = model_tasks_config["class"](args.model)
    optimizer_no_oslo = AdamW(model_no_oslo.parameters(), lr=3e-5)

    parallel_context = ParallelContext.from_torch(
        data_parallel_size=args.data_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        tensor_parallel_mode=tensor_parallel_mode_map[args.tensor_parallel_mode],
    )

    if args.tensor_parallel_size > 1:
        model_oslo = TensorParallel(model_oslo, parallel_context)

    if args.data_parallel_size > 1:
        model_oslo = DDP(model_oslo, parallel_context)

    assert (
        args.tensor_parallel_size > 1 or args.data_parallel_size > 1
    ), "Check the parallel strategy"

    oslo.ready(model_oslo, parallel_context)

    if args.tensor_parallel_size == 1 and args.data_parallel_size > 1:
        torch.cuda.set_device(dist.get_rank())
        model_no_oslo = model_no_oslo.cuda(dist.get_rank())
        model_no_oslo = torch.nn.parallel.DistributedDataParallel(
            model_no_oslo, device_ids=[dist.get_rank()], find_unused_parameters=False
        )

    # 4. Initialize wandb and create folders
    if not dist.is_initialized() or dist.get_rank() == 0:
        wandb.init(project="test", name=name)
        os.makedirs("tests/ckpt", exist_ok=True)
        os.makedirs("tests/cache", exist_ok=True)

    dist.barrier()

    # 5. Load dataset and do preprocessing
    dataset = model_tasks_config["load_dataset"]
    torch_dataset = model_tasks_config["preprocessing_map_func"](
        dataset, tokenizer, args
    )

    if args.data_parallel_size > 1:
        oslo_model_dataloader = torch_ddp_dataloader(
            torch_dataset, args.batch_size, parallel_context, args
        )
    else:
        oslo_model_dataloader = DataLoader(
            torch_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True
        )

    # 6. Train model
    step = 0
    model_no_oslo.cuda()
    model_no_oslo.train()
    model_oslo.train()

    for ep in range(args.epoch):
        save_model_dir = f"tests/ckpt/checkpoint_{str(ep)}"

        if dist.get_rank() == 0:
            print(f"Start training epoch: {ep}")

        for _, sample in enumerate(tqdm(oslo_model_dataloader)):
            optimizer_oslo.zero_grad()
            optimizer_no_oslo.zero_grad()
            inputs = {k: v.cuda() for k, v in sample.items() if k != "guid"}

            # 7. Run no oslo model
            oslo_loss = model_oslo(**inputs).loss

            # 8. Run no oslo model
            no_oslo_loss = model_no_oslo(**inputs).loss

            if dist.get_rank() == 0:
                print(
                    f"[tp/no_tp loss]: {oslo_loss.item():.4f} / {no_oslo_loss.item():.4f}"
                )
                wandb.log(
                    {
                        "oslo_loss": f"{oslo_loss.item():.4f}",
                        "no_oslo_loss": f"{no_oslo_loss.item():.4f}",
                        "time": step,
                    }
                )

            step += 1

            oslo_loss.backward()
            optimizer_oslo.step()

            no_oslo_loss.backward()
            optimizer_no_oslo.step()

        dist.barrier()
        # 9. Save oslo model
        if ep % args.save_interval == 0:
            if not dist.is_initialized() or dist.get_rank() == 0:
                os.makedirs(save_model_dir, exist_ok=True)

            model_oslo.save_pretrained(
                save_directory=save_model_dir, merge_checkpoints=False
            )

    dist.barrier()
    # 10. Save last oslo model where if not saved model in last epoch
    if ep % args.save_interval != 0:
        if not dist.is_initialized() or dist.get_rank() == 0:
            os.makedirs(save_model_dir, exist_ok=True)

        model_oslo.save_pretrained(
            save_directory=save_model_dir, merge_checkpoints=False
        )


if __name__ == "__main__":
    main()
