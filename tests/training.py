import os
import random
import numpy as np
import torch
import torch.distributed as dist
import transformers
import oslo

from tensorboardX import SummaryWriter
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tqdm import tqdm
from tests.tasks.model_task import ModelTask
from oslo import ParallelContext, ParallelMode
from oslo.torch.nn.parallel import TensorParallel
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


def set_cpu_maximum_parallelism():
    """Maximize parallelism"""
    conf_str = torch.__config__.parallel_info()
    inter_str = conf_str.split("hardware_concurrency() : ")[1]
    max_concurrency = inter_str.split("\n")[0]

    os.environ["OMP_NUM_THREADS"] = max_concurrency
    print(f"environmental variable OMP_NUM_THREADS is set to {max_concurrency}.")


def torch_ddp_dataloader(dataset, batch_size):
    """DDP func샤ㅐㅜ"""
    num_workers = 1
    d_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=num_workers,
        sampler=DistributedSampler(dataset, shuffle=False),
    )
    return d_loader


def get_summary_writer(name, base=".."):
    """Returns a tensorboard summary writer"""
    return SummaryWriter(log_dir=os.path.join(base, name))


def write_summary_events(summary_writer, summary_events):
    """Add tensor on summary writer"""
    for event in summary_events:
        summary_writer.add_scalar(event[0], event[1], event[2])


def train_step(name, model, optimizer, inputs, step, summary_writer):
    """Forward-Backward-Step"""
    loss = model(**inputs).loss
    loss.backward()
    optimizer.step()

    if not dist.is_initialized() or dist.get_rank() == 0:
        summary_events = [(f"{name}/loss", loss.item(), step)]
        write_summary_events(summary_writer, summary_events)
        summary_writer.flush()


def main():
    args = get_args()
    set_cpu_maximum_parallelism()
    name = (
        f"{args.model}-{args.task}-"
        f"bsz={args.batch_size}-"
        f"len={args.sequence_length}"
    )

    # 1. set tokenizer
    args.tokenizer = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
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

    parallel_context = ParallelContext.from_torch(
        data_parallel_size=args.data_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        tensor_parallel_mode=tensor_parallel_mode_map[args.tensor_parallel_mode],
        tensor_parallel_depth=args.tensor_parallel_depth,
    )

    model_oslo = TensorParallel(model_oslo, parallel_context)
    oslo.ready(model_oslo, parallel_context)
    optimizer_oslo = AdamW(params=model_oslo.parameters(), lr=3e-5)

    model_no_oslo = model_tasks_config["class"](args.model).to("cuda").train()
    optimizer_no_oslo = AdamW(params=model_no_oslo.parameters(), lr=3e-5)

    # 4. Initialize wandb and create folders
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs("tests/ckpt", exist_ok=True)
        os.makedirs("tests/cache", exist_ok=True)

    # 5. Load dataset and do preprocessing
    dataset = model_tasks_config["load_dataset"].select(range(args.train_step))
    torch_dataset = model_tasks_config["preprocessing_map_func"](
        dataset, tokenizer, args.sequence_length
    )

    base_dataloader = DataLoader(
        torch_dataset, batch_size=args.batch_size, shuffle=False
    )

    if args.data_parallel_size > 1:
        oslo_model_dataloader = torch_ddp_dataloader(torch_dataset, args.batch_size)
        print(f"Set up DDP dataloader and size: {len(oslo_model_dataloader)}")
    else:
        oslo_model_dataloader = base_dataloader

    if dist.get_rank() == 0:
        print(f"base dataloader size: {len(base_dataloader)}")

    # 6. Set up summary writer
    tensor_writer = get_summary_writer(name=f"tensorboard", base="tests/ckpt")

    # 7. Train model
    oslo_train_step = 0
    no_oslo_train_step = 0
    for ep in range(args.epoch):

        save_model_dir = f"tests/ckpt/checkpoint_{str(ep)}"

        if dist.get_rank() == 0:
            print(f"Start training epoch: {ep}")

        # 8. Run no oslo model
        for _, sample in enumerate(tqdm(oslo_model_dataloader)):
            inputs = {k: v.cuda() for k, v in sample.items()}
            optimizer_oslo.zero_grad()
            train_step(
                name=f"oslo",
                model=model_oslo,
                optimizer=optimizer_oslo,
                inputs=inputs,
                step=oslo_train_step,
                summary_writer=tensor_writer,
            )
            oslo_train_step += 1

        # 9. Run no oslo model
        for _, sample in enumerate(tqdm(base_dataloader)):
            inputs = {k: v.cuda() for k, v in sample.items()}
            optimizer_no_oslo.zero_grad()
            train_step(
                name=f"no_oslo",
                model=model_no_oslo,
                optimizer=optimizer_no_oslo,
                inputs=inputs,
                step=no_oslo_train_step,
                summary_writer=tensor_writer,
            )
            no_oslo_train_step += 1

        # 10. Save oslo model
        if ep % args.save_interval == 0:
            if not dist.is_initialized() or dist.get_rank() == 0:
                os.makedirs(save_model_dir, exist_ok=True)

            model_oslo.save_pretrained(
                save_directory=save_model_dir,
                # merge_checkpoints=False
            )

    # 11. Save last oslo model where if not saved model in last epoch
    if ep % args.save_interval != 0:
        if not dist.is_initialized() or dist.get_rank() == 0:
            os.makedirs(save_model_dir, exist_ok=True)

        model_oslo.save_pretrained(
            save_directory=save_model_dir,
            # merge_checkpoints=True
        )


if __name__ == "__main__":
    main()
