import os
import random
import numpy as np
import torch
import torch.distributed as dist
import transformers
import oslo

from copy import deepcopy
from tensorboardX import SummaryWriter
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tqdm import tqdm
from tests.tasks.model_task import ModelTask
from oslo import ParallelContext, ParallelMode
from oslo.torch.nn.parallel import (
    TensorParallel,
    PipelineParallel,
    DistributedDataParallel,
)
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


def main():
    args = get_args()
    name = (
        f"{args.model}-{args.task}-"
        f"bsz={args.batch_size}-"
        f"len={args.sequence_length}"
    )

    args.local_rank = int(os.environ["LOCAL_RANK"])
    print(args.local_rank)

    # 1. Create parallelized model
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

    # 2. Load parallelized model
    model_oslo.from_parallelized(path=args.merge_dir)

    # 3. Save and merge model checkpoint
    saved_merge_dir = args.merge_dir + "_merge"
    model_oslo.save_pretrained(save_directory=saved_merge_dir, merge_checkpoints=True)

    if torch.distributed.get_rank() == 0:
        print("Complete checkpoint merge")


if __name__ == "__main__":
    main()
