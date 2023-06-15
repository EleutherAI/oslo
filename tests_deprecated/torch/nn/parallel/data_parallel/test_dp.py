import os

from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import oslo
from oslo import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.data_parallel import DistributedDataParallel as DDP


BATCH_SIZE = 4
SEQ_LEN = 64
SAVE_INTERVAL = 50
TRAIN_STEP = 100
LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])


def train():

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    optimizer = Adam(model.parameters(), lr=3e-5)

    # Add pad token for batch training because GPT2 tokenizer doesn't have pad token.
    tokenizer.pad_token = tokenizer.eos_token

    parallel_context = ParallelContext.from_torch(
        data_parallel_size=LOCAL_WORLD_SIZE,
    )

    model = DDP(model, parallel_context)
    oslo.ready(model, parallel_context)

    rank = parallel_context.get_local_rank(ParallelMode.DATA)

    datasets = load_dataset("squad").data["train"]["context"]
    datasets = [str(_) for _ in datasets[: TRAIN_STEP * BATCH_SIZE]]
    train_sampler = DistributedSampler(
        datasets, num_replicas=LOCAL_WORLD_SIZE, rank=rank
    )
    dataloader = DataLoader(datasets, batch_size=BATCH_SIZE, sampler=train_sampler)

    for step, batch in enumerate(dataloader):
        model.zero_grad()

        # Make batch
        input_batch = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=SEQ_LEN,
        ).to("cuda")

        # Forward-Backward-Step
        loss = model(**input_batch, labels=input_batch["input_ids"]).loss
        loss.backward()
        optimizer.step()

    # Save the parallelized model using `save_pretrained`
    model.save_pretrained(save_directory="./parallel_ckpt")


if __name__ == "__main__":
    train()

# torchrun --nnodes=1  --nproc_per_node=4 --master_addr=localhost --master_port=12345 test_dp.py
