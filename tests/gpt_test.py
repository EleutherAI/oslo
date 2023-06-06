BATCH_SIZE = 32
SEQ_LEN = 64
SAVE_INTERVAL = 50
TRAIN_STEP = 200

import torch
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist

model = AutoModelForCausalLM.from_pretrained("gpt2")
optimizer = Adam(model.parameters(), lr=3e-5)
tokenizer = AutoTokenizer.from_pretrained("gpt2")


no_model = AutoModelForCausalLM.from_pretrained("gpt2")
no_optimizer = Adam(no_model.parameters(), lr=3e-5)

# Add pad token for batch training because GPT2 tokenizer doesn't have pad token.
tokenizer.pad_token = tokenizer.eos_token

from oslo import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.data_parallel import DistributedDataParallel as DDP
import oslo

dp_size = 2

parallel_context = ParallelContext.from_torch(
    data_parallel_size=dp_size,
)
model = DDP(model, parallel_context)
oslo.ready(model, parallel_context)


torch.cuda.set_device(dist.get_rank())
no_model = no_model.cuda(dist.get_rank())
no_model = torch.nn.parallel.DistributedDataParallel(
    no_model, device_ids=[dist.get_rank()], find_unused_parameters=False
)


from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler

rank = parallel_context.get_local_rank(ParallelMode.DATA)

datasets = load_dataset(
    "wikitext", "wikitext-2-raw-v1", split="train", cache_dir="tests/cache"
)
dataset = datasets.select(range(TRAIN_STEP * BATCH_SIZE))

train_sampler = DistributedSampler(datasets, num_replicas=dp_size, rank=rank)
dataloader = DataLoader(datasets, batch_size=BATCH_SIZE, sampler=train_sampler)

no_model.train()
model.train()

for step, batch in enumerate(dataloader):
    model.zero_grad()
    no_model.zero_grad()

    # Make batch
    input_batch = tokenizer(
        batch["text"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=SEQ_LEN,
    ).to("cuda")

    # Forward-Backward-Step
    loss = model(**input_batch, labels=input_batch["input_ids"]).loss
    no_loss = no_model(**input_batch, labels=input_batch["input_ids"]).loss

    if rank == 0:
        print(f"oslo/no_oslo : {loss.item():.4f} / {no_loss.item():.4f}")

    loss.backward()
    optimizer.step()

    no_loss.backward()
    no_optimizer.step()
