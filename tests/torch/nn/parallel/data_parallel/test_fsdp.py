import os
import argparse
import torch
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from oslo.torch.nn.parallel.data_parallel.data_parallel import DataParallel

from oslo.torch.distributed import ParallelContext

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def cleanup():
    dist.destroy_process_group()

def train(args, model, tokenizer, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)

    for data in train_loader:
        optimizer.zero_grad()

        inputs = tokenizer(
            data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(rank)

        loss = model(**inputs, labels=inputs["input_ids"]).loss
        loss.backward()
        optimizer.step()

        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))


def test(model, tokenizer, rank, world_size, test_loader):
    model.eval()
    ddp_loss = torch.zeros(2).to(rank)
    with torch.no_grad():
        for data in test_loader:
            inputs = tokenizer(
                data,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(rank)
            loss = model(**inputs, labels=inputs["input_ids"]).loss
            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        print('Test \tLoss: {:.6f}'.format(ddp_loss[0] / ddp_loss[1]))


def fsdp_main(rank, world_size, args):
    parallel_context = ParallelContext.from_torch(
        data_parallel_size=2,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
    )
    rank = parallel_context.get_local_rank()
    world_size = parallel_context.get_world_size()

    dataset1 = load_dataset("squad").data["train"]["context"]
    dataset1 = [str(sample) for sample in dataset1[:500]]
    dataset2 = load_dataset('squad').data['validation']["context"]
    dataset2 = [str(sample) for sample in dataset2[:500]]

    sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel(GPT2Config.from_pretrained("gpt2")).to(rank)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    model = DataParallel(
        module=model,
	    optimizer=optimizer,
        parallel_context=parallel_context,
        zero_stage=3
    )

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    init_start_event.record()
    for epoch in range(1, args.epochs + 1):
        train(args, model,tokenizer, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        test(model, tokenizer, rank, world_size, test_loader)
        scheduler.step()

    init_end_event.record()

    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        # print(f"{model}")

    # if args.save_model:
    #     # use a barrier to make sure training is done on all ranks
    #     dist.barrier()
    #     # state_dict for FSDP model is only available on Nightlies for now
    #     states = model.state_dict()
    # if rank == 0:
    #     torch.save(states, "mnist_cnn.pt")

    cleanup()

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)