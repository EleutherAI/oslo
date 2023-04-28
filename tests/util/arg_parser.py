import argparse
import sys
import os
from tensorboardX import SummaryWriter

SUMMARY_WRITER_DIR_NAME = "runs"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", default=0, type=int)
    # parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--tokenizer", default=None, type=str)
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--sequence_length", required=True, type=int)
    parser.add_argument("--train_step", required=True, type=int)
    parser.add_argument("--save_interval", required=True, type=int)
    parser.add_argument("--tensor_parallel_size", default=1, type=int)
    parser.add_argument("--data_parallel_size", default=1, type=int)
    parser.add_argument("--pipeline_parallel_size", default=1, type=int)
    parser.add_argument("--tensor_parallel_depth", default=1, type=int)
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--tensor_parallel_mode", default="1D", type=str)
    args = parser.parse_args()
    return args


def get_summary_writer(name, base=".."):
    """Returns a tensorboard summary writer"""
    return SummaryWriter(log_dir=os.path.join(base, SUMMARY_WRITER_DIR_NAME, name))


def write_summary_events(summary_writer, summary_events):
    for event in summary_events:
        summary_writer.add_scalar(event[0], event[1], event[2])
