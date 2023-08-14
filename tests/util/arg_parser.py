import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", default=0, type=int)
    # parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--tokenizer", default=None, type=str)
    parser.add_argument("--batch_size", required=False, type=int)
    parser.add_argument("--sequence_length", required=False, type=int)
    parser.add_argument("--train_step", required=False, type=int)
    parser.add_argument("--save_interval", required=False, type=int)
    parser.add_argument("--tensor_parallel_size", default=1, type=int)
    parser.add_argument("--data_parallel_size", default=1, type=int)
    parser.add_argument("--pipeline_parallel_size", default=1, type=int)
    parser.add_argument("--tensor_parallel_depth", default=1, type=int)
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--tensor_parallel_mode", default="1D", type=str)
    parser.add_argument("--merge_dir", required=False, type=str)
    args = parser.parse_args()
    return args
