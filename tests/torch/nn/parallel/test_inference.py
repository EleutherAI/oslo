import os
from argparse import ArgumentParser

from transformers import AutoTokenizer

from tasks.causal_lm_task import CausalLMTask
from tasks.masked_lm_task import MaskedLMTask
from tasks.seq2seq_lm_task import Seq2SeqLMTask
from tasks.sequence_classification_task import SequenceClassificationTask
from utils import initialize_oslo, print_rank_0

os.environ["TOKENIZERS_PARALLELISM"] = "true"

inference_tasks = {
    task.name(): task
    for task in [
        Seq2SeqLMTask(),
        CausalLMTask(),
        MaskedLMTask(),
        SequenceClassificationTask(),
    ]
}

parser = ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--tokenizer", type=str, default=None)
parser.add_argument("--input", type=str, default=None)
parser.add_argument("--data_parallel_size", type=int, default=1)
parser.add_argument("--pipeline_parallel_size", type=int, default=1)
parser.add_argument("--tensor_parallel_size", type=int, default=1)
parser.add_argument("--tensor_parallel_depth", type=int, default=1)
parser.add_argument("--tensor_parallel_mode", type=str, default="1D")
args = parser.parse_args()


assert args.task in inference_tasks, (
    f"{args.task} is not supported task. "
    f"Please choose one of {inference_tasks}. "
    "If there are no major problems, it will work for other tasks as well, "
    "but I haven't tested it, so if you encounter any problems, "
    "please report them through the github issue."
)

task = inference_tasks[args.task]

if args.tokenizer is None:
    args.tokenizer = args.model

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
model = task.get_model_class()(args.model)
input = args.input if args.input is not None else task.get_inference_sample(tokenizer)

output_before = task.get_inference_output(
    tokenizer, task.forward(model)(**tokenizer(input, return_tensors="pt"))
)

model, pc = initialize_oslo(args, model)

output_after = task.get_inference_output(
    tokenizer, task.forward(model)(**tokenizer(input, return_tensors="pt").to("cuda"))
)

print_rank_0(
    message=f"""
Result:
> Input: {input}
> Output before: {output_before}
> Output after: {output_after}""",
    pc=pc,
)
