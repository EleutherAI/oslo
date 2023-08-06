## inference shell code
# EXAMPLE: ``sh ./tests/test_script/run_inference.sh 4 bert-base-cased masked-lm ``
# EXAMPLE: ``sh ./tests/test_script/run_inference.sh 4 ishan/bert-base-uncased-mnli sequence-classification ``
# EXAMPLE: ``sh ./tests/test_script/run_inference.sh 4 gpt2 causal-lm ``
# EXAMPLE: ``sh ./tests/test_script/run_inference.sh 4 EleutherAI/gpt-neo-1.3B causal-lm ``
# EXAMPLE: ``sh ./tests/test_script/run_inference.sh 4 t5-base seq2seq-lm ``

NUM_GPUS=$1
MODEL=$2
TASK=$3

python -m torch.distributed.launch \
       --nproc_per_node="$NUM_GPUS" \
       ./tests/inference.py \
       --task=$TASK \
       --model=$MODEL \
       --tensor_parallel_size="$NUM_GPUS"
