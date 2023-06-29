###########################################################
# If you use only two gpu example
# Checkpoint directory : tests/ckpt/checkpoint_0
# saved merge directory: tests/ckpt/checkpoint_0_merge
###########################################################

# EXAMPLE merge TP case BERT:`sh ./tests/test_script/run_merge.sh ishan/bert-base-uncased-mnli sequence-classification 2 1 1 2 1`

# EXAMPLE merge TP case GPT:`sh ./tests/test_script/run_merge.sh gpt2 causal-lm 2 1 1 2 1`

# EXAMPLE merge TP case T5:`sh ./tests/test_script/run_merge.sh t5-base seq2seq 2 1 1 2 1`


MODEL=$1
TASK=$2

NUM_GPUS=$3
DATA_PARALLEL_SIZE=$4
PIPELINE_PARALLEL_SIZE=$5
TENSOR_PARALLEL_SIZE=$6
TENSOR_PARALLEL_DEPTH=$7

# tensor parallel mode
# "1D", "2D", "2D_ROW", "2D_COL", "2P5D", "2P5D_ROW", "2P5D_COL"
# "2P5D_DEP", "2P5D_XZ", "3D", "3D_INPUT", "3D_WEIGHT", "3D_OUTPUT"
TENSOR_PARALLEL_MODE=1D
MERGE_DIR=tests/ckpt/checkpoint_0

run_cmd="torchrun --standalone --nproc_per_node=${NUM_GPUS} \
       ./tests/merge.py \
       --task=$TASK \
       --model=$MODEL \
       --tensor_parallel_size=$TENSOR_PARALLEL_SIZE \
       --data_parallel_size=$DATA_PARALLEL_SIZE \
       --pipeline_parallel_size=$PIPELINE_PARALLEL_SIZE \
       --tensor_parallel_mode=$TENSOR_PARALLEL_MODE \
       --tensor_parallel_depth=$TENSOR_PARALLEL_DEPTH \
       --merge_dir=$MERGE_DIR
       "

echo ${run_cmd}
eval ${run_cmd}
