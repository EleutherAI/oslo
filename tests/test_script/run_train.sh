# EXAMPLE TP case BERT:`sh ./tests/test_script/run_train.sh ishan/bert-base-uncased-mnli sequence-classification 2 128 128 100 1 1 1 2 1 1 1D`

# EXAMPLE DP case BERT:`sh ./tests/test_script/run_train.sh ishan/bert-base-uncased-mnli sequence-classification 2 128 128 100 1 2 1 1 1 1 1D`


# EXAMPLE TP case GPT:`sh ./tests/test_script/run_train.sh gpt2 causal-lm 2 64 64 100 1 1 1 2 1 1 1D`

# EXAMPLE TP case T5:`sh ./tests/test_script/run_train.sh t5-base seq2seq 2 64 128 100 1 1 1 2 1 1 1D`


# Check a checkpoint result on wandb

# Task specific model
#  - BERT case
#    - Sequence classification
#      - ishan/bert-base-uncased-mnli

# Task specific model
#  - GPT case
#    - causal-lm
#      - gpt2

# Task specific model
#  - T5 case
#    - seq2seq
#      - t5-base

#########################################
# !!Feature still in development
# 1. Pipeline parallelism
# 2. Tensor parallelism + data pallelism
#########################################

MODEL=$1
TASK=$2

# Define variable of parallel model setting
NUM_GPUS=$3
BATCH_SIZE=$4
SEQ_LENGTH=$5
TRAIN_STEP=$6
TOTAL_TRAIN_STEP=$((TRAIN_STEP*BATCH_SIZE))
SAVE_INTERVAL=$7
DATA_PARALLEL_SIZE=$8
PIPELINE_PARALLEL_SIZE=$9
TENSOR_PARALLEL_SIZE=${10}
TENSOR_PARALLEL_DEPTH=${11}
EPOCH=${12}
# tensor parallel mode
# "1D", "2D", "2D_ROW", "2D_COL", "2P5D", "2P5D_ROW", "2P5D_COL"
# "2P5D_DEP", "2P5D_XZ", "3D", "3D_INPUT", "3D_WEIGHT", "3D_OUTPUT"
TENSOR_PARALLEL_MODE=${13}

run_cmd="torchrun --standalone --nproc_per_node=${NUM_GPUS} \
       ./tests/training.py \
       --task=$TASK \
       --model=$MODEL \
       --batch_size=$BATCH_SIZE \
       --sequence_length=$SEQ_LENGTH \
       --train_step=$TOTAL_TRAIN_STEP \
       --save_interval=$SAVE_INTERVAL \
       --epoch=$EPOCH \
       --tensor_parallel_size=$TENSOR_PARALLEL_SIZE \
       --data_parallel_size=$DATA_PARALLEL_SIZE \
       --pipeline_parallel_size=$PIPELINE_PARALLEL_SIZE \
       --tensor_parallel_mode=$TENSOR_PARALLEL_MODE \
       --tensor_parallel_depth=$TENSOR_PARALLEL_DEPTH
       "

echo ${run_cmd}
eval ${run_cmd}
