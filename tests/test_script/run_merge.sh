# EXAMPLE:`sh ./tests/test_script/run_train.sh aloxatel/bert-base-mnli sequence-classification 4 128 128 100 2 1 2 2 1 4 1D`
# Check tensorboard: `tensorboard --logdir tests/ckpt/tensorboard`

# task specific model
#  - BERT case
#    - Sequence classification
#      - aloxatel/bert-base-mnli
MODEL=aloxatel/bert-base-mnli
TASK=sequence-classification

NUM_GPUS=2
DATA_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1
TENSOR_PARALLEL_SIZE=2
TENSOR_PARALLEL_DEPTH=1

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
