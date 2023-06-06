## inference shell code
# EXAMPLE: ``sh ./run_inference.sh 4 bert-base-cased masked-lm ``

NUM_GPUS=$1
MODEL=$2
TASK=$3

python -m torch.distributed.launch \
       --nproc_per_node="$NUM_GPUS" \
       ../inference.py \
       --task=$TASK \
       --model=$MODEL \
       --tensor_parallel_size="$NUM_GPUS"
