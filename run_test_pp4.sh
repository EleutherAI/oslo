PYTHONPATH=. CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port 12222 tests_deprecated/torch/nn/parallel/pipeline_parallel/test_pp4.py
#PYTHONPATH=. CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node 2 --master_port 12222 tests_deprecated/torch/nn/parallel/pipeline_parallel/test_pp4.py
