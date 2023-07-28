PYTHONPATH=. CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node 4 --master_port 12222 tests_deprecated/torch/nn/parallel/pipeline_parallel/test_pp.py
#PYTHONPATH=. CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node 2 --master_port 12222 tests_deprecated/torch/nn/parallel/pipeline_parallel/test_tp.py
#PYTHONPATH=. python tests_deprecated/torch/nn/parallel/pipeline_parallel/compare_output_pptp_no.py > pptp_no.txt
#PYTHONPATH=. python tests_deprecated/torch/nn/parallel/pipeline_parallel/compare_output_tp_no.py > tp_no.txt
#PYTHONPATH=. python tests_deprecated/torch/nn/parallel/pipeline_parallel/compare_grad_pptp_no.py > grad_pptp_no.txt
#PYTHONPATH=. python tests_deprecated/torch/nn/parallel/pipeline_parallel/compare_grad_tp_no.py > grad_tp_no.txt

#PYTHONPATH=. python tests_deprecated/torch/nn/parallel/pipeline_parallel/compare_output_pp_nopp.py > pp_nopp.txt
#PYTHONPATH=. python tests_deprecated/torch/nn/parallel/pipeline_parallel/compare_output_pp_nopp.py
#PYTHONPATH=. python tests_deprecated/torch/nn/parallel/pipeline_parallel/compare_grad_pp_nopp.py

