import torch
from oslo.torch.nn.parallel.data_parallel.zero.memory_tracer.memory_stats import (
    MemStats,
)


def test_mem_stats():
    # initialize MemStats object
    mem_stats = MemStats()

    # test increase_preop_step method
    p1 = torch.nn.Parameter(torch.randn(3, 3))
    p2 = torch.nn.Parameter(torch.randn(3, 3))
    mem_stats.increase_preop_step([p1, p2])
    assert len(mem_stats._step_param_dict) == 1
    assert len(mem_stats._param_step_dict) == 2
    assert mem_stats.param_used_step(p1) == [0]
    assert mem_stats.param_used_step(p2) == [0]

    # test record_max_cuda_model_data method
    mem_stats.record_max_cuda_model_data(100)
    assert mem_stats._prev_md_cuda == 100

    # test record_max_cuda_overall_data method
    mem_stats.record_max_cuda_overall_data(200)
    assert mem_stats._prev_overall_cuda == 200
    assert mem_stats._max_overall_cuda == 200

    # test calc_max_cuda_non_model_data method
    mem_stats.record_max_cuda_overall_data(400)
    mem_stats.calc_max_cuda_non_model_data()
    assert mem_stats._step_nmd_dict[0] == 300
    assert mem_stats.non_model_data_list("cuda") == [300]
    assert mem_stats.max_non_model_data("cuda") == 300

    # test clear method
    mem_stats.clear()
    assert len(mem_stats._step_param_dict) == 0
    assert len(mem_stats._param_step_dict) == 0
    assert mem_stats.param_used_step(p1) is None
    assert mem_stats.param_used_step(p2) is None
    assert mem_stats._preop_step == 0
    assert mem_stats._prev_overall_cuda == -1
    assert mem_stats._prev_md_cuda == -1
