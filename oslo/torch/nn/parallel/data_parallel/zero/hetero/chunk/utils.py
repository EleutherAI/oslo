# Copyright 2021 HPC-AI Technology Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by EleutherAI on 2023.

from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from oslo.torch.nn.parallel.data_parallel.zero.hetero.chunk.manager import (
    ChunkManager,
)

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch.distributed as dist
import torch.nn as nn

from oslo.torch.nn.parallel.data_parallel._utils import (
    is_ddp_ignored,
)
from oslo.torch.nn.parallel.data_parallel.zero.hetero.memory_tracer import (
    MemStats,
    OrderedParamGenerator,
)
from oslo.torch.distributed.parallel_mode import ParallelMode

from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.distributed.parallel_mode import ParallelMode


def _filter_exlarge_params(model: nn.Module, size_dict: Dict[int, List[int]]):
    """
    Filter those parameters whose size is too large (more than 3x standard deviations) from others.

    Args:
        model (nn.Module): the model
        size_dict (Dict[int, List[int]]): the size dict
    """
    agg_size_list = []
    for key in size_dict:
        agg_size_list.extend(size_dict[key])

    if len(agg_size_list) == 0:
        return

    params_size_arr = np.array(agg_size_list)

    std = np.std(params_size_arr)
    mean = np.mean(params_size_arr)
    upper_limit = mean + 3 * std

    for key in size_dict:
        org_list = size_dict[key]
        size_dict[key] = list(filter(lambda x: x <= upper_limit, org_list))


def _get_unused_byte(size_list: List[int], chunk_size: int) -> int:
    """Get unused byte for a certain chunk size.

    Args:
        size_list (List[int]): the size list
        chunk_size (int): the chunk size

    Returns:
        int: the unused byte
    """
    acc = 0
    left = 0
    for s in size_list:
        if s > left:
            acc += left
            left = chunk_size
        left -= s
    return left + acc


def classify_params_by_dp_degree(
    param_order: OrderedParamGenerator,
) -> Dict[int, List[torch.Tensor]]:
    """classify_params_by_dp_degree

    Classify the parameters by their dp degree

    Args:
        param_order (OrderedParamGenerator): the order of param be visied

    Returns:
        Dict[int, List[torch.Tensor]]: a dict contains the classification results.
        The keys are dp_degrees and the values are parameters.
    """
    params_dict: Dict[int, List[torch.Tensor]] = dict()
    for param in param_order.generate():
        if is_ddp_ignored(param):
            continue

        param_key = ParallelContext.get_context().get_world_size(ParallelMode.DATA)

        if param_key not in params_dict:
            params_dict[param_key] = []
        params_dict[param_key].append(param)

    return params_dict


def search_chunk_configuration(
    model: nn.Module,
    search_range_mb: float,
    search_interval_byte: int,  # hidden size is the best value for the interval
    min_chunk_size_mb: float = 32,
    filter_exlarge_params: bool = True,
    memstas: Optional[MemStats] = None,
) -> Tuple[Dict, int, int]:
    """search_chunk_configuration

    Args:
        model (nn.Module): torch module
        search_range_mb (float): searching range in mega byte.
        search_interval_byte (int): searching interval in byte.
        min_chunk_size_mb (float, optional): the minimum size of a chunk.
        filter_exlarge_params (bool, optional): filter extreme large parameters. Defaults to True.

    Returns:
        Tuple[Dict, int]: chunk config (a dict of dp_degree -> chunk init args) and its memory chunk waste in byte.
    """

    if memstas is not None:
        param_order = memstas.param_order()
    else:
        # build the param visited order right now
        param_order = OrderedParamGenerator()
        for p in model.parameters():
            param_order.append(p)

    search_range_byte = round(search_range_mb * 1024**2)
    min_chunk_size_byte = round(min_chunk_size_mb * 1024**2)
    assert search_range_byte >= 0

    params_dict = classify_params_by_dp_degree(param_order)
    size_lcm = np.lcm.reduce(list(params_dict.keys()))
    config_dict: Dict[int, Dict] = dict()
    total_param_size = 0

    size_dict: Dict[int, List[int]] = dict()
    for dp_degree in params_dict:
        params_list = params_dict[dp_degree]
        size_list = [p.numel() for p in params_list]
        group_acc_size = sum(size_list)
        total_param_size += group_acc_size

        # let small parameters keep gathered in CUDA all the time
        if group_acc_size < min_chunk_size_byte:
            config_dict[dp_degree] = dict(chunk_size=group_acc_size, keep_gathered=True)
        else:
            size_dict[dp_degree] = size_list

    if filter_exlarge_params:
        _filter_exlarge_params(model, size_dict)

    max_size = min_chunk_size_byte
    for key in size_dict:
        max_size = max(max_size, max(size_dict[key]))
    start_size = int(math.ceil(max_size / search_interval_byte) * search_interval_byte)

    min_chunk_waste = float("+inf")
    best_chunk_size = start_size

    for chunk_size in range(
        start_size, start_size + search_range_byte + 1, search_interval_byte
    ):
        temp_waste = 0
        for key in size_dict:
            temp_waste += _get_unused_byte(size_dict[key], chunk_size)
        if temp_waste < min_chunk_waste:
            min_chunk_waste = temp_waste
            best_chunk_size = chunk_size

    # the chunk size needs to be divided by each groups sizes
    best_chunk_size = best_chunk_size + (-best_chunk_size % size_lcm)
    for dp_degree in params_dict:
        if dp_degree in config_dict:
            continue
        config_dict[dp_degree] = dict(chunk_size=best_chunk_size, keep_gathered=False)

    return config_dict, total_param_size, min_chunk_waste


def init_chunk_manager(
    model: nn.Module,
    init_device: Optional[torch.device] = None,
    hidden_dim: Optional[int] = None,
    **kwargs
) -> ChunkManager:
    """init_chunk_manager

    Init the chunk manager with the given model.

    Args:
        model (nn.Module): torch module
        init_device (Optional[torch.device], optional): the device to init the chunk manager.
        hidden_dim (Optional[int], optional): the hidden dimension of the model.

    Returns:
        ChunkManager: the chunk manager
    """

    if hidden_dim:
        search_interval_byte = hidden_dim
    else:
        search_interval_byte = 1024  # defaults to 1kb
    kwargs["search_interval_byte"] = search_interval_byte

    config_dict, *_ = search_chunk_configuration(model, **kwargs)

    chunk_manager = ChunkManager(config_dict, init_device)
    return chunk_manager
