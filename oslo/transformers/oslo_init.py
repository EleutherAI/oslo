import json
import logging
from enum import Enum
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple

from oslo.torch.distributed import ParallelContext
from oslo.torch.distributed.parallel_mode import ParallelMode
from oslo.torch.nn.parallel import (
    PipelineParallel,
    TensorParallel,
)
from .trainer_utils import log_dist

NoneType = type(None)


def _type(_dtype):
    return lambda key, val: {
        "check": isinstance(val, _dtype),
        "msg": f"{key}: {val} is not a valid set. it must be type of {_dtype}",
    }


def _values(*args):
    return lambda key, val: {
        "check": val in args,
        "msg": f"{key}: {val} is not a valid set. it must be one of {list(args)}",
    }


TENSOR_PARALLEL_MAPPING = {
    "1d": ParallelMode.TENSOR_1D,
    "2d": ParallelMode.TENSOR_2D,
    "3d": ParallelMode.TENSOR_3D,
    "2.5d": ParallelMode.TENSOR_2P5D,
}


class SupportedBackend(Enum):
    TORCH = "torch"
    SLURM = "slurm"
    OPENMPI = "openmpi"


SUPPORTED_FEATURES = {
    "backend": {"name": str, "host": str, "port": str},
    "mixed_precision": {
        "enable": _type(bool),
    },
    "activation_checkpointing": {
        "partitioned_checkpointing": _type(bool),
        "contiguous_checkpointing": _type(bool),
        "cpu_checkpointing": _type(bool),
    },
    "sequence_parallelism": {
        "enable": _type(bool),
        "parallel_size": _type(int),
    },
    "data_parallelism": {
        "enable": _type(bool),
        "parallel_size": _type(int),
        "zero_stage": _values(0, 1, 2, 3),
        # "params": lambda stage: DATA_PARALLEL_CONFIGS_BY_ZERO_STAGE[stage],
    },
    "tensor_parallelism": {
        "enable": _type(bool),
        "parallel_size": _type(int),
        "parallel_mode": _values(*TENSOR_PARALLEL_MAPPING.keys()),
        "params": {
            "parallel_depth_2.5d": _type(int),
        },
    },
    "pipeline_parallelism": {
        "enable": _type(bool),
        "parallel_size": _type(int),
        "params": {
            "memory_computation_balance": _type(float),
            "num_micro_batches": _type(int),
        },
    },
    "expert_parallelism": {
        "enable": _type(bool),
        "parallel_size": _type(int),
        "params": {
            "top_k": _type(int),
            "capacity_factor_train": _type(int),
            "capacity_factor_eval": _type(int),
            "select_policy": _values("first", "random"),
            "noisy_policy": _values("jitter", "gaussian"),
            "drop_tokens": _type(bool),
            "use_rts": _type(bool),
            "use_residual": _type(bool),
        },
    },
}


def _config_check(arg, user_config):
    # assert len(user_config) > 0, "There are no arguments in dictionary."

    if isinstance(user_config, dict):
        for k in user_config:
            if isinstance(arg, dict):
                assert k in arg, (
                    f"An argument ``{k}`` is not available. "
                    f"We only support the arguments like {list(arg.keys())}."
                )
            else:
                raise Exception(
                    f"``{k}: {user_config[k]} is not a valid set. "
                    f"please check your configuration.``"
                )

            if isinstance(user_config[k], dict):
                _config_check(arg[k], user_config[k])
            else:
                assert not isinstance(arg[k], dict), (
                    f"``{k}: {user_config[k]} is not a valid set. "
                    f"please check your configuration.``"
                )
                check_result = arg[k](k, user_config[k])
                assert check_result["check"], check_result["msg"]
    else:
        raise TypeError("configuration must be type of <class 'dict'>")


class OsloTrainerConfig:
    """
    This object contains a Oslo feature configuration dictionary

    [Oslo `TrainingArguments`] uses this class to set oslo features includes parallel, fused optimizer etc.
    json file or dictionary form should be like the following:
        SUPPORTED_FEATURES = {
            "backend": {
                "name": str,
                "host": str,
                "port": str
            },
            "mixed_precision": {
                "enable": _type(bool),
            },
            "activation_checkpointing": {
                "partitioned_checkpointing": _type(bool),
                "contiguous_checkpointing": _type(bool),
                "cpu_checkpointing": _type(bool),
            },
            "sequence_parallelism": {
                "enable": _type(bool),
                "parallel_size": _type(int),
            },
            "data_parallelism": {
                "enable": _type(bool),
                "parallel_size": _type(int),
                "zero_stage": _values(0, 1, 2, 3),
                "params": lambda stage: DATA_PARALLEL_CONFIGS_BY_ZERO_STAGE[stage],
            },
            "tensor_parallelism": {
                "enable": _type(bool),
                "parallel_size": _type(int),
                "parallel_mode": _values(*TENSOR_PARALLEL_MAPPING.keys()),
                "params": {
                    "parallel_depth_2.5d": _type(int),
                },
            },
            "pipeline_parallelism": {
                "enable": _type(bool),
                "parallel_size": _type(int),
                "params": {
                    "memory_computation_balance": _type(float),
                    "num_micro_batches": _type(int)
                },
            },
            "expert_parallelism": {
                "enable": _type(bool),
                "parallel_size": _type(int),
                "params": {
                    "top_k": _type(int),
                    "capacity_factor_train": _type(int),
                    "capacity_factor_eval": _type(int),
                    "select_policy": _values("first", "random"),
                    "noisy_policy": _values("jitter", "gaussian"),
                    "drop_tokens": _type(bool),
                    "use_rts": _type(bool),
                    "use_residual": _type(bool),
                },
            },
        }


    Args:
        config_file_or_dict (`Union[str, Dict]`): path to Oslo user config file or dict.

    """

    def __init__(self, config_file_or_dict):
        self.cpu_offload = False
        self.mixed_precision = False
        self.activation_checkpointing = None
        self.sequence_parallelism = None
        self.data_parallelism = None
        self.tensor_parallelism = None
        self.pipeline_parallelism = None
        self.expert_parallelism = None
        self.backend = None
        self.host = None
        self.port = None

        if isinstance(config_file_or_dict, dict):
            # Don't modify user's data should they want to reuse it (e.g. in tests), because once we
            # modified it, it will not be accepted here again, since `auto` values would have been overridden
            cfg = deepcopy(config_file_or_dict)
        elif isinstance(config_file_or_dict, str):
            with open(config_file_or_dict, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            raise ValueError("Expecting either a path to a oslo config file or a dict")
        _config_check(SUPPORTED_FEATURES, cfg)

        log_dist("*** OSLO CONFIG ***")

        if "backend" not in cfg:
            self.backend = SupportedBackend.TORCH
        elif cfg["backend"] in SupportedBackend:
            self.backend = SupportedBackend[cfg["backend"]]
            if self.backend in [SupportedBackend.OPENMPI]:
                if "host" in cfg["backend"]:
                    self.host = cfg["backend"]["host"]
                    log_dist(f"host: {self.host}")
                else:
                    raise ValueError(f"host is required to use {self.backend}")
                if "port" in cfg["backend"]:
                    self.port = cfg["backend"]["port"]
                    log_dist(f"host: {self.host}")
                else:
                    raise ValueError(f"post is required to use {self.backend}")
        log_dist(f"backend engine: {self.backend}")

        if "mixed_precision" in cfg and cfg["mixed_precision"]["enable"] is True:
            self.mixed_precision = True
            log_dist("mixed_precision: enabled")

        if "data_parallelism" in cfg and cfg["data_parallelism"]["enable"] is True:
            if cfg["data_parallelism"]["parallel_size"] is None:
                raise ValueError(
                    f"data_parallelism can not be usable because parallel_size is required."
                )
            elif cfg["data_parallelism"]["zero_stage"] is None:
                raise ValueError(
                    f"data_parallelism can not be usable because zero_stage is required."
                )
            else:
                if (
                    "params" in cfg["data_parallelism"]
                    and cfg["data_parallelism"]["params"]["cpu_offload"]
                ):
                    self.cpu_offload = True
                self.data_parallelism = cfg["data_parallelism"]
                if "params" not in self.data_parallelism:
                    self.data_parallelism["params"] = {}
                log_dist(
                    f"data_parallelism: enabled"
                    f"\tparallel_size: {self.data_parallelism['parallel_size']}"
                    f"\tzero_stage: {self.data_parallelism['zero_stage']}"
                    f"\tcpu_offload: {self.cpu_offload}"
                )

        if (
            "sequence_parallelism" in cfg
            and cfg["sequence_parallelism"]["enable"] is True
        ):
            if cfg["sequence_parallelism"]["parallel_size"] is None:
                raise ValueError(
                    f"sequence_parallelism can not be usable because parallel_size is required."
                )
            else:
                self.sequence_parallelism = cfg["sequence_parallelism"]
                if "params" not in self.sequence_parallelism:
                    self.sequence_parallelism["params"] = {}
                log_dist(
                    f"sequence_parallelism: enabled\n\tparallel_size: {self.sequence_parallelism['parallel_size']}"
                )

        if "tensor_parallelism" in cfg and cfg["tensor_parallelism"]["enable"] is True:
            if cfg["tensor_parallelism"]["parallel_size"] is None:
                raise ValueError(
                    "tensor_parallelism can not be usable because parallel_size is required."
                )
            elif cfg["tensor_parallelism"]["parallel_mode"] is None:
                raise ValueError(
                    "tensor_parallelism can not be usable because parallel_mode is required."
                )
            else:
                self.tensor_parallelism = cfg["tensor_parallelism"]
                if "params" not in self.tensor_parallelism:
                    self.tensor_parallelism["params"] = {}
                log_dist(
                    f"tensor_parallelism: enabled\n\tparallel_size: {self.tensor_parallelism['parallel_size']}\n\tparallel_mode: {self.tensor_parallelism['parallel_mode']}"
                )

        if (
            "pipeline_parallelism" in cfg
            and cfg["pipeline_parallelism"]["enable"] is True
        ):
            if cfg["pipeline_parallelism"]["parallel_size"] is None:
                raise ValueError(
                    "pipeline_parallelism can not be usable because parallel_size is required."
                )
            else:
                self.pipeline_parallelism = cfg["pipeline_parallelism"]
                if "params" not in self.pipeline_parallelism:
                    self.pipeline_parallelism["params"] = {}
                log_dist(
                    f"pipeline_parallelism: enabled\n\tparallel_size: {self.pipeline_parallelism['parallel_size']}"
                )

        if "expert_parallelism" in cfg and cfg["expert_parallelism"]["enable"] is True:
            if cfg["expert_parallelism"]["parallel_size"] is None:
                raise ValueError(
                    "expert_parallelism can not be usable because parallel_size is required."
                )
            else:
                self.expert_parallelism = cfg["expert_parallelism"]
                if "params" not in self.expert_parallelism:
                    self.expert_parallelism["params"] = {}
                log_dist(
                    f"expert_parallelism: enabled\n\tparallel_size: {self.expert_parallelism['parallel_size']}"
                )

    def is_exist(self, item):
        if item not in self.__dict__:
            return False
        return True

    def __getitem__(self, item):
        if not self.is_exist(item):
            if item == "params":
                return {}
            return None
        else:
            return getattr(self, item)

    def __repr__(self):
        return str(self.__dict__.items())


def init_oslo_features(
    oslo_init_config: OsloTrainerConfig,
) -> Tuple[ParallelContext, List]:
    """
    Init OSLO features with json or dict configuration user passed.
    ParallelContext or other effective features should be defined on this function
    and Trainer could use this outputs

    This function returns two object, ParallelContext and WrapperModule from user config
    TrainArgumet class use this to re-define model
    >> model = ...
    >> parallel_context = ParallelContext.from_torch(...)
    >> wrapper_model = TensorParallel(model, parallel_context)
    >> allocate_params(wrapper_model, parallel_context)
    """
    cfg = oslo_init_config
    data_parallel_size = (
        cfg.data_parallelism["parallel_size"] if cfg.data_parallelism else 1
    )
    sequence_parallel_size = (
        cfg.sequence_parallelism["parallel_size"] if cfg.sequence_parallelism else 1
    )
    expert_parallel_size = (
        cfg.expert_parallelism["parallel_size"] if cfg.expert_parallelism else 1
    )
    pipeline_parallel_size = (
        cfg.pipeline_parallelism["parallel_size"] if cfg.pipeline_parallelism else 1
    )
    tensor_parallel_size, tensor_parallel_depth, tensor_parallel_mode = (
        1,
        1,
        TENSOR_PARALLEL_MAPPING["1d"],
    )
    if cfg.tensor_parallelism:
        tensor_parallel_size = cfg.tensor_parallelism["parallel_size"]
        tensor_parallel_mode = TENSOR_PARALLEL_MAPPING[
            cfg.tensor_parallelism["parallel_mode"]
        ]
        if (
            "param" in cfg.tensor_parallelism
            and "parallel_depth_2.5d" in cfg.tensor_parallelism["param"]
        ):
            tensor_parallel_depth = cfg.tensor_parallelism["param"][
                "parallel_depth_2.5d"
            ]

    if cfg.backend == SupportedBackend.TORCH:
        parallel_context = ParallelContext.from_torch(
            data_parallel_size=data_parallel_size,
            sequence_parallel_size=sequence_parallel_size,
            expert_parallel_size=expert_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_depth=tensor_parallel_depth,
            tensor_parallel_mode=tensor_parallel_mode,
        )

    elif cfg.backend == SupportedBackend.SLURM:
        parallel_context = ParallelContext.from_slurm(
            host=cfg.host,
            port=cfg.port,
            data_parallel_size=data_parallel_size,
            sequence_parallel_size=sequence_parallel_size,
            expert_parallel_size=expert_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_depth=tensor_parallel_depth,
            tensor_parallel_mode=tensor_parallel_mode,
        )

    elif cfg.backend == SupportedBackend.OPENMPI:
        parallel_context = ParallelContext.from_openmpi(
            host=cfg.host,
            port=cfg.port,
            data_parallel_size=data_parallel_size,
            sequence_parallel_size=sequence_parallel_size,
            expert_parallel_size=expert_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_depth=tensor_parallel_depth,
            tensor_parallel_mode=tensor_parallel_mode,
        )
    else:
        raise ValueError(f"Wrong backend config: {cfg.backend}")

    if tensor_parallel_size > 1 and sequence_parallel_size > 1:
        raise ValueError(
            "TensorParallel and SequenceParallel can't be used at the same time. Modify oslo config to avoid wrong parallel setting"
        )

    model_wrapper = []

    if tensor_parallel_size > 1:
        model_wrapper.append(TensorParallel)
    if pipeline_parallel_size > 1:
        model_wrapper.append(PipelineParallel)
    # TODO expert mode
    return parallel_context, model_wrapper
