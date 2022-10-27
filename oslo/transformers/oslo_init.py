import json
import logging
from copy import deepcopy
from typing import List, Tuple
from dataclasses import dataclass
from oslo.torch.distributed import ParallelContext
from oslo.torch.nn.parallel import (
    PipelineParallel,
    TensorParallel,
)
from oslo.torch.nn.parallel.sequence_parallel import SequenceParallel
from oslo.torch.nn.parallel.data_parallel.data_parallel import DataParallel
from oslo.torch.nn.parallel.data_parallel._ddp.distributed_data_parallel import (
    DistributedDataParallel,
)
from oslo.torch.distributed.parallel_mode import ParallelMode
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


SUPPORTED_FEATURES = {
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
            "memory_priority": _type(bool),
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


@dataclass
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.mkconfig(self)

    @staticmethod
    def mkconfig(obj):
        for k, v in obj.__dict__.items():
            if isinstance(v, dict):
                obj.__setattr__(k, Config(**v))

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


class OsloTrainerConfig(Config):
    """
    This object contains a Oslo feature configuration dictionary

    [Oslo `TrainingArguments`] uses this class to set oslo features includes parallel, fused optimizer etc.
    json file or dictionary form should be like the following:
        SUPPORTED_FEATURES = {
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
                    "memory_priority": _type(bool),
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
        super(OsloTrainerConfig, self).__init__()
        self.cpu_offload = False
        if isinstance(config_file_or_dict, dict):
            # Don't modify user's data should they want to reuse it (e.g. in tests), because once we
            # modified it, it will not be accepted here again, since `auto` values would have been overridden
            cfg = deepcopy(config_file_or_dict)
        elif isinstance(config_file_or_dict, str):
            with open(config_file_or_dict, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            raise ValueError("expecting either a path to a oslo config file or a dict")
        _config_check(SUPPORTED_FEATURES, cfg)
        super(OsloTrainerConfig, self).__init__(**cfg)
        log_dist("*** OSLO CONFIG ***")
        if not self.is_exist("mixed_precision") or not self.mixed_precision["enable"]:
            self.mixed_precision = None
        else:
            log_dist("mixed_precision: enabled")

        if not self.is_exist("data_parallelism") or not self.data_parallelism["enable"]:
            self.data_parallelism = None
        else:
            if self.data_parallelism["parallel_size"] is None:
                log_dist(
                    "data_parallelism can not be usable because parallel_size is required.",
                    logging.WARNING,
                )
                self.data_parallelism = None

            elif self.data_parallelism["zero_stage"] is None:
                logging.warning(
                    "data_parallelism can not be usable because zero_stage is required."
                )
                self.data_parallelism = None
            else:
                log_dist(
                    f"data_parallelism: enabled\n\tparallel_size: {self.data_parallelism['parallel_size']}\n\tzero_stage: {self.data_parallelism['zero_stage']}"
                )
                if (
                    hasattr(self.data_parallelism, "params")
                    and self.data_parallelism.params["cpu_offload"]
                ):
                    self.cpu_offload = True

        if (
            not self.is_exist("sequence_parallelism")
            or not self.sequence_parallelism["enable"]
        ):
            self.sequence_parallelism = None
        else:
            if self.sequence_parallelism["parallel_size"] is None:
                log_dist(
                    "sequence_parallelism can not be usable because parallel_size is required.",
                    logging.WARNING,
                )
                self.sequence_parallelism = None
            else:
                log_dist(
                    f"sequence_parallelism: enabled\n\tparallel_size: {self.sequence_parallelism['parallel_size']}"
                )

        if (
            not self.is_exist("tensor_parallelism")
            or not self.tensor_parallelism["enable"]
        ):
            self.tensor_parallelism = None
        else:
            if self.tensor_parallelism["parallel_size"] is None:
                ValueError(
                    "tensor_parallelism can not be usable because parallel_size is required."
                )
            elif self.tensor_parallelism["parallel_mode"] is None:
                log_dist(
                    "tensor_parallelism can not be usable because parallel_mode is required.",
                    logging.WARNING,
                )
                self.tensor_parallelism = None
            else:
                log_dist(
                    f"tensor_parallelism: enabled\n\tparallel_size: {self.tensor_parallelism['parallel_size']}\n\tparallel_mode: {self.tensor_parallelism['parallel_mode']}"
                )

        if (
            not self.is_exist("pipeline_parallelism")
            or not self.pipeline_parallelism["enable"]
        ):
            self.pipeline_parallelism = None
        else:
            if self.pipeline_parallelism["parallel_size"] is None:
                log_dist(
                    "pipeline_parallelism can not be usable because parallel_size is required.",
                    logging.WARNING,
                )
                self.pipeline_parallelism = None
            else:
                log_dist(
                    f"pipeline_parallelism: enabled\n\tparallel_size: {self.pipeline_parallelism['parallel_size']}"
                )

        if (
            not self.is_exist("expert_parallelism")
            or not self.expert_parallelism["enable"]
        ):
            self.expert_parallelism = None
        else:
            if self.expert_parallelism["parallel_size"] is None:
                log_dist(
                    "expert_parallelism can not be usable because parallel_size is required.",
                    logging.WARNING,
                )
                self.expert_parallelism = None
            else:
                log_dist(
                    f"expert_parallelism: enabled\n\tparallel_size: {self.expert_parallelism['parallel_size']}"
                )


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
        cfg.data_parallelism.parallel_size if cfg.data_parallelism else 1
    )
    sequence_parallel_size = (
        cfg.sequence_parallelism.parallel_size if cfg.sequence_parallelism else 1
    )
    expert_parallel_size = (
        cfg.expert_parallelism.parallel_size if cfg.expert_parallelism else 1
    )
    pipeline_parallel_size = (
        cfg.pipeline_parallelism.parallel_size if cfg.pipeline_parallelism else 1
    )
    tensor_parallel_size, tensor_parallel_depth, tensor_parallel_mode = (
        1,
        1,
        TENSOR_PARALLEL_MAPPING["1d"],
    )
    if cfg.tensor_parallelism:
        tensor_parallel_size = cfg.tensor_parallelism.parallel_size
        tensor_parallel_mode = TENSOR_PARALLEL_MAPPING[
            cfg.tensor_parallelism.parallel_mode
        ]
        if cfg.tensor_parallelism.is_exist("param"):
            tensor_parallel_depth = cfg.tensor_parallelism.param["parallel_depth_2.5d"]

    parallel_context = ParallelContext.from_torch(
        data_parallel_size=data_parallel_size,
        sequence_parallel_size=sequence_parallel_size,
        expert_parallel_size=expert_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_depth=tensor_parallel_depth,
        tensor_parallel_mode=tensor_parallel_mode,
    )

    if tensor_parallel_size > 1 and sequence_parallel_size > 1:
        raise ValueError(
            "TensorParallel and SequenceParallel can't be used at the same time. Modify oslo config to avoid wrong parallel setting"
        )

    model_wrapper = []

    if data_parallel_size > 1:
        model_wrapper.append(DataParallel)
    if tensor_parallel_size > 1:
        model_wrapper.append(TensorParallel)
    if sequence_parallel_size > 1:
        model_wrapper.append(SequenceParallel)
    if pipeline_parallel_size > 1:
        model_wrapper.append(PipelineParallel)
    # TODO expert mode

    return parallel_context, model_wrapper
