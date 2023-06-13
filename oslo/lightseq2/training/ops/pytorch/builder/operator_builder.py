# Copyright 2021 The LightSeq Team
# Copyright Microsoft DeepSpeed
# This file is adapted from Microsoft DeepSpeed

import torch
import pathlib
from .builder import CUDAOpBuilder
from .builder import installed_cuda_version


class OperatorBuilder(CUDAOpBuilder):
    NAME = "lightseq_operator"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f"op_builder.{self.NAME}_op"

    def sources(self):
        return [
            "csrc/kernels/cuda/cublas_wrappers.cu",
            "csrc/kernels/cuda/transform_kernels.cu",
            "csrc/kernels/cuda/transform_kernels_new.cu",
            "csrc/kernels/cuda/dropout_kernels.cu",
            "csrc/kernels/cuda/normalize_kernels.cu",
            "csrc/kernels/cuda/softmax_kernels_new.cu",
            "csrc/kernels/cuda/softmax_kernels.cu",
            "csrc/kernels/cuda/general_kernels.cu",
            "csrc/kernels/cuda/cuda_util.cu",
            "csrc/kernels/cuda/embedding_kernels.cu",
            "csrc/kernels/cuda/cross_entropy.cu",
            "csrc/lsflow/allocator.cpp",
            "csrc/lsflow/context.cpp",
            "csrc/lsflow/layer.cpp",
            "csrc/lsflow/lsflow_util.cpp",
            "csrc/lsflow/manager.cpp",
            "csrc/lsflow/node.cpp",
            "csrc/lsflow/operator.cpp",
            "csrc/lsflow/shape.cpp",
            "csrc/lsflow/tensor.cpp",
            "csrc/lsflow/variable.cpp",
            "csrc/ops_new/split_head_op.cpp",
            "csrc/pybind/pybind_op.cpp",
        ]

    def include_paths(self):
        paths = [
            "csrc/kernels/cuda/includes",
            "csrc/ops_new/includes",
            "csrc/lsflow/includes",
            "csrc/models/includes",
        ]
        cuda_major, cuda_minor = installed_cuda_version()
        if cuda_major < 11:
            paths.append(str(pathlib.Path(__file__).parents[4] / "3rdparty" / "cub"))
        return paths

    def nvcc_args(self):
        args = [
            "-O3",
            "--use_fast_math",
            "-std=c++14",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
        ]

        return args + self.compute_capability_args()

    def cxx_args(self):
        return [
            "-O3",
            "-std=c++14",
            "-g",
            "-Wno-reorder",
            "-DONLY_OP",
            "-DPYBIND_INTERFACE",
            "-DLIGHTSEQ_cuda",
        ]
