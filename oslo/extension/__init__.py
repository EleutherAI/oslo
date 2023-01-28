import os
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils import cpp_extension

from oslo.torch.jit._utils import _set_jit_fusion_options


DEFAULT_TORCH_EXTENSION_PATH = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "torch_extensions",
    "oslo",
)

class ExtensionBinder(object):
    def __init__(self, sources):
        """
        sources : ["layers/cross_entropy_layer.cpp", "kernels/cross_entropy_kernel.cu", "pybind/pybind_cross_entropy_layer.cpp"]
        """
        self.compat = self.get_compatibility_version()
        self.sources = sources
        self.dirnames = [os.path.dirname(source) for source in self.sources]
        
    @property
    def base_path(self):
        from oslo.extension import csrc

        return Path(csrc.__file__).parent.absolute()

    @property
    def name(self):
        return "oslo"

    def includes(self):
        return [
            os.path.join(base_path, "includes") for base_path in self.base_paths
        ]

    def sources(self):
        return self.sources

    @staticmethod
    def get_compatibility_version():
        a, b = torch.cuda.get_device_capability(torch.cuda.current_device())
        return int(str(a) + str(b))

    def bind(self, verbose: bool = False):
        try:
            import ninja
            import pybind11

        except ImportError:
            raise ImportError(
                "Unable to compile C++ code due to ``ninja`` or ``pybind11`` not being installed. "
                "please install them using ``pip install ninja pybind11``."
            )

        # Ensure directory exists to prevent race condition in some cases
        ext_path = os.environ.get("TORCH_EXTENSIONS_DIR", DEFAULT_TORCH_EXTENSION_PATH)
        ext_path = os.path.join(ext_path, self.name)
        os.makedirs(ext_path, exist_ok=True)

        op_module = cpp_extension.load(
            name=self.name,
            sources=[os.path.join(self.base_path, path) for path in self.sources()],
            extra_include_paths=self.includes(),
            extra_cflags=self.cxx_args(),
            extra_cuda_cflags=self.nvcc_args(),
            verbose=verbose,
            extra_ldflags=self.strip_empty_entries(self.extra_ldflags()),
        )

        return op_module

    @staticmethod
    def cxx_args():
        if sys.platform == "win32":
            return [
                "-O2",
                "-Wno-reorder",
                "-Wno-deprecated",
                "-Wno-deprecated-declarations",
            ]
        else:
            return [
                "-O3",
                "-std=c++14",
                "-g",
                "-Wno-reorder",
                "-Wno-deprecated",
                "-Wno-deprecated-declarations",
            ]

    def nvcc_args(self, maxrregcount: int = None):
        nvcc_flags = [
            "-O3",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ]

        additional_flags = [
            "-gencode",
            f"arch=compute_{self.compat},code=sm_{self.compat}",
        ]

        if maxrregcount:
            additional_flags.append(f"-maxrregcount={maxrregcount}")

        return nvcc_flags + additional_flags

    def extra_ldflags(self):
        return []

    @staticmethod
    def strip_empty_entries(args):
        """
        Drop any empty strings from the list of compile and link flags
        """
        return [x for x in args if len(x) > 0]


def get_extension_cross_entropy_layer():
    global _EXTENSION_CROSS_ENTROPY_LAYER

    try:
        if _EXTENSION_CROSS_ENTROPY_LAYER is None:
            _set_jit_fusion_options()
            _EXTENSION_CROSS_ENTROPY_LAYER = ExtensionCrossEntropyLayerBinder().bind()
    except Exception:
        raise EnvironmentError(
            "Failed compiling custom CUDA kernels. "
            "please check your CUDA environment."
        )

    return _EXTENSION_CROSS_ENTROPY_LAYER


class ExtensionCrossEntropyLayerBinder(ExtensionBinder):
    @property
    def name(self):
        return "oslo_cross_entropy_layer"

    def sources(self):
        return [
            "layers/cross_entropy_layer.cpp", 
            "kernels/cross_entropy_kernel.cu", 
            "pybind/pybind_cross_entropy_layer.cpp"
        ]