import os
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils import cpp_extension

from oslo.torch.jit._utils import _set_jit_fusion_options

_ADAM_KERNEL = None
_ADAGRAD_KERNEL = None
_NOVOGRAD_KERNEL = None
_SGD_KERNEL = None
_MIXED_PRECISION_LAMB_KERNEL = None
_LAMB_KERNEL = None
_CPU_ADAM_KERNEL = None
_CPU_ADAGRAD_KERNEL = None
_L2NORM_KERNEL = None
_MIXED_PRECISION_L2NORM_KERNEL = None
_LAYER_NORM_NORM_KERNEL = None
_EXPERT_PARALLEL_KERNEL = None
_NGRAM_REPEAT_BLOCK_KERNEL = None

YELLOW = "\033[93m"
END = "\033[0m"
WARNING = f"{YELLOW} [WARNING] {END}"

TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])


def get_adam_kernel():
    global _ADAM_KERNEL

    try:
        if _ADAM_KERNEL is None:
            _set_jit_fusion_options()
            _ADAM_KERNEL = FusedAdamBinder().bind()
    except Exception:
        raise EnvironmentError(
            "Failed compiling custom CUDA kernels. "
            "please check your CUDA environment."
        )

    return _ADAM_KERNEL


def get_adagrad_kernel():
    global _ADAGRAD_KERNEL

    try:
        if _ADAGRAD_KERNEL is None:
            _set_jit_fusion_options()
            _ADAGRAD_KERNEL = FusedAdagradBinder().bind()
    except Exception:
        raise EnvironmentError(
            "Failed compiling custom CUDA kernels. "
            "please check your CUDA environment."
        )

    return _ADAGRAD_KERNEL


def get_novograd_kernel():
    global _NOVOGRAD_KERNEL

    try:
        if _NOVOGRAD_KERNEL is None:
            _set_jit_fusion_options()
            _NOVOGRAD_KERNEL = FusedNovogradBinder().bind()
    except Exception:
        raise EnvironmentError(
            "Failed compiling custom CUDA kernels. "
            "please check your CUDA environment."
        )

    return _NOVOGRAD_KERNEL


def get_sgd_kernel():
    global _SGD_KERNEL

    try:
        if _SGD_KERNEL is None:
            _set_jit_fusion_options()
            _SGD_KERNEL = FusedSGDBinder().bind()
    except Exception:
        raise EnvironmentError(
            "Failed compiling custom CUDA kernels. "
            "please check your CUDA environment."
        )

    return _SGD_KERNEL


def get_l2norm_kernel():
    global _L2NORM_KERNEL

    try:
        if _L2NORM_KERNEL is None:
            _set_jit_fusion_options()
            _L2NORM_KERNEL = FusedL2NormBinder().bind()
    except Exception:
        raise EnvironmentError(
            "Failed compiling custom CUDA kernels. "
            "please check your CUDA environment."
        )

    return _L2NORM_KERNEL


def get_l2norm_mp_kernel():
    global _MIXED_PRECISION_L2NORM_KERNEL

    try:
        if _MIXED_PRECISION_L2NORM_KERNEL is None:
            _set_jit_fusion_options()
            _MIXED_PRECISION_L2NORM_KERNEL = FusedMixedPrecisionL2NormBinder().bind()
    except Exception:
        raise EnvironmentError(
            "Failed compiling custom CUDA kernels. "
            "please check your CUDA environment."
        )

    return _MIXED_PRECISION_L2NORM_KERNEL


def get_lamb_kernel():
    global _LAMB_KERNEL

    try:
        if _LAMB_KERNEL is None:
            _set_jit_fusion_options()
            _LAMB_KERNEL = FusedLambBinder().bind()
    except Exception:
        raise EnvironmentError(
            "Failed compiling custom CUDA kernels. "
            "please check your CUDA environment."
        )

    return _LAMB_KERNEL


def get_lamb_mp_kernel():
    global _MIXED_PRECISION_LAMB_KERNEL

    try:
        if _MIXED_PRECISION_LAMB_KERNEL is None:
            _set_jit_fusion_options()
            _MIXED_PRECISION_LAMB_KERNEL = FusedMixedPrecisionLambBinder().bind()
    except Exception:
        raise EnvironmentError(
            "Failed compiling custom CUDA kernels. "
            "please check your CUDA environment."
        )

    return _MIXED_PRECISION_LAMB_KERNEL


def get_cpu_adam_kernel():
    global _CPU_ADAM_KERNEL

    try:
        if _CPU_ADAM_KERNEL is None:
            _set_jit_fusion_options()
            _CPU_ADAM_KERNEL = CPUAdamBinder().bind()
    except Exception:
        raise EnvironmentError(
            "Failed compiling custom CUDA kernels. "
            "please check your CUDA environment."
        )

    return _CPU_ADAM_KERNEL


def get_cpu_adagrad_kernel():
    global _CPU_ADAGRAD_KERNEL

    try:
        if _CPU_ADAGRAD_KERNEL is None:
            _set_jit_fusion_options()
            _CPU_ADAGRAD_KERNEL = CPUAdagradBinder().bind()
    except Exception:
        raise EnvironmentError(
            "Failed compiling custom CUDA kernels. "
            "please check your CUDA environment."
        )

    return _CPU_ADAGRAD_KERNEL


def get_layernorm_kernel():
    global _LAYER_NORM_NORM_KERNEL

    try:
        if _LAYER_NORM_NORM_KERNEL is None:
            _set_jit_fusion_options()
            _LAYER_NORM_NORM_KERNEL = FusedLayerNormBinder().bind()
    except Exception:
        raise EnvironmentError(
            "Failed compiling custom CUDA kernels. "
            "please check your CUDA environment."
        )

    return _LAYER_NORM_NORM_KERNEL


def get_expert_parallel_kernel():
    global _EXPERT_PARALLEL_KERNEL

    try:
        if _EXPERT_PARALLEL_KERNEL is None:
            _set_jit_fusion_options()
            _EXPERT_PARALLEL_KERNEL = ExpertParallelBinder().bind()
    except Exception:
        raise EnvironmentError(
            "Failed compiling custom CUDA kernels. "
            "please check your CUDA environment."
        )

    return _EXPERT_PARALLEL_KERNEL


def get_ngram_repeat_block_kernel():
    global _NGRAM_REPEAT_BLOCK_KERNEL

    try:
        if _NGRAM_REPEAT_BLOCK_KERNEL is None:
            _set_jit_fusion_options()
            _NGRAM_REPEAT_BLOCK_KERNEL = NgramRepeatBlockBinder().bind()
    except Exception:
        raise EnvironmentError(
            "Failed compiling custom CUDA kernels. "
            "please check your CUDA environment."
        )

    return _NGRAM_REPEAT_BLOCK_KERNEL


DEFAULT_TORCH_EXTENSION_PATH = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "torch_extensions",
    "oslo",
)


class Binder(object):
    _is_rocm_version = None
    _is_rocm_pytorch = None

    def __init__(self):
        self.compat = self.get_compatibility_version()

    @property
    def base_path(self):
        from oslo.torch._C import csrc

        return Path(csrc.__file__).parent.absolute()

    @property
    def name(self):
        return "oslo"

    def includes(self):
        return [
            os.path.join(self.base_path, "includes"),
        ]

    def sources(self):
        return []

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
    def is_rocm_pytorch():
        if Binder._is_rocm_pytorch is not None:
            return Binder._is_rocm_pytorch

        _is_rocm_pytorch = False
        try:
            import torch
        except ImportError:
            pass
        else:
            if TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 5):
                _is_rocm_pytorch = (
                    hasattr(torch.version, "hip") and torch.version.hip is not None
                )
                if _is_rocm_pytorch:
                    from torch.utils.cpp_extension import ROCM_HOME

                    _is_rocm_pytorch = ROCM_HOME is not None
        Binder._is_rocm_pytorch = _is_rocm_pytorch
        return Binder._is_rocm_pytorch

    @staticmethod
    def installed_rocm_version():
        if Binder._rocm_version:
            return Binder._rocm_version

        ROCM_MAJOR = "0"
        ROCM_MINOR = "0"
        if Binder.is_rocm_pytorch():
            from torch.utils.cpp_extension import ROCM_HOME

            with open("/opt/rocm/.info/version-dev", "r") as file:
                ROCM_VERSION_DEV_RAW = file.read()
            ROCM_MAJOR = ROCM_VERSION_DEV_RAW.split(".")[0]
            ROCM_MINOR = ROCM_VERSION_DEV_RAW.split(".")[1]
        Binder._rocm_version = (int(ROCM_MAJOR), int(ROCM_MINOR))
        return Binder._rocm_version

    @staticmethod
    def strip_empty_entries(args):
        """
        Drop any empty strings from the list of compile and link flags
        """
        return [x for x in args if len(x) > 0]


class CPUBinder(Binder):
    def extra_ldflags(self):
        if not self.is_rocm_pytorch():
            return ["-lcurand"]
        else:
            return []

    def cxx_args(self):
        import torch

        if not self.is_rocm_pytorch():
            CUDA_LIB64 = os.path.join(torch.utils.cpp_extension.CUDA_HOME, "lib64")
        else:
            CUDA_LIB64 = os.path.join(torch.utils.cpp_extension.ROCM_HOME, "lib")
        CPU_ARCH = self.cpu_arch()
        SIMD_WIDTH = self.simd_width()

        args = super().cxx_args()
        args += [
            f"-L{CUDA_LIB64}",
            "-lcudart",
            "-lcublas",
            "-g",
            CPU_ARCH,
            "-fopenmp",
            SIMD_WIDTH,
        ]
        return args

    def warning(self, msg):
        self.error_log = f"{msg}"
        print(f"{WARNING} {msg}")

    def command_exists(self, cmd):
        if "|" in cmd:
            cmds = cmd.split("|")
        else:
            cmds = [cmd]
        valid = False
        for cmd in cmds:
            result = subprocess.Popen(f"type {cmd}", stdout=subprocess.PIPE, shell=True)
            valid = valid or result.wait() == 0

        if not valid and len(cmds) > 1:
            print(
                f"{WARNING} {self.name} requires one of the following commands '{cmds}', but it does not exist!"
            )
        elif not valid and len(cmds) == 1:
            print(
                f"{WARNING} {self.name} requires the '{cmd}' command, but it does not exist!"
            )
        return valid

    def _backup_cpuinfo(self):
        # Construct cpu_info dict from lscpu that is similar to what py-cpuinfo provides
        if not self.command_exists("lscpu"):
            self.warning(
                f"{self.name} attempted to query 'lscpu' after failing to use py-cpuinfo "
                "to detect the CPU architecture. 'lscpu' does not appear to exist on "
                "your system, will fall back to use -march=native and non-vectorized execution."
            )
            return None
        result = subprocess.check_output("lscpu", shell=True)
        result = result.decode("utf-8").strip().lower()

        cpu_info = {}
        cpu_info["arch"] = None
        cpu_info["flags"] = ""
        if "genuineintel" in result or "authenticamd" in result:
            cpu_info["arch"] = "X86_64"
            if "avx512" in result:
                cpu_info["flags"] += "avx512,"
            if "avx2" in result:
                cpu_info["flags"] += "avx2"
        elif "ppc64le" in result:
            cpu_info["arch"] = "PPC_"

    def cpu_arch(self):
        try:
            from cpuinfo import get_cpu_info
        except ImportError:
            cpu_info = self._backup_cpuinfo()
            if cpu_info is None:
                return "-march=native"

        try:
            cpu_info = get_cpu_info()
        except Exception as e:
            self.warning(
                f"{self.name} attempted to use `py-cpuinfo` but failed (exception type: {type(e)}, {e}), "
                "falling back to `lscpu` to get this information."
            )
            cpu_info = self._backup_cpuinfo()
            if cpu_info is None:
                return "-march=native"

        if cpu_info["arch"].startswith("PPC_"):
            # gcc does not provide -march on PowerPC, use -mcpu instead
            return "-mcpu=native"
        return "-march=native"

    def simd_width(self):
        try:
            from cpuinfo import get_cpu_info
        except ImportError:
            cpu_info = self._backup_cpuinfo()
            if cpu_info is None:
                return "-D__SCALAR__"

        try:
            cpu_info = get_cpu_info()
        except Exception as e:
            self.warning(
                f"{self.name} attempted to use `py-cpuinfo` but failed (exception type: {type(e)}, {e}), "
                "falling back to `lscpu` to get this information."
            )
            cpu_info = self._backup_cpuinfo()
            if cpu_info is None:
                return "-D__SCALAR__"

        if cpu_info["arch"] == "X86_64":
            if "avx512" in cpu_info["flags"]:
                return "-D__AVX512__"
            elif "avx2" in cpu_info["flags"]:
                return "-D__AVX256__"
        return "-D__SCALAR__"

    def libraries_args(self):
        if sys.platform == "win32":
            return ["cublas", "curand"]
        else:
            return []


class FusedLayerNormBinder(Binder):
    @property
    def name(self):
        return "oslo_fused_layer_norm"

    def sources(self):
        return [
            "fused_layer_norm.cu",
            "FusedLayerNormBinder.cpp",
        ]


class ExpertParallelBinder(Binder):
    @property
    def name(self):
        return "oslo_expert_parallel"

    def sources(self):
        return [
            "expert_parallel_cuda_kernel.cu",
            "ExpertParallelBinder.cpp",
        ]


class FusedAdamBinder(Binder):
    @property
    def name(self):
        return "oslo_adam"

    def sources(self):
        return [
            "multi_tensor_adam.cu",
            "FusedAdamBinder.cpp",
        ]


class FusedAdagradBinder(Binder):
    @property
    def name(self):
        return "oslo_adagrad"

    def sources(self):
        return [
            "multi_tensor_adagrad.cu",
            "FusedAdagradBinder.cpp",
        ]


class FusedNovogradBinder(Binder):
    @property
    def name(self):
        return "oslo_novograd"

    def sources(self):
        return [
            "multi_tensor_l2norm.cu",
            "multi_tensor_novograd.cu",
            "FusedNovogradBinder.cpp",
        ]


class FusedSGDBinder(Binder):
    @property
    def name(self):
        return "oslo_sgd"

    def sources(self):
        return [
            "multi_tensor_sgd.cu",
            "FusedSGDBinder.cpp",
        ]


class FusedLambBinder(Binder):
    @property
    def name(self):
        return "oslo_lamb"

    def sources(self):
        return ["multi_tensor_l2norm.cu", "multi_tensor_lamb.cu", "FusedLambBinder.cpp"]


class FusedMixedPrecisionLambBinder(Binder):
    @property
    def name(self):
        return "oslo_lamb_mp"

    def sources(self):
        return [
            "multi_tensor_l2norm_mp.cu",
            "multi_tensor_lamb_mp.cu",
            "FusedMixedPrecisionLambBinder.cpp",
        ]


class FusedL2NormBinder(Binder):
    @property
    def name(self):
        return "oslo_l2norm"

    def sources(self):
        return ["multi_tensor_l2norm.cu", "FusedL2NormBinder.cpp"]


class FusedMixedPrecisionL2NormBinder(Binder):
    @property
    def name(self):
        return "oslo_l2norm_mp"

    def sources(self):
        return ["multi_tensor_l2norm_mp.cu", "FusedMixedPrecisionL2NormBinder.cpp"]


class NgramRepeatBlockBinder(Binder):
    @property
    def name(self):
        return "oslo_ngram_repeat_block"

    def sources(self):
        return ["ngram_repeat_block_cuda_kernel.cu", "ngram_repeat_block_cuda.cpp"]


class CPUAdamBinder(CPUBinder):
    @property
    def name(self):
        return "oslo_cpu_adam"

    def sources(self):
        return [
            "custom_cuda_kernel.cu",
            "CPUAdamBinder.cpp",
        ]

    def libraries_args(self):
        args = super().libraries_args()
        if not self.is_rocm_pytorch():
            args += ["curand"]
        return args


class CPUAdagradBinder(CPUBinder):
    @property
    def name(self):
        return "oslo_cpu_adagrad"

    def sources(self):
        return [
            "custom_cuda_kernel.cu",
            "CPUAdagradBinder.cpp",
        ]
