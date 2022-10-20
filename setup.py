"""
Copyright 2021 TUNiB Inc.

To build oslo on Windows:
    1. Install Pytorch, such as PyTorch 1.12.1 + cu116 (recommended)
    2. Install Visual C++ Build Tools
    3. Launch terminal with Administrator privilege for creating required symlink folders

oslo dist will be located in: dist/*.egg
oslo copied build files will be located in: build/lib/oslo

"""

import os
import sys
import time
import platform
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME, BuildExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# CUDA OS environment configuration check

if not CUDA_HOME:
    CUDA_HOME = os.environ.get("CUDA_PATH")
if not CUDA_HOME:
    raise RuntimeError("ENV variable for CUDA_HOME or CUDA_PATH is not set")

# ----------------------------------------------------------------------
# Python & PyTorch configuration check

python_min_version = (3, 6, 0)
python_min_version_str = ".".join(map(str, python_min_version))
pytorch_min_version = (1, 11, 0)
pytorch_min_version_str = ".".join(map(str, pytorch_min_version))

if sys.version_info < python_min_version:
    print(
        "You are using Python {}. Python >={} is required.".format(
            platform.python_version(), python_min_version_str
        )
    )
    sys.exit(-1)

if sys.version_info < (3,):
    print(
        "You are using Python {}. Python >={} is required.".format(
            platform.python_version(), python_min_version_str
        )
    )
    sys.exit(-1)

if sys.platform == "win32" and sys.maxsize.bit_length() == 31:
    print(
        "32-bit Windows Python runtime is not supported. Please switch to 64-bit Python."
    )
    sys.exit(-1)


try:
    import torch
except ModuleNotFoundError as err:
    print(
        "PyTorch could not import. Please install torch >= 1.11.0 to use OSLO.\nBuild exited with error: {}".format(
            err
        )
    )
    sys.exit(-1)

if not torch.__version__ >= pytorch_min_version_str:
    print(
        "OSLO requires PyTorch >= 1.11.0.\n"
        + "You are using torch.__version__ = {}.\nRequired version for torch.__version__ is >= {}. Please upgrade your PyTorch version.".format(
            torch.__version__, pytorch_min_version_str
        )
    )
    sys.exit(-1)

TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])
if TORCH_MAJOR < 1 or (TORCH_MAJOR == 1 and TORCH_MINOR < 10):
    raise RuntimeError(
        "OSLO requires PyTorch >= 1.11.0.\n"
        + "The required release can be obtained from https://pytorch.org/"
    )

if not torch.cuda.is_available():
    print(
        "\nWarning: Torch did not find available GPUs on this system.",
        "If your intention is to cross-compile, this is not an error.\n"
        "By default, OSLO will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n"
        "Volta (compute capability 7.0), Turing (compute capability 7.5),\n"
        "and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n"
        "If you wish to cross-compile for a single specific architecture,\n"
        'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n',
    )

# ----------------------------------------------------------------------
# CUDA & nvcc compile configurations check


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(
        cuda_dir
    )
    torch_binary_major = torch.version.cuda.split(".")[0]
    torch_binary_minor = torch.version.cuda.split(".")[1]

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_major != torch_binary_major) or (
        bare_metal_minor != torch_binary_minor
    ):
        raise RuntimeError(
            "CUDA extensions are being compiled with a version of CUDA that does "
            + "not match the version used to compile Pytorch binaries.  "
            + "Pytorch binaries were compiled with Cuda {}.\n".format(
                torch.version.cuda
            )
            + "In some cases, a minor-version mismatch will not cause later errors:  "
            + "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
            "You can try commenting out this check (at your own risk)."
        )


def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
    if int(bare_metal_major) >= 11 and int(bare_metal_minor) >= 2:
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args


cmdclass = {}
ext_modules = []
version_dependent_macros = ["-DVERSION_GE_1_1", "-DVERSION_GE_1_3", "-DVERSION_GE_1_5"]

if CUDA_HOME is None:
    print(
        "Are you sure your environment has nvcc available?\n",
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch,"
        "only images whose names contain 'devel' will provide nvcc.",
    )
else:

    def cuda_ext_helper(name, sources, extra_cuda_flags):
        return CUDAExtension(
            name=name,
            sources=[os.path.join("oslo/torch/_C/csrc", path) for path in sources],
            include_dirs=[os.path.join(this_dir, "oslo/torch/_C/csrc/includes")],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": append_nvcc_threads(
                    ["-O3", "--use_fast_math"]
                    + version_dependent_macros
                    + extra_cuda_flags
                ),
            },
        )

    cc_flag = ["-gencode", "arch=compute_70,code=sm_70"]
    _, bare_metal_major, _ = get_cuda_bare_metal_version(CUDA_HOME)
    if int(bare_metal_major) >= 11:
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")

    extra_cuda_flags = [
        "-std=c++14",
        "-maxrregcount=50",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]

    # TODO: add ext_modules

cmdclass = {}
ext_modules = []
VERSION = {}  # type: ignore

with open("oslo/__version__.py", "r") as version_file:
    exec(version_file.read(), VERSION)

with open("requirements.txt", "r") as requirements_file:
    INSTALL_REQUIRES = requirements_file.read().splitlines()

with open("README.md", "r", encoding="utf-8") as README:
    long_description = README.read()

start_time = time.time()

setup(
    name="oslo-core",
    version=VERSION["version"],
    description="OSLO: Open Source framework for Large-scale transformer Optimization",
    long_description_content_type="text/markdown",
    url="https://github.com/eleutherai/oslo",
    author="TUNiB OSLO Team",
    author_email="contact@tunib.ai",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(
        include=["oslo", "oslo.*"],
        exclude=("tests", "tutorial", "docs"),
    ),
    python_requires=">={}".format(python_min_version_str),
    # ext_modules=ext_modules,
    # cmdclass={'build_ext': BuildExtension} if ext_modules else {},
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: Linux",
        "Operating System :: Microsoft :: Windows",
    ],
    package_data={},
    dependency_links=[],
    include_package_data=True,
    zip_safe=False,
    license="Apache-2.0",
)

end_time = time.time()
print(f"oslo build time finished at = {end_time - start_time} seconds")
