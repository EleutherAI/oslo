# Copyright 2021 TUNiB Inc.

import os
import sys
import platform
import subprocess
from setuptools import find_packages, setup

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

VERSION = {}  # type: ignore

with open("oslo/__version__.py", "r") as version_file:
    exec(version_file.read(), VERSION)

with open("requirements.txt", "r") as requirements_file:
    INSTALL_REQUIRES = requirements_file.read().splitlines()

with open("README.md", "r", encoding="utf-8") as README:
    long_description = README.read()

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
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: Linux",
        "Operating System :: Microsoft :: Windows",
    ],
    # ext_modules=ext_modules(),
    package_data={},
    dependency_links=[],
    include_package_data=True,
    zip_safe=False,
    license="Apache-2.0",
)
