<div align="center">

![](assets/oslo.png)

<br>

## O S L O

**O**pen **S**ource framework for **L**arge-scale transformer **O**ptimization

<p align="center">
<a href="https://github.com/tunib-ai/oslo/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/tunib-ai/oslo.svg" /></a> 
<a href="https://github.com/tunib-ai/oslo/blob/master/LICENSE.apache-2.0"><img alt="Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"/></a> <a href="https://github.com/tunib-ai/oslo/blob/master/USAGE.md"><img alt="Docs" src="https://img.shields.io/badge/docs-passing-success.svg"/></a>
<a href="https://github.com/tunib-ai/oslo/issues"><img alt="Issues" src="https://img.shields.io/github/issues/tunib-ai/oslo"/></a>


</div>

<br><br>

### What's New:
* December 30, 2021 [Add Deployment Launcher](https://github.com/tunib-ai/oslo/releases/tag/v1.0).
* December 21, 2021 [Released OSLO 1.0](https://github.com/tunib-ai/oslo/releases/tag/v1.0).

## What is OSLO about?
OSLO is a framework that provides various GPU based optimization technologies for large-scale modeling. 3D Parallelism and Kernel Fusion which could be useful when training a large model like [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B) are the key features. OSLO makes these technologies easy-to-use by magical compatibility with [Hugging Face Transformers](https://github.com/huggingface/transformers) that is being considered as a <i>de facto</i> standard in 2021. Currently, the architectures such as GPT2, GPTNeo, and GPTJ are supported, but we plan to support more soon.

## Installation
OSLO can be easily installed using the pip package manager.
All the dependencies such as [torch](https://pypi.org/project/torch/), [transformers](https://pypi.org/project/transformers/), [dacite](https://pypi.org/project/dacite/),
[ninja](https://pypi.org/project/ninja/) and [pybind11](https://pypi.org/project/pybind11/) should be installed automatically with the following command.
Be careful that the 'core' is in the PyPI project name.
```console
pip install oslo-core
```

Some of features rely on the C++ language.
So we provide an option, `CPP_AVAILABLE`, to decide whether or not you install them.

- If the C++ is available:
```console
CPP_AVAILABLE=1 pip install oslo-core
```

- If the C++ is not available:
```console
CPP_AVAILABLE=0 pip install oslo-core
```

Note that the default value of `CPP_AVAILABLE` is 0 in Windows and 1 in Linux.

## Key Features

```python
import deepspeed
from oslo import GPTJForCausalLM

# 1. 3D Parallelism
model = GPTJForCausalLM.from_pretrained_with_parallel(
    "EleutherAI/gpt-j-6B", tensor_parallel_size=2, pipeline_parallel_size=2,
)

# 2. Kernel Fusion
model = model.fuse()

# 3. DeepSpeed Support
engines = deepspeed.initialize(
    model=model.gpu_modules(), model_parameters=model.gpu_parameters(), ...,
)

# 4. Data Processing
from oslo import (
    DatasetPreprocessor,
    DatasetBlender,
    DatasetForCausalLM,
    ...
)

# 5. Deployment Launcher
model = GPTJForCausalLM.from_pretrained_with_parallel(..., deployment=True)
```

OSLO offers the following features.

- **3D Parallelism**: The state-of-the-art technique for training a large-scale model with multiple GPUs.
- **Kernel Fusion**: A GPU optimization method to increase training and inference speed.
- **DeepSpeed Support**: We support [DeepSpeed](https://github.com/microsoft/DeepSpeed) which provides ZeRO data parallelism.
- **Data Processing**: Various utilities for efficient large-scale data processing.
- **Deployment Launcher**: A launcher for easily deploying a parallelized model to the web server.

See [USAGE.md](USAGE.md) to learn how to use them.

## Administrative Notes

### Citing OSLO
If you find our work useful, please consider citing:

```
@misc{oslo,
  author       = {Ko, Hyunwoong and Kim, Soohwan and Park, Kyubyong},
  title        = {OSLO: Open Source framework for Large-scale transformer Optimization},
  howpublished = {\url{https://github.com/tunib-ai/oslo}},
  year         = {2021},
}
```

### Licensing

The Code of the OSLO project is licensed under the terms of the [Apache License 2.0](LICENSE.apache-2.0).

Copyright 2021 TUNiB Inc. http://www.tunib.ai All Rights Reserved.

### Acknowledgements

The OSLO project is built with GPU support from the [AICA (Artificial Intelligence Industry Cluster Agency)](http://www.aica-gj.kr).
