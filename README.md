# OSLO: Open Source for Large-scale Optimization

![](https://raw.githubusercontent.com/EleutherAI/oslo/main/assets/logo.png)

## What is OSLO about?

OSLO is a framework that provides various GPU based optimization technologies for large-scale modeling.
Features like 3D parallelism and kernel fusion which could be useful when training a large model are the key features.
OSLO makes these technologies easy-to-use by magical compatibility with [Hugging Face Transformers](https://github.com/huggingface/transformers) that is being considered as a de facto standard in NLP field.
We look forward large-scale modeling technologies to be more democratized by significantly decreasing the difficulty of using these technologies using OSLO.

## Installation

OSLO can be easily installed using the pip package manager.
Be careful that the ‘core’ is in the PyPI project name.

```console
pip install oslo-core
```

## Administrative Notes

### Citing OSLO

If you find our work useful, please consider citing:

```
@misc{oslo,
  author       = {},
  title        = {OSLO: Open Source for Large-scale Optimization},
  howpublished = {\url{https://github.com/EleutherAI/oslo}},
  year         = {2022},
}
```

### Licensing

The code of the OSLO is licensed under the terms of the Apache License 2.0.

Copyright 2022 EleutherAI. All Rights Reserved.
