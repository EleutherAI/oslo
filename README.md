# OSLO

OSLO Python library stands for *Open-Source Framework for Large-scale Model Optimization.* This framework that provides various GPU-based optimization layers for large-scale modeling. Data Parallelism and Kernel Fusion which could be useful when training a large model like EleutherAI/gpt-j-6B are the key features. OSLO makes these technologies easy-to-use by magical compatibility with Hugging Face Transformers.


## Build Requirements

Requirements for building OSLO without any issues:

- Linux Driver (450.80.02 or later) Windows Driver (456.38 or later)
- CUDA Toolkit 11.0 to 11.7
- PyTorch >= 1.11.0
- Turing or Ampere GPU

## Compiling

To compile the oslo build, run:

```
python3 setup.py install
```
