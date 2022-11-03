# OSLO

OSLO Python library stands for *Open-Source Framework for Large-scale Model Optimization.* This framework that provides various GPU-based optimization technologies for large-scale modeling. 3D Parallelism and Kernel Fusion which could be useful when training a large model like EleutherAI/gpt-j-6B are the key features. OSLO makes these technologies easy-to-use by magical compatibility with Hugging Face Transformers.


## Build Requirements

Requirements for building OSLO without any issues:

- CUDA 11 & nvcc
- PyTorch >= 1.11.0
- Turing or Ampere GPU

```
python3 setup.py install
```
