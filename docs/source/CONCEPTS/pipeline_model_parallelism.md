# Concept of Pipeline Model Parallelism
- Authors: Kichang Yang, Kevin Ko

In this session, we will learn about pipeline parallelism.

## 1. Inter-layer model parallelism
Pipeline parallelism improves inter-layer model parallelism, a model parallelization method that assigns certain layers to specific GPUs, as shown in the figure below. In the figure, layers 1, 2, and 3 are assigned to GPU 1, and layers 4 and 5 are assigned to GPU 2. The divided piece is called a 'stage.' In the example below, it is divided into two stages.

![](../images/inter_layer.png)

However, due to the nature of neural networks that use the output of the previous layer as the input of the next layer, another GPU cannot start operation until the operation on a specific GPU is finished. In other words, inter-layer model parallelism has a fatal limitation that only one GPU can be used at a time, as shown in the figures below.

![](../images/inter_layer_2.png)
![](../images/inter_layer_3.gif)

## 2. GPipe
GPipe is a pipeline parallelism technique developed by Google to reduce the idle time of GPUs during inter-layer model parallelism and operates by dividing mini-batches into micro-batches and pipeline learning.

![](../images/gpipe_1.png)

![](../images/pipeline_parallelism2.png)

### Micro-batch
- A mini-batch is a sub-sample set obtained by dividing the entire dataset into n.
- A micro-batch is a sub-sample set obtained by dividing a mini-batch into m.

![](../images/gpipe_2.png)

### Pipelining
GPipe divides mini-batches into micro-batches and pipelines the operations. The red part (idle time of the GPU) is called Bubble time. As the micro-batch size increases, the bubble time decreases.

![](../images/gpipe_3.gif)

### GPipe with PyTorch
You can easily use GPipe with `torchgpipe`, which was released by kakaobrain. However, only models wrapped in `nn.Sequential` can be used, and the input and output types of all modules are limited to `torch.Tensor` or `Tuple[torch.Tensor]`. Therefore, coding can be quite challenging.
"""



