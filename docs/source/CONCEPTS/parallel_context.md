# Concept of Parallel Context
Parallel Context is distributed process group manager in OSLO. 
You can easily create and manage distributed process groups by putting the desired parallelization sizes into Parallel Context class.

## 1. Create Parallel Context object
There are three methods to create Parallel Context: `from_torch`, `from_slurm`, `from_openmpi`.

### 1.1. `from_torch`
If you are using PyTorch distributed launcher, you can use `from_torch` function to create it.

```python
from oslo.torch import ParallelContext

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    sequence_parallel_size=1,
    expert_parallel_size=1,
)
```

### 1.2. `from_slurm`
If you are using Slurm launcher, you can use `from_slurm` function to create it.
In this case, you must input `host` and `port` together.

```python
from oslo.torch import ParallelContext

YOUR_HOST = ...
YOUR_PORT = ...

parallel_context = ParallelContext.from_slurm(
    host=YOUR_HOST,
    port=YOUR_PORT,
    data_parallel_size=1,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    sequence_parallel_size=1,
    expert_parallel_size=1,
)
```

### 1.3. `from_openmpi`
If you are using OpenMPI launcher, you can use `from_openmpi` function to create it.
Similar with `from_slurm`, you must input `host` and `port` together.

```python
from oslo.torch import ParallelContext

YOUR_HOST = ...
YOUR_PORT = ...

parallel_context = ParallelContext.from_openmpi(
    host=YOUR_HOST,
    port=YOUR_PORT,
    data_parallel_size=1,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    sequence_parallel_size=1,
    expert_parallel_size=1,
)
```

## 2. Check device ranks and world sizes easily
There is an enum class named `ParallelMode`, you can easily check device ranks and world sizes easily with it.

### 2.1. Ranks

```python
from oslo.torch import ParallelMode

# create parallel context object
parallel_context = ...

global_rank = parallel_context.get_local_rank(ParallelMode.GLOBAL)
data_parallel_rank = parallel_context.get_local_rank(ParallelMode.DATA)
tensor_parallel_rank = parallel_context.get_local_rank(ParallelMode.TENSOR)
pipeline_parallel_rank = parallel_context.get_local_rank(ParallelMode.PIPELINE)
sequence_parallel_rank = parallel_context.get_local_rank(ParallelMode.SEQUENCE)
expoert_parallel_rank = parallel_context.get_local_rank(ParallelMode.EXPERT)
```

### 2.2. World sizes

```python
from oslo.torch import ParallelMode

# create parallel context object
parallel_context = ...

global_size = parallel_context.get_world_size(ParallelMode.GLOBAL)
data_parallel_size = parallel_context.get_world_size(ParallelMode.DATA)
tensor_parallel_size = parallel_context.get_world_size(ParallelMode.TENSOR)
pipeline_parallel_size = parallel_context.get_world_size(ParallelMode.PIPELINE)
sequence_parallel_size = parallel_context.get_world_size(ParallelMode.SEQUENCE)
expoert_parallel_size = parallel_context.get_world_size(ParallelMode.EXPERT)
```