from typing import Optional

import torch

from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.nn.parallel.data_parallel.zero.heterogeneous_manager.chunk import (
    init_chunk_manager,
)
from oslo.torch.nn.parallel.data_parallel.zero.heterogeneous_manager.manager import (
    HeterogeneousMemoryManager,
)
from oslo.torch.nn.parallel.data_parallel.zero.heterogeneous_manager.memory_tracer import (
    MemStats,
)

from oslo.torch.nn.parallel.data_parallel.fully_sharded_data_parallel import (
    _FullyShardedDataParallel,
)


class _HeterogeneousDataParallel(_FullyShardedDataParallel):
    def __init__(
        self,
        module: torch.nn.Module,
        device: torch.device,
        parallel_context: ParallelContext = None,
        placement_policy: str = "cpu",
        pin_memory: bool = False,
        force_outputs_fp32: bool = False,
        strict_ddp_mode: bool = False,
        search_range_mb: int = 32,
        hidden_dim: Optional[int] = None,
        min_chunk_size_mb: float = 32,
        memstats: Optional[MemStats] = None,
    ) -> None:
        """
        A torch.nn.Module wrapper using ZeRO-DP and HeterogeneousMemoryManager.
        WARNING: The class will modify the module inline!
        Example:
            model is initialized under the context of oslo.ready
            >>> model = _HeterogeneousDataParallel(model, torch.cuda.current_device(), "cuda")
            >>> logits = model(x)
            >>> loss = criterion(logits, labels)
            >>> loss.backward()
        Args:
            module (torch.nn.Module): the model to be wrapped.
            device (torch.device): device to place the model.
            parallel_context (ParallelContext): process group object.
            placement_policy (str, optional): "cpu", "cuda", "auto". Defaults to "cpu".
            pin_memory (bool, optional): use pin memory on CPU. Defaults to False.
            force_outputs_fp32 (bool, optional): force outputs are fp32. Defaults to False.
            search_range_mb (int, optional): chunk size searching range in MegaByte. Defaults to 32.
            hidden_dim (int, optional): the hidden dimension of DNN.
                Users can provide this argument to speed up searching.
                If users do not know this argument before training, it is ok. We will use a default value 1024.
            min_chunk_size_mb (float, optional): the minimum chunk size in MegaByte.
                If the aggregate size of parameters is still smaller than the minimum chunk size,
                all parameters will be compacted into one small chunk.
            memstats (MemStats, optional) the memory statistics collector by a runtime memory tracer.
        """
        # some ugly hotfix for the compatibility with Lightning
        if search_range_mb is None:
            search_range_mb = 32

        chunk_manager = init_chunk_manager(
            model=module,
            init_device=device,
            hidden_dim=hidden_dim,
            search_range_mb=search_range_mb,
            min_chunk_size_mb=min_chunk_size_mb,
            strict_ddp_flag=strict_ddp_mode,
        )
        heterogeneous_manager = HeterogeneousMemoryManager(
            placement_policy, chunk_manager, memstats
        )
        super().__init__(
            module,
            heterogeneous_manager,
            parallel_context,
            pin_memory,
            force_outputs_fp32,
            strict_ddp_mode,
        )
