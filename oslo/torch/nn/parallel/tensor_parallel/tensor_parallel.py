import copy
from typing import Optional

import torch
import torch.nn as nn

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.tensor_parallel._1d._wrapper import (
    _TensorParallel1D,
)
from oslo.torch.nn.parallel.tensor_parallel._2d._wrapper import (
    _TensorParallel2D,
)
from oslo.torch.nn.parallel.tensor_parallel._2p5d._wrapper import (
    _TensorParallel2p5D,
)
from oslo.torch.nn.parallel.tensor_parallel._3d._wrapper import (
    _TensorParallel3D,
)
from oslo.torch.nn.parallel.tensor_parallel.mapping import (
    TensorParallelMapping,
)
from oslo.torch.nn.parallel.utils import (
    get_parallel_context,
    add_wrapper,
    OsloParallelWrapper,
)
from oslo.transformers.mapping_utils import (
    _TensorParallelMapping,
)


def get_divisible_by(parallel_context: ParallelContext):
    if parallel_context.tensor_parallel_mode == ParallelMode.TENSOR_1D:
        divisible_by = parallel_context.get_world_size(ParallelMode.TENSOR_1D)
    elif parallel_context.tensor_parallel_mode == ParallelMode.TENSOR_2D:
        divisible_by = parallel_context.get_world_size(ParallelMode.TENSOR_2D_COL) ** 2
    elif parallel_context.tensor_parallel_mode == ParallelMode.TENSOR_2P5D:
        divisible_by = parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)
    elif parallel_context.tensor_parallel_mode == ParallelMode.TENSOR_3D:
        divisible_by = (
            parallel_context.get_world_size(ParallelMode.TENSOR_3D_INPUT) ** 2
        )
    else:
        raise ValueError(
            "currently, only 1D, 2D, 2.5D, 3D tensor parallelism is supported."
        )
    return divisible_by


def TensorParallel(
    module: nn.Module,
    parallel_context: Optional[ParallelContext] = None,
):
    tp = _TensorParallel(
        module=module,
        parallel_context=parallel_context,
    )
    add_wrapper(
        module, mode=ParallelMode.TENSOR, wrapper=tp, parallel_context=parallel_context
    )
    return module


class _TensorParallel(OsloParallelWrapper):
    """
    Tensor parallel module

    Args:
        module (nn.Module): PyTorch module object
        parallel_context (ParallelContext): process context

    Notes:
        1. Similar design with `torch.nn.parallel.DistributedDataParallel`.
        2. Support auto de-parallelism

    Examples:
        >>> from oslo.torch.nn.parallel import TensorParallel

        >>> model = TransformersModel()
        >>> optimizer = AnyOptimizer(model.parameters(), lr=3e-5)
        >>> tp_wrapper = TensorParallel(model, parallel_context=..., ...)

        >>> output = tp_wrapper(input_data)
        >>> output.backward()
        >>> optimizer.step()
    """

    def __init__(self, module: nn.Module, parallel_context: ParallelContext):
        super().__init__(parallelism_priority=0)
        self.module = module
        self.parallel_context = get_parallel_context(module, parallel_context)

    def forward(self, *args, **kwargs):
        return self.module_forward(*args, **kwargs)

    @staticmethod
    def _resize_vocab_size(model, parallel_context):
        assert hasattr(
            model, "get_input_embeddings"
        ), "model object must have `get_input_embeddings` method."

        module = model.get_input_embeddings()
        if not hasattr(module, "weight"):
            module.weight = None
            return model

        vocab_size, embedding_dim = module.weight.size()
        new_vocab_size = vocab_size

        divisible_by = get_divisible_by(parallel_context)
        while new_vocab_size % divisible_by != 0:
            new_vocab_size += 1

        if new_vocab_size != vocab_size:
            padding = torch.zeros(
                new_vocab_size - vocab_size,
                embedding_dim,
                dtype=module.weight.dtype,
                device=module.weight.device,
            )
            new_embeddings = torch.cat(
                tensors=[module.weight.data, padding],
                dim=0,
            )

            module.weight.data = new_embeddings
            module.num_embeddings = new_vocab_size
        setattr(module, "orig_num_classes", vocab_size)
        setattr(model, "orig_vocab_size", vocab_size)
        return model

    @staticmethod
    def _resize_num_classes(model, parallel_context):
        mapping = _TensorParallelMapping().get_mapping(model)
        tensor_parallel_mapping = TensorParallelMapping(mapping)
        divisible_by = get_divisible_by(parallel_context)

        for param_name, module in model.named_modules():
            if tensor_parallel_mapping.is_head(model, param_name) and isinstance(
                module, nn.Linear
            ):
                if module.weight is model.get_input_embeddings().weight:
                    module.out_features = model.get_input_embeddings().num_embeddings

                    assert hasattr(
                        model.get_input_embeddings(), "orig_num_classes"
                    ), "call _resize_vocab before _resize_num_classes"
                    out_features = model.get_input_embeddings().orig_num_classes
                    setattr(module, "orig_num_classes", out_features)
                    setattr(
                        model,
                        f"orig_{param_name.split('.')[-1]}_num_classes",
                        out_features,
                    )

                    if hasattr(module, "bias") and module.bias is not None:
                        out_features = module.bias.size()[0]
                        new_out_features = out_features

                        while new_out_features % divisible_by != 0:
                            new_out_features += 1

                        if new_out_features != out_features:
                            padding = torch.zeros(
                                new_out_features - out_features,
                                dtype=module.bias.dtype,
                                device=module.bias.device,
                            )
                            new_bias = torch.cat(
                                tensors=[module.bias.data, padding],
                                dim=0,
                            )
                            module.bias.data = new_bias
                else:
                    out_features, in_features = module.weight.size()
                    new_out_features = out_features

                    while new_out_features % divisible_by != 0:
                        new_out_features += 1

                    if new_out_features != out_features:
                        padding = torch.zeros(
                            new_out_features - out_features,
                            in_features,
                            dtype=module.weight.dtype,
                            device=module.weight.device,
                        )
                        new_weight = torch.cat(
                            tensors=[module.weight.data, padding],
                            dim=0,
                        )

                        if hasattr(module, "bias") and module.bias is not None:
                            padding = torch.zeros(
                                new_out_features - out_features,
                                dtype=module.weight.dtype,
                                device=module.weight.device,
                            )
                            new_bias = torch.cat(
                                tensors=[module.bias.data, padding],
                                dim=0,
                            )
                            module.bias.data = new_bias

                        module.weight.data = new_weight
                        module.out_features = new_out_features
                        setattr(module, "orig_num_classes", out_features)
                        setattr(
                            model,
                            f"orig_{param_name.split('.')[-1]}_num_classes",
                            out_features,
                        )
        return model

    def deparallelize(self):
        self.module.deparallelize()

    def parallelize(self):
        self.module = self._resize_vocab_size(self.module, self.parallel_context)
        self.module = self._resize_num_classes(self.module, self.parallel_context)

        if self.parallel_context.tensor_parallel_mode == ParallelMode.TENSOR_1D:
            self.module = _TensorParallel1D(self.module, self.parallel_context)
        elif self.parallel_context.tensor_parallel_mode == ParallelMode.TENSOR_2D:
            self.module = _TensorParallel2D(self.module, self.parallel_context)
        elif self.parallel_context.tensor_parallel_mode == ParallelMode.TENSOR_2P5D:
            self.module = _TensorParallel2p5D(self.module, self.parallel_context)
        elif self.parallel_context.tensor_parallel_mode == ParallelMode.TENSOR_3D:
            self.module = _TensorParallel3D(self.module, self.parallel_context)
        else:
            raise ValueError(
                "currently, only 1d, 2d, 2p5d, 3d tensor parallelism are supported."
            )
        self.module_forward = copy.copy(self.module.forward)
        self.module.parallelize()
