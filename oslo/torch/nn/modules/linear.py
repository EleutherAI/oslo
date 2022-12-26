from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from oslo.torch.distributed import ParallelContext, ParallelMode


class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        skip_bias_add: bool = False,
        device: torch.device = None,
    ):
        super(Linear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.skip_bias_add = skip_bias_add

    def forward(self, input: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if not self.skip_bias_add:
            return F.linear(input, self.weight, self.bias)
        else:
            return F.linear(input, self.weight), self.bias


class ColLinear1D(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        skip_bias_add: bool = False,
        gather_output: bool = False,
        parallel_context: Optional[ParallelContext] = None,
    ):
        self.gather_output = gather_output
        self.parallel_context = parallel_context
        self.reversed = False

        self.world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)
        assert (
            out_features % self.world_size == 0
        ), "out_features must be divisible by world_size for ColLinear1D."

        super().__init__(
            in_features=in_features,
            out_features=out_features // self.world_size,
            skip_bias_add=skip_bias_add,
            bias=bias,
            dtype=dtype,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"gather_output={self.gather_output}"
        )

    def forward(self, input: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        from oslo.torch.nn.parallel.tensor_parallel._1d._ops import (
            gather_tensor_1d,
            broadcast_tensor_1d,
        )

        input = broadcast_tensor_1d(input, self.parallel_context)
        outputs = F.linear(input, self.weight)

        if self.bias is not None:
            if self.skip_bias_add:
                return outputs, self.bias
            else:
                outputs = outputs + self.bias

        if self.gather_output:
            outputs = gather_tensor_1d(
                outputs,
                dim=-1,
                parallel_context=self.parallel_context,
            )
            if hasattr(self, "orig_num_classes"):
                outputs = outputs[..., : self.orig_num_classes]
            if not outputs.is_contiguous():
                outputs = outputs.contiguous()

        return outputs


class RowLinear1D(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        skip_bias_add: bool = False,
        parallel_input: bool = True,
        parallel_context: Optional[ParallelContext] = None,
    ):
        self.parallel_input = parallel_input
        self.parallel_context = parallel_context
        self.reversed = False

        self.world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)
        assert (
            in_features % self.world_size == 0
        ), "in_features must be divisible by world_size for RowLinear1D."

        super().__init__(
            in_features=in_features // self.world_size,
            out_features=out_features,
            bias=bias,
            dtype=dtype,
            skip_bias_add=skip_bias_add,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"parallel_input={self.parallel_input}"
        )

    def forward(self, input: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        from oslo.torch.nn.parallel.tensor_parallel._1d._ops import (
            reduce_tensor_1d,
            scatter_tensor_1d,
        )

        if not self.parallel_input:
            input = scatter_tensor_1d(
                input,
                dim=-1,
                parallel_context=self.parallel_context,
            )
        outputs = F.linear(input, self.weight)
        outputs = reduce_tensor_1d(outputs, parallel_context=self.parallel_context)
        if self.bias is not None:
            if self.skip_bias_add:
                return outputs, self.bias
            else:
                return outputs + self.bias

        if not outputs.is_contiguous():
            outputs = outputs.contiguous()

        return outputs


class Linear2D(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        skip_bias_add: bool = False,
        gather_output: bool = False,
        parallel_context: Optional[ParallelContext] = None,
    ):
        self.gather_output = gather_output
        self.parallel_context = parallel_context
        self.reversed = False
        self.summa_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2D_COL
        )
        assert (
            in_features % self.summa_dim == 0
        ), "in_features must be divisible by summa_dim for Linear2D."
        assert (
            out_features % (self.summa_dim**2) == 0
        ), "out_features must be divisible by summa_dim^2 for Linear2D."

        super().__init__(
            in_features=in_features // self.summa_dim,
            out_features=out_features // self.summa_dim,
            bias=False,
            dtype=dtype,
            skip_bias_add=skip_bias_add,
        )
        if bias:
            self.bias = Parameter(
                torch.empty(
                    out_features // (self.summa_dim**2),
                    device=self.weight.device,
                    dtype=dtype,
                )
            )
            self.reset_parameters()

        self.row_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2D_ROW)
        self.col_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
        self.data_parallel_rank = self.parallel_context.get_local_rank(
            ParallelMode.DATA
        )
        self.pipeline_parallel_rank = self.parallel_context.get_local_rank(
            ParallelMode.PIPELINE
        )

        self.tensor_parallel_size = self.parallel_context.get_world_size(
            ParallelMode.TENSOR
        )
        self.pipeline_parallel_size = self.parallel_context.get_world_size(
            ParallelMode.PIPELINE
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, gather_output={self.gather_output}"
        )

    def forward(self, input: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        from oslo.torch.nn.parallel.tensor_parallel._2d._ops import (
            Matmul_ABT_2D,
            add_bias_2d,
            all_gather_tensor_2d,
        )

        # input: [m/q, n/q, k/q]
        # output: [m/q, n/q, h/q]
        out_shape = input.shape[:-1] + (self.out_features,)
        outputs = Matmul_ABT_2D.apply(
            input,
            self.weight,
            self.summa_dim,
            out_shape,
            self.row_rank,
            self.col_rank,
            self.data_parallel_rank,
            self.pipeline_parallel_rank,
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
            self.parallel_context,
            ParallelMode.TENSOR_2D_ROW,
            ParallelMode.TENSOR_2D_COL,
        )

        if self.bias is not None:
            if self.skip_bias_add:
                bias = add_bias_2d(
                    None,
                    self.bias,
                    self.out_features,
                    self.row_rank,
                    self.col_rank,
                    True,
                    self.data_parallel_rank,
                    self.pipeline_parallel_rank,
                    self.pipeline_parallel_size,
                    self.tensor_parallel_size,
                    self.parallel_context,
                    ParallelMode.TENSOR_2D_ROW,
                    ParallelMode.TENSOR_2D_COL,
                )
                return outputs, bias
            else:
                outputs = add_bias_2d(
                    outputs,
                    self.bias,
                    self.out_features,
                    self.row_rank,
                    self.col_rank,
                    False,
                    self.data_parallel_rank,
                    self.pipeline_parallel_rank,
                    self.pipeline_parallel_size,
                    self.tensor_parallel_size,
                    self.parallel_context,
                    ParallelMode.TENSOR_2D_ROW,
                    ParallelMode.TENSOR_2D_COL,
                )
        if self.gather_output:
            outputs = all_gather_tensor_2d(
                outputs,
                dim=0,
                parallel_context=self.parallel_context,
                parallel_mode=ParallelMode.TENSOR_2D_COL,
            )
            outputs = all_gather_tensor_2d(
                outputs,
                dim=-1,
                parallel_context=self.parallel_context,
                parallel_mode=ParallelMode.TENSOR_2D_ROW,
            )
            if hasattr(self, "orig_num_classes"):
                outputs = outputs[..., : self.orig_num_classes]

            if not outputs.is_contiguous():
                outputs = outputs.contiguous()

        return outputs


class Linear2p5D(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        skip_bias_add: bool = False,
        gather_output: bool = False,
        parallel_context: Optional[ParallelContext] = None,
    ):
        self.gather_output = gather_output
        self.parallel_context = parallel_context
        self.reversed = False
        self.tesseract_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2P5D_COL
        )
        assert (
            in_features % self.tesseract_dim == 0
        ), "in_features must be divisible by tesseract_dim for Linear2p5D."
        assert (
            out_features % self.tesseract_dim == 0
        ), "out_features must be divisible by tesseract_dim for Linear2p5D."

        super().__init__(
            in_features=in_features // self.tesseract_dim,
            out_features=out_features // self.tesseract_dim,
            bias=bias,
            dtype=dtype,
            skip_bias_add=skip_bias_add,
        )

        self.row_rank = self.parallel_context.get_local_rank(
            ParallelMode.TENSOR_2P5D_ROW
        )
        self.col_rank = self.parallel_context.get_local_rank(
            ParallelMode.TENSOR_2P5D_COL
        )
        self.dep_rank = self.parallel_context.get_local_rank(
            ParallelMode.TENSOR_2P5D_DEP
        )
        self.data_parallel_rank = self.parallel_context.get_local_rank(
            ParallelMode.DATA
        )
        self.pipeline_parallel_rank = self.parallel_context.get_local_rank(
            ParallelMode.PIPELINE
        )

        self.tensor_parallel_size = self.parallel_context.get_world_size(
            ParallelMode.TENSOR
        )
        self.pipeline_parallel_size = self.parallel_context.get_world_size(
            ParallelMode.PIPELINE
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, gather_output={self.gather_output}"
        )

    def forward(self, input: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        from oslo.torch.nn.parallel.tensor_parallel._2p5d._ops import (
            Matmul_ABT_2p5D,
            add_bias_2p5d,
            all_gather_tensor_2p5d,
        )

        out_shape = input.shape[:-1] + (self.out_features,)

        outputs = Matmul_ABT_2p5D.apply(
            input,
            self.weight,
            self.tesseract_dim,
            out_shape,
            self.row_rank,
            self.col_rank,
            self.dep_rank,
            self.data_parallel_rank,
            self.pipeline_parallel_rank,
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
            self.parallel_context,
            ParallelMode.TENSOR_2P5D_ROW,
            ParallelMode.TENSOR_2P5D_COL,
        )

        if self.bias is not None:
            if self.skip_bias_add:
                bias = add_bias_2p5d(
                    None,
                    self.bias,
                    self.out_features,
                    self.tesseract_dim,
                    self.row_rank,
                    self.col_rank,
                    self.dep_rank,
                    True,
                    self.data_parallel_rank,
                    self.pipeline_parallel_rank,
                    self.pipeline_parallel_size,
                    self.tensor_parallel_size,
                    self.parallel_context,
                    ParallelMode.TENSOR_2P5D_COL,
                )
                return outputs, bias
            else:
                outputs = add_bias_2p5d(
                    outputs,
                    self.bias,
                    self.out_features,
                    self.tesseract_dim,
                    self.row_rank,
                    self.col_rank,
                    self.dep_rank,
                    False,
                    self.data_parallel_rank,
                    self.pipeline_parallel_rank,
                    self.pipeline_parallel_size,
                    self.tensor_parallel_size,
                    self.parallel_context,
                    ParallelMode.TENSOR_2P5D_COL,
                )
        if self.gather_output:
            outputs = all_gather_tensor_2p5d(
                outputs,
                dim=-1,
                parallel_context=self.parallel_context,
                col_parallel_mode=ParallelMode.TENSOR_2P5D_ROW,
            ).clone()
            outputs = all_gather_tensor_2p5d(
                outputs,
                dim=0,
                parallel_context=self.parallel_context,
                col_parallel_mode=ParallelMode.TENSOR_2P5D_COL,
            ).clone()
            if hasattr(self, "orig_num_classes"):
                outputs = outputs[..., : self.orig_num_classes]

            if not outputs.is_contiguous():
                outputs = outputs.contiguous()

        return outputs


class Linear3D(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        skip_bias_add: bool = False,
        gather_output: bool = False,
        parallel_context: Optional[ParallelContext] = None,
    ):
        self.gather_output = gather_output
        self.parallel_context = parallel_context
        self.reversed = False
        self.cubic_dim = parallel_context.get_world_size(ParallelMode.TENSOR_3D_INPUT)

        assert (
            in_features % self.cubic_dim == 0
        ), "in_features must be divisible by cubic_dim for Linear3D."
        assert (
            out_features % (self.cubic_dim**2) == 0
        ), "out_features must be divisible by cubic_dim^2 for Linear3D."

        super().__init__(
            in_features=in_features // self.cubic_dim,
            out_features=out_features // (self.cubic_dim**2),
            bias=False,
            dtype=dtype,
            skip_bias_add=skip_bias_add,
        )
        if bias:
            self.bias = Parameter(
                torch.empty(
                    out_features // self.cubic_dim,
                    device=self.weight.device,
                    dtype=dtype,
                )
            )
            self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._3d._ops import (
            Matmul_ABT_3D,
            all_gather_tensor_3d,
        )

        outputs = Matmul_ABT_3D.apply(
            input,
            self.weight,
            self.bias,
            0,
            0,
            0,
            self.parallel_context,
            ParallelMode.TENSOR_3D_INPUT,
            ParallelMode.TENSOR_3D_WEIGHT,
            ParallelMode.TENSOR_3D_OUTPUT,
        )
        if self.gather_output:
            outputs = all_gather_tensor_3d(
                outputs,
                dim=-1,
                parallel_context=self.parallel_context,
                parallel_mode=ParallelMode.TENSOR_3D_OUTPUT,
            )
            outputs = all_gather_tensor_3d(
                outputs,
                dim=0,
                parallel_context=self.parallel_context,
                parallel_mode=ParallelMode.TENSOR_3D_INPUT,
            )
            outputs = all_gather_tensor_3d(
                outputs,
                dim=0,
                parallel_context=self.parallel_context,
                parallel_mode=ParallelMode.TENSOR_3D_WEIGHT,
            )
            if hasattr(self, "orig_num_classes"):
                outputs = outputs[..., : self.orig_num_classes]

            if not outputs.is_contiguous():
                outputs = outputs.contiguous()
        return outputs
