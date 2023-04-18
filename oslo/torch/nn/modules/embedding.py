from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.models.vit.modeling_vit import ViTConfig, ViTEmbeddings

from oslo.torch.distributed import ParallelContext, ParallelMode


class VocabUtility:
    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank):
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
        assert (
            global_vocab_size % world_size == 0
        ), "vocab size must be divisible by world size"

        per_partition_vocab_size = global_vocab_size // world_size
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank
        )


class Embedding1D(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: Optional[torch.dtype] = None,
        gather_output: bool = True,
        parallel_context: Optional[ParallelContext] = None,
    ):
        assert parallel_context is not None, "parallel_context must be provided"
        self.gather_output = gather_output
        self.parallel_context = parallel_context
        self.world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)
        assert (
            embedding_dim % self.world_size == 0
        ), "embedding_dim must be divisible by world_size for Embedding1D."

        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim // self.world_size,
            device=torch.device(torch.cuda.current_device()),
            dtype=dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._1d._ops import (
            gather_tensor_1d,
        )

        output = F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        if self.gather_output:
            output = gather_tensor_1d(
                output,
                -1,
                self.parallel_context,
            )
        return output


class VocabParallelEmbedding1D(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: Optional[torch.dtype] = None,
        parallel_context: Optional[ParallelContext] = None,
    ):
        assert parallel_context is not None, "parallel_context must be provided"
        self.parallel_context = parallel_context
        self.rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_1D)
        self.world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)
        assert (
            num_embeddings % self.world_size == 0
        ), "num_embeddings must be divisible by world_size for VocabParallelEmbedding1D."
        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = VocabUtility.vocab_range_from_global_vocab_size(
            num_embeddings, self.rank, self.world_size
        )

        super().__init__(
            num_embeddings=num_embeddings // self.world_size,
            embedding_dim=embedding_dim,
            device=torch.device(torch.cuda.current_device()),
            dtype=dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._1d._ops import (
            reduce_tensor_1d,
        )

        if self.world_size > 1:
            input_mask = (input < self.vocab_start_index) | (
                input >= self.vocab_end_index
            )
            masked_input = input.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input

        output_parallel = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # Mask the output embedding.
        if self.world_size > 1:
            output_parallel[input_mask, :] = 0.0

        # Reduce across all the model parallel GPUs.
        output = reduce_tensor_1d(output_parallel, self.parallel_context)
        return output


class Embedding2D(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: Optional[torch.dtype] = None,
        parallel_context: Optional[ParallelContext] = None,
    ):
        assert parallel_context is not None, "parallel_context must be provided"
        self.parallel_context = parallel_context
        self.summa_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2D_COL
        )
        assert (
            embedding_dim % (self.summa_dim**2) == 0
        ), "embedding_dim must be divisible by summa_dim^2 for Embedding2D."
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim // (self.summa_dim**2),
            device=torch.device(torch.cuda.current_device()),
            dtype=dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._2d._ops import (
            all_gather_tensor_2d,
        )

        weight = all_gather_tensor_2d(
            self.weight,
            dim=-1,
            parallel_context=self.parallel_context,
            parallel_mode=ParallelMode.TENSOR_2D_COL,
        )
        output = F.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return output


class VocabParallelEmbedding2D(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: Optional[torch.dtype] = None,
        parallel_context: Optional[ParallelContext] = None,
    ):
        assert parallel_context is not None, "parallel_context must be provided"
        self.parallel_context = parallel_context
        self.summa_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2D_COL
        )
        assert (
            num_embeddings % self.summa_dim == 0
        ), "num_embeddings must be divisible by summa_dim for VocabParallelEmbedding2D."
        assert (
            embedding_dim % self.summa_dim == 0
        ), "embedding_dim must be divisible by summa_dim for VocabParallelEmbedding2D."
        rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = VocabUtility.vocab_range_from_global_vocab_size(
            num_embeddings,
            rank,
            self.summa_dim,
        )
        super().__init__(
            num_embeddings=num_embeddings // self.summa_dim,
            embedding_dim=embedding_dim // self.summa_dim,
            device=torch.device(torch.cuda.current_device()),
            dtype=dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._2d._ops import (
            gather_batch_2d,
            reduce_scatter_tensor_2d,
        )

        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR)
        input = gather_batch_2d(
            input,
            dim=0,
            parallel_context=self.parallel_context,
        )
        if world_size > 1:
            input_mask = (input < self.vocab_start_index) | (
                input >= self.vocab_end_index
            )
            masked_input = input.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input

        output = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        output[input_mask, :] = 0.0
        output = reduce_scatter_tensor_2d(
            output,
            dim=0,
            parallel_context=self.parallel_context,
            parallel_mode=ParallelMode.TENSOR_2D_COL,
        )
        return output


class ViTEmbedding2D(ViTEmbeddings):
    def __init__(
        self, config: ViTConfig, use_mask_token: bool = False, parallel_context=None
    ) -> None:
        assert parallel_context is not None, "parallel_context must be provided"
        self.parallel_context = parallel_context
        self.summa_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2D_COL
        )

        super().__init__(config, use_mask_token=use_mask_token)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        from oslo.torch.distributed.nn.functional import all_gather

        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_token = all_gather(
            self.cls_token,
            dim=-1,
            parallel_context=self.parallel_context,
            parallel_mode=ParallelMode.TENSOR_2D_COL,
        )
        cls_tokens = cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width
            )
        else:
            position_embeddings = all_gather(
                self.position_embeddings,
                dim=-1,
                parallel_context=self.parallel_context,
                parallel_mode=ParallelMode.TENSOR_2D_COL,
            )
            embeddings = embeddings + position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class Embedding2p5D(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: Optional[torch.dtype] = None,
        parallel_context: Optional[ParallelContext] = None,
    ):
        assert parallel_context is not None, "parallel_context must be provided"
        self.parallel_context = parallel_context
        self.tesseract_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2P5D_COL
        )
        assert (
            embedding_dim % (self.tesseract_dim**2) == 0
        ), "embedding_dim must be divisible by tesseract_dim^2 for Embedding2p5D."
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim // (self.tesseract_dim**2),
            device=torch.device(torch.cuda.current_device()),
            dtype=dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._2p5d._ops import (
            all_gather_tensor_2p5d,
        )

        weight = all_gather_tensor_2p5d(
            self.weight,
            dim=-1,
            parallel_context=self.parallel_context,
            col_parallel_mode=ParallelMode.TENSOR_2P5D_COL,
        )

        output = F.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return output


# TODO: Implement this class.
class VocabParallelEmbedding2p5D(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: Optional[torch.dtype] = None,
        parallel_context: Optional[ParallelContext] = None,
    ):
        assert parallel_context is not None, "parallel_context must be provided"
        self.parallel_context = parallel_context
        self.tesseract_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2P5D_COL
        )
        assert (
            num_embeddings % self.tesseract_dim == 0
        ), "num_embeddings must be divisible by tesseract_dim for VocabParallelEmbedding2p5D."
        assert (
            embedding_dim % self.tesseract_dim == 0
        ), "embedding_dim must be divisible by tesseract_dim for VocabParallelEmbedding2p5D."

        rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = VocabUtility.vocab_range_from_global_vocab_size(
            num_embeddings,
            rank,
            self.tesseract_dim,
        )
        super().__init__(
            num_embeddings=num_embeddings // self.tesseract_dim,
            embedding_dim=embedding_dim // self.tesseract_dim,
            device=torch.device(torch.cuda.current_device()),
            dtype=dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._2p5d._ops import (
            reduce_scatter_tensor_2p5d,
            gather_batch_2p5d,
        )

        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR)
        input = gather_batch_2p5d(
            x,
            dim=0,
            parallel_context=self.parallel_context,
        )
        if world_size > 1:
            input_mask = (input < self.vocab_start_index) | (
                input >= self.vocab_end_index
            )
            masked_input = input.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input

        output = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        output[input_mask, :] = 0.0
        output = reduce_scatter_tensor_2p5d(
            output,
            dim=0,
            parallel_context=self.parallel_context,
            parallel_mode=ParallelMode.TENSOR_2P5D_COL,
        )
        return output


class Embedding3D(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: Optional[torch.dtype] = None,
        parallel_context: Optional[ParallelContext] = None,
    ):
        assert parallel_context is not None, "parallel_context must be provided"
        self.parallel_context = parallel_context
        self.cubic_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_3D_INPUT,
        )
        assert (
            embedding_dim % self.cubic_dim == 0
        ), "embedding_dim must be divisible by cubic_dim for Embedding3D."

        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim // self.cubic_dim,
            device=torch.device(torch.cuda.current_device()),
            dtype=dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._3d._ops import (
            broadcast_weight_3d_from_diagonal,
        )

        weight = broadcast_weight_3d_from_diagonal(
            self.weight,
            parallel_context=self.parallel_context,
            input_parallel_mode=ParallelMode.TENSOR_3D_INPUT,
            weight_parallel_mode=ParallelMode.TENSOR_3D_WEIGHT,
            output_parallel_mode=ParallelMode.TENSOR_3D_OUTPUT,
        )
        output = F.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        return output


class VocabParallelEmbedding3D(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: Optional[torch.dtype] = None,
        parallel_context: Optional[ParallelContext] = None,
    ):
        assert parallel_context is not None, "parallel_context must be provided"
        self.parallel_context = parallel_context
        self.cubic_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_3D_INPUT,
        )
        assert (
            num_embeddings % (self.cubic_dim**2) == 0
        ), "num_embeddings must be divisible by cubic_dim^2 for VocabParallelEmbedding3D."
        assert (
            embedding_dim % self.cubic_dim == 0
        ), "embedding_dim must be divisible by cubic_dim for VocabParallelEmbedding3D."

        rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_3D_INPUT)
        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = VocabUtility.vocab_range_from_global_vocab_size(
            num_embeddings,
            rank,
            self.cubic_dim,
        )
        super().__init__(
            num_embeddings=num_embeddings // (self.cubic_dim**2),
            embedding_dim=embedding_dim // self.cubic_dim,
            device=torch.device(torch.cuda.current_device()),
            dtype=dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._3d._ops import (
            gather_batch_3d,
            all_gather_tensor_3d,
            reduce_scatter_tensor_3d,
        )

        input = gather_batch_3d(
            input,
            dim=0,
            parallel_context=self.parallel_context,
        )
        input_mask = (input < self.vocab_start_index) | (input >= self.vocab_end_index)
        masked_input = input.clone() - self.vocab_start_index
        masked_input[input_mask] = 0

        weight = all_gather_tensor_3d(
            self.weight,
            dim=0,
            parallel_context=self.parallel_context,
            parallel_mode=ParallelMode.TENSOR_3D_WEIGHT,
        )
        output_parallel = F.embedding(
            masked_input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        output_parallel[input_mask, :] = 0.0
        output = reduce_scatter_tensor_3d(
            output_parallel,
            0,
            parallel_context=self.parallel_context,
            parallel_mode=ParallelMode.TENSOR_3D_INPUT,
        )
        return output
