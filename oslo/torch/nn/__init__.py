from oslo.torch.nn.modules.conv import Conv1D #, LazyConv1D
from oslo.torch.nn.modules.dropout import (
    FusedBiasDropout,
)
from oslo.torch.nn.modules.activation import (
    FusedBiasGeLU,
)
from oslo.torch.nn.modules.embedding import (
    Embedding1D,
    Embedding2D,
    Embedding2p5D,
    Embedding3D,
    # LazyEmbedding,
    VocabParallelEmbedding1D,
    VocabParallelEmbedding2D,
    VocabParallelEmbedding2p5D,
    VocabParallelEmbedding3D,
)
from oslo.torch.nn.modules.functional import (
    fused_bias_dropout,
    fused_bias_gelu,
    fused_gelu,
    multi_head_attention_forward,
    fused_scale_mask_softmax,
    fused_rms_norm_affine,
    fused_layer_norm,
    mixed_dtype_fused_layer_norm_affine,
    fused_layer_norm_affine,
    mixed_dtype_fused_rms_norm_affine,
    fused_rms_norm,
)
from oslo.torch.nn.modules.layer_norm import (
    LayerNorm1D,
    LayerNorm2D,
    LayerNorm2p5D,
    LayerNorm3D,
    FusedLayerNorm,
    MixedFusedLayerNorm,
    MixedFusedRMSNorm,
    FusedRMSNorm,
)
from oslo.torch.nn.modules.linear import (
    ColLinear1D,
    # LazyLinear,
    Linear,
    Linear2D,
    Linear2p5D,
    Linear3D,
    RowLinear1D,
)


from oslo.torch.nn.modules.softmax import FusedScaleMaskSoftmax

from oslo.torch.nn.modules.ngram_repeat_block import NGramRepeatBlock

from oslo.torch.nn.modules.functional import _NGramRepeatBlockFunction
