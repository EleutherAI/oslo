"""
python3 test_loading.py
"""
from oslo.torch._C import (
    CPUAdamBinder,
    CPUAdagradBinder,
    FusedAdagradBinder,
    FusedAdamBinder,
    FusedNovogradBinder,
    FusedLambBinder,
    FusedLayerNormBinder,
    FusedSGDBinder,
    FusedMixedPrecisionLambBinder,
    FusedMixedPrecisionL2NormBinder,
    FusedL2NormBinder,
    ExpertParallelBinder,
    NgramRepeatBlockBinder,
)


def test_cpu_adam_bind():
    print("> Test load CPUAdam...", end="")
    CPUAdamBinder().bind()
    print("OK")


def test_cpu_adagrad_bind():
    print("> Test load CPUAdagrad...", end="")
    CPUAdagradBinder().bind()
    print("OK")


def test_fused_adagrad_bind():
    print("> Test load FusedAdagrad...", end="")
    FusedAdagradBinder().bind()
    print("OK")


def test_fused_adam_bind():
    print("> Test load FusedAdam...", end="")
    FusedAdamBinder().bind()
    print("OK")


def test_fused_novograd_bind():
    print("> Test load FusedNovograd...", end="")
    FusedNovogradBinder().bind()
    print("OK")


def test_fused_lamb_bind():
    print("> Test load FusedLamb...", end="")
    FusedLambBinder().bind()
    print("OK")


def test_fused_layer_norm_bind():
    print("> Test load FusedLayerNorm...", end="")
    FusedLayerNormBinder().bind()
    print("OK")


def test_fused_sgd_bind():
    print("> Test load FusedSGD...", end="")
    FusedSGDBinder().bind()
    print("OK")


def test_fused_mixed_precision_lamb_bind():
    print("> Test load FusedMixedPrecisionLamb...", end="")
    FusedMixedPrecisionLambBinder().bind()
    print("OK")


def test_fused_mixed_precision_l2_norm_bind():
    print("> Test load FusedMixedPrecisionL2Norm...", end="")
    FusedMixedPrecisionL2NormBinder().bind()
    print("OK")


def test_fused_l2_norm_bind():
    print("> Test load FusedL2Norm...", end="")
    FusedL2NormBinder().bind()
    print("OK")


def test_expert_parallel_bind():
    print("> Test load ExpertParallel...", end="")
    ExpertParallelBinder().bind()
    print("OK")


def test_ngram_repeat_block_bind():
    print("> Test load NgramRepeatBlock...", end="")
    NgramRepeatBlockBinder().bind()
    print("OK")


if __name__ == "__main__":
    print("Test tests/torch/_C/test_loading.py")
    test_cpu_adam_bind()
    test_cpu_adagrad_bind()
    test_fused_adagrad_bind()
    test_fused_adam_bind()
    test_fused_novograd_bind()
    test_fused_lamb_bind()
    test_fused_layer_norm_bind()
    test_fused_sgd_bind()
    test_fused_mixed_precision_lamb_bind()
    test_fused_mixed_precision_l2_norm_bind()
    test_fused_l2_norm_bind()
    test_expert_parallel_bind()
    test_ngram_repeat_block_bind()
