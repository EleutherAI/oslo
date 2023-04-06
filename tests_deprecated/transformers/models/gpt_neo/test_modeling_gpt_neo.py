import sys
sys.path.append("/admin/home-ingyu/oslo/oslo")
import torch

from oslo.transformers.models.gpt_neo.modeling_gpt_neo import (
    GPTNeoModel,
    GPTNeoForCausalLM,
    GPTNeoForSequenceClassification,
)

try:
    from transformers import (
        GPTNeoModel as TransformersGPTNeoModel,
        GPTNeoForCausalLM as TransformersGPTNeoForCausalLM,
        GPTNeoForSequenceClassification as TransformersGPTNeoForSequenceClassification,
        AutoConfig,
    )
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")


def gradient_check(
    oslo_model, transformers_model, batch_size=8, seq_length=10, return_logits=True
):
    print("\n======================= Gradient Check =======================\n")
    batch_size, seq_length = batch_size, seq_length
    sample = torch.randint(0, 1000, (batch_size, seq_length))

    oslo_model.eval()
    transformers_model.eval()

    if return_logits:
        oslo_result = oslo_model(sample).logits
        orig_result = transformers_model(sample).logits
    else:
        oslo_result = oslo_model(sample).last_hidden_state
        orig_result = transformers_model(sample).last_hidden_state

    if torch.allclose(oslo_result, orig_result, atol=1e-3):
        print("Forward result is same\n")
    else:
        print("Forward result is different\n")

    oslo_result.sum().backward()
    orig_result.sum().backward()

    multiple = lambda x, y: x * y
    for oslo, orig in zip(
        oslo_model.named_parameters(), transformers_model.named_parameters()
    ):
        oslo_name, oslo_param = oslo
        orig_name, orig_param = orig

        if oslo_param.grad.dim() == 2:
            num_params = multiple(*oslo_param.grad.size())
        else:
            num_params = len(oslo_param.grad)

        if oslo_name == orig_name:
            result = torch.isclose(oslo_param.grad, orig_param.grad, atol=1e-5).sum()
            if return_logits:
                print(
                    f"{oslo_name:36s} same_grad_ratio:  {result/num_params:.4f}   num_params:{num_params:9d}   num_same_grad:{result:9d}"
                )
            else:
                print(
                    f"{oslo_name:24s} same_grad_ratio:  {result/num_params:.4f}   num_params:{num_params:9d}   num_same_grad:{result:9d}"
                )

    oslo_model.zero_grad()
    transformers_model.zero_grad()
    print("\n============================ End ============================\n")


if __name__ == "__main__":
    oslo_model = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-125M")
    orig_model = TransformersGPTNeoModel.from_pretrained("EleutherAI/gpt-neo-125M")
    gradient_check(oslo_model, orig_model, return_logits=False)

    oslo_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    orig_model = TransformersGPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    gradient_check(oslo_model, orig_model)

    oslo_model = GPTNeoForSequenceClassification.from_pretrained("EleutherAI/gpt-neo-125M")
    orig_model = TransformersGPTNeoForSequenceClassification.from_pretrained("EleutherAI/gpt-neo-125M")
    gradient_check(oslo_model, orig_model, batch_size=1)

    config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-125M")
    config.reorder_and_upcast_attn = True
    oslo_model = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-125M", config=config)
    orig_model = TransformersGPTNeoModel.from_pretrained("EleutherAI/gpt-neo-125M", config=config)
    gradient_check(oslo_model, orig_model, return_logits=False)

    config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-125M")
    config.softmax_in_fp32 = False  # default is True
    oslo_model = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-125M", config=config)
    orig_model = TransformersGPTNeoModel.from_pretrained("EleutherAI/gpt-neo-125M", config=config)
    gradient_check(oslo_model, orig_model, return_logits=False)
