import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, set_seed

set_seed(1234)
# fix seed value for reproducibility


def backward_hook(*args, **kwargs):
    print("executed custom backward")


def is_same(model1, model2):
    check_results = []
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        check_results.append(torch.allclose(param1, param2, rtol=0, atol=0))
        # (rtol=0, atol=0) means exact sameness

    return all(check_results)


model_name = "hf-internal-testing/tiny-random-GPT2Model"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
config = GPT2Config.from_pretrained(
    model_name, attn_pdrop=0, embd_pdrop=0, resid_pdrop=0
)
# disable all dropouts for reproducibility

model_no_custom = GPT2LMHeadModel(config).train().cuda()
model_custom = GPT2LMHeadModel(config).train().cuda()
model_custom.load_state_dict(model_no_custom.state_dict())
# copy model_no_custom's parameters to model_custom for reproducibility
model_custom.register_backward_hook(backward_hook)

optimizer_no_custom = torch.optim.SGD(model_no_custom.parameters(), lr=5e-3)
optimizer_custom = torch.optim.SGD(model_custom.parameters(), lr=5e-3)

for step in range(500):
    optimizer_custom.zero_grad()
    optimizer_no_custom.zero_grad()

    input_ids = tokenizer("hello", return_tensors="pt").input_ids.cuda()
    loss_no_custom = model_no_custom(input_ids, labels=input_ids).loss
    loss_custom = model_custom(input_ids, labels=input_ids).loss
    # this guarantees mathematical same result with original backward pass.

    print("executed model forward and backward")

    loss_custom.backward()
    loss_no_custom.backward()
    # do backward

    optimizer_custom.step()
    optimizer_no_custom.step()
    # do update

    print(
        f"step={step}, "
        f"no_custom={loss_no_custom}, "
        f"custom={loss_custom}, "
        f"is_same={is_same(model_no_custom, model_custom)}"
        f"\n"
    )

"""
executed custom forward
executed model forward and backward
executed custom backward
step=0, no_custom=6.988645553588867, custom=6.988645553588867, is_same=True

executed custom forward
executed model forward and backward
executed custom backward
step=1, no_custom=6.519345760345459, custom=6.519345760345459, is_same=True

executed custom forward
executed model forward and backward
executed custom backward
step=2, no_custom=6.37519645690918, custom=6.37519645690918, is_same=True

executed custom forward
executed model forward and backward
executed custom backward
step=3, no_custom=6.266267776489258, custom=6.266267776489258, is_same=True

executed custom forward
executed model forward and backward
executed custom backward
step=4, no_custom=6.170551300048828, custom=6.170551300048828, is_same=True
"""
