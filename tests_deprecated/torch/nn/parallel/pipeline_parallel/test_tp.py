import os.path
import time
import types
from copy import deepcopy

# for debugging
from typing import Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import load_dataset
from torch.distributed import rpc
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    T5ForConditionalGeneration,
    set_seed,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
)

from oslo.torch.distributed import ParallelContext
from oslo.torch.distributed.parallel_mode import ParallelMode
from oslo.torch.nn.parallel import PipelineParallel
from oslo.torch.nn.parallel.pipeline_parallel._buffers import _MODULE_DEVICE_LOCATIONS
from oslo.torch.nn.parallel.tensor_parallel.tensor_parallel import TensorParallel
from oslo.torch.nn.parallel.utils import parallelize
from oslo.transformers.constants import BATCH_DIMENSIONS_PP


class T5Debug(T5ForConditionalGeneration):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


matplotlib.use("Agg")
torch.autograd.set_detect_anomaly(True)
set_seed(42)

data_parallel_size = 1
parallel_context = ParallelContext.from_torch(
    data_parallel_size=data_parallel_size,
    pipeline_parallel_size=1,
    tensor_parallel_size=2,
)

current_device = torch.cuda.current_device()
num_micro_batches = 1

model_name = "gpt2"
config = GPT2Config.from_pretrained(model_name)
model = GPT2LMHeadModel(config)

# model_name = "t5-small"
# config = T5Config.from_pretrained(model_name)
# config.dropout_rate = 0.0
# model = T5ForConditionalGeneration(config)
# model = T5Debug(config)

# model_name = "facebook/bart-base"
# config = BartConfig.from_pretrained(model_name)
# config.dropout_rate = 0.
# model = BartForConditionalGeneration(config)


for n, m in model.named_modules():
    if isinstance(m, nn.Dropout):
        m.p = 0.0

model_no_tp = deepcopy(model)
model_no_tp.cuda()

wrapper = TensorParallel(model, parallel_context=parallel_context)

wrapper.train()

optimizer_tp = Adam(wrapper.parameters(), lr=3e-4)
optimizer_no_tp = Adam(model_no_tp.parameters(), lr=3e-4)

parallelize(wrapper, parallel_context)


if torch.distributed.get_rank() == 1:
    for k, v in _MODULE_DEVICE_LOCATIONS.items():
        print(f"{k}: {v}")


target_step = 50
# save_dir = None
save_dir = "tmp_tp"
if save_dir is not None:
    os.makedirs(save_dir, exist_ok=True)

    def save_output_hook(name, is_parallel):
        def hook(module, inp, outp):
            if name == "":
                return

            if is_parallel:
                if isinstance(outp, types.GeneratorType):
                    return
                torch.cuda.synchronize()
                torch.save(
                    outp,
                    f"{save_dir}/output_{name}_tp_{torch.distributed.get_rank()}.pkl",
                )
            else:
                torch.cuda.synchronize()
                torch.save(outp, f"{save_dir}/output_{name}_no_tp.pkl")

        return hook

    for name, m in wrapper.named_modules():
        m.register_forward_hook(save_output_hook(name, True))

    for name, m in model_no_tp.named_modules():
        m.register_forward_hook(save_output_hook(name, False))


def run():
    batch_size = 8 * num_micro_batches
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    datasets = load_dataset("squad").data["train"]["context"]
    datasets = [str(sample) for sample in datasets[:8192]]
    dataloader = DataLoader(datasets, batch_size=batch_size)

    tp_losses = []
    no_tp_losses = []

    step_count = 0
    with torch.enable_grad():
        # with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs = tokenizer(
                data,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to("cuda")

            inputs["input_ids"][inputs["input_ids"] == tokenizer.pad_token] = -100

            optimizer_tp.zero_grad(set_to_none=True)
            optimizer_no_tp.zero_grad(set_to_none=True)

            out_tp = wrapper(**inputs, labels=inputs["input_ids"])
            loss_tp = out_tp.loss
            loss_tp.backward()

            out_no_tp = model_no_tp(**inputs, labels=inputs["input_ids"])
            loss_no_tp = out_no_tp.loss
            loss_no_tp.backward()

            if dist.get_rank() == 0:
                print(f"{dist.get_rank()}, {loss_tp}, {loss_no_tp}")

            print(f"RANK {torch.distributed.get_rank()} | call step {step_count}")

            # TODO; split optimizer_pp? barrier?

            if save_dir is not None and step_count == target_step:
                for name, param in wrapper.named_parameters():
                    if param.grad is not None:
                        torch.save(
                            param.grad,
                            f"{save_dir}/grad_{name}_tp_{torch.distributed.get_rank()}.pkl",
                        )

                for name, param in model_no_tp.named_parameters():
                    torch.save(param.grad, f"{save_dir}/grad_{name}_no_tp.pkl")

                break

            optimizer_tp.step()
            optimizer_no_tp.step()

            step_count += 1

            tp_losses.append(loss_tp.item())
            no_tp_losses.append(loss_no_tp.item())

    if dist.get_rank() == 0:
        plt.figure(figsize=(32, 8))
        plt.plot(tp_losses, label="TP")
        plt.plot(no_tp_losses, label="no TP")
        plt.legend()
        plt.title(f"{model_name}")
        plt.savefig(f"{model_name} tp_vs_no_tp.png")


if __name__ == "__main__":
    run()
