from oslo.transformers.models.gpt2.modeling_gpt2 import (
    GPT2Model,
    GPT2LMHeadModel,
    GPT2DoubleHeadsModel,
    GPT2ForSequenceClassification,
    GPT2ForTokenClassification,
    GPT2PreTrainedModel,
)

from oslo.transformers.models.bert.modeling_bert import (
    BertForPreTraining,
    BertLMHeadModel,
    BertForMaskedLM,
    BertForNextSentencePrediction,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertForMultipleChoice,
)

__ALL__ = [GPT2Model, GPT2LMHeadModel, ...]

# TODO : add rest of oslo models
