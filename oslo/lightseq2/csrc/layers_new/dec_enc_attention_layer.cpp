#include "dec_enc_attention_layer.h"

namespace lightseq {

template <typename T1, typename T2>
DecEncAttentionLayer<T1, T2>::DecEncAttentionLayer(
    size_t layer_id, size_t max_batch_tokens, size_t max_seq_len,
    size_t hidden_size, size_t num_heads, float attn_prob_dropout_ratio,
    float hidden_output_dropout_ratio, bool is_pre_ln)
    : Layer("DecEncAttentionLayer"), // necessary
      _layer_id(layer_id), _max_batch_tokens(max_batch_tokens),
      _max_seq_len(max_seq_len), _hidden_size(hidden_size), _heads(num_heads),
      _is_pre_ln(is_pre_ln),
      // operators
      _attn_ln(
          new LayerNormalizeOp<T1, T2>(max_batch_tokens, hidden_size, false)),
      _q_linear(
          new LinearOp<T1, T2>(max_batch_tokens, hidden_size, hidden_size)),
      _bias_add_transform_20314_q(new BiasAddTrans20314<T1, T2>(
          max_batch_tokens, num_heads, hidden_size, 1)),
      _attn_scores(new StridedBatchGemmOp<T1, T2>(
          max_batch_tokens * num_heads * max_seq_len,
          (float(1.0) / float(sqrt(hidden_size / num_heads))), T1(0.0),
          MATRIX_OP::Transpose, MATRIX_OP::NonTranspose)),
      _softmax(new SoftmaxOp<T1, T2>(max_batch_tokens, max_seq_len, num_heads)),
      _attn_prob_dropout(new DropoutOp<T1, T2>(
          attn_prob_dropout_ratio, max_batch_tokens * num_heads * max_seq_len)),
      _attn_context(new StridedBatchGemmOp<T1, T2>(
          max_batch_tokens * hidden_size, float(1.0), float(0.0),
          MATRIX_OP::NonTranspose, MATRIX_OP::NonTranspose)),
      _transform_0213(
          new Transform0213OP<T1, T2>(max_batch_tokens * hidden_size)),
      _attn_out_linear(
          new LinearOp<T1, T2>(max_batch_tokens, hidden_size, hidden_size)),
      _attn_dropout(new BiasDropoutResOp<T1, T2>(
          hidden_output_dropout_ratio, max_batch_tokens, hidden_size)) {
  // parameters
  _attn_qw = new Variable("_attn_qw", g_dtype<T1>(), g_dtype<T2>());
  _attn_qb = new Variable("_attn_qb", g_dtype<T1>(), g_dtype<T2>());

  _attn_ow = new Variable("_attn_ow", g_dtype<T1>(), g_dtype<T2>());
  _attn_ob = new Variable("_attn_ob", g_dtype<T1>(), g_dtype<T2>());

  _attn_nw = new Variable("_attn_nw", g_dtype<T1>(), g_dtype<T2>());
  _attn_nb = new Variable("_attn_nb", g_dtype<T1>(), g_dtype<T2>());

  this->_context_ptr->exit_layer(); // necessary
}

template <typename T1, typename T2>
Variable *DecEncAttentionLayer<T1, T2>::
operator()(Variable *inp, Variable *enc_mask, Variable *enc_k,
           Variable *enc_v) {
  set_inputs({inp, enc_mask, enc_k, enc_v});

  Variable *q_linear_out = nullptr;
  if (_is_pre_ln) {
    Variable *attn_ln_out = (*_attn_ln)(inp, _attn_nw, _attn_nb);
    q_linear_out = (*_q_linear)(attn_ln_out, _attn_qw);
  } else {
    q_linear_out = (*_q_linear)(inp, _attn_qw);
  }

  Variable *transform_20314_out =
      (*_bias_add_transform_20314_q)(q_linear_out, _attn_qb);

  Variable *attn_score = (*_attn_scores)(enc_k, transform_20314_out);

  Variable *soft_out = (*_softmax)(attn_score, enc_mask);

  Variable *prob_dropout = (*_attn_prob_dropout)(soft_out);
  Variable *attn_context = (*_attn_context)(enc_v, prob_dropout);

  Variable *transform_0213_out = (*_transform_0213)(attn_context);

  Variable *attn_linear = (*_attn_out_linear)(transform_0213_out, _attn_ow);

  Variable *attn_dropout_residual =
      (*_attn_dropout)(attn_linear, _attn_ob, inp);
  if (_is_pre_ln) {
    set_outputs({attn_dropout_residual});
    return attn_dropout_residual;
  }

  Variable *attn_ln_out =
      (*_attn_ln)(attn_dropout_residual, _attn_nw, _attn_nb);
  set_outputs({attn_ln_out});
  return attn_ln_out;
}

template <typename T1, typename T2>
void DecEncAttentionLayer<T1, T2>::before_forward(int batch_size,
                                                  int trg_seq_len,
                                                  int src_seq_len) {
  _batch_tokens = batch_size * trg_seq_len;
  _batch_heads = batch_size * _heads;
  _batch_dim = _batch_tokens * _hidden_size;

  _attn_ln->before_forward(batch_size, trg_seq_len);

  _q_linear->before_forward(_batch_tokens);

  _bias_add_transform_20314_q->before_forward(batch_size, trg_seq_len);

  _attn_scores->before_forward(src_seq_len, trg_seq_len, _hidden_size / _heads,
                               _batch_heads);

  _softmax->before_forward(batch_size, trg_seq_len, src_seq_len);

  _attn_prob_dropout->before_forward(_batch_heads * trg_seq_len * src_seq_len);

  _attn_context->before_forward(_hidden_size / _heads, trg_seq_len, src_seq_len,
                                _batch_heads);

  _transform_0213->before_forward(batch_size, _heads, trg_seq_len,
                                  _hidden_size / _heads);

  _attn_out_linear->before_forward(_batch_tokens);

  _attn_dropout->before_forward(_batch_tokens, _hidden_size);
}

template <typename T1, typename T2>
void DecEncAttentionLayer<T1, T2>::before_backward() {}

template <typename T1, typename T2>
size_t
DecEncAttentionLayer<T1, T2>::load_para_and_grad(const T1 *para_ptr,
                                                 T2 *grad_ptr) { // for training
  size_t offset = 0;
  _attn_qw->set_value((char *)(para_ptr + offset));
  _attn_qw->set_grad((char *)(grad_ptr + offset));
  _attn_qw->set_shape({_hidden_size, _hidden_size});
  offset += _hidden_size * _hidden_size;

  _attn_qb->set_value((char *)(para_ptr + offset));
  _attn_qb->set_grad((char *)(grad_ptr + offset));
  _attn_qb->set_shape({_hidden_size});
  offset += _hidden_size;

  _attn_ow->set_value((char *)(para_ptr + offset));
  _attn_ow->set_grad((char *)(grad_ptr + offset));
  _attn_ow->set_shape({_hidden_size, _hidden_size});
  offset += _hidden_size * _hidden_size;

  _attn_ob->set_value((char *)(para_ptr + offset));
  _attn_ob->set_grad((char *)(grad_ptr + offset));
  _attn_ob->set_shape({_hidden_size});
  offset += _hidden_size;

  _attn_nw->set_value((char *)(para_ptr + offset));
  _attn_nw->set_grad((char *)(grad_ptr + offset));
  _attn_nw->set_shape({_hidden_size});
  offset += _hidden_size;

  _attn_nb->set_value((char *)(para_ptr + offset));
  _attn_nb->set_grad((char *)(grad_ptr + offset));
  _attn_nb->set_shape({_hidden_size});
  offset += _hidden_size;

  return offset;
}

template <typename T1, typename T2>
int DecEncAttentionLayer<T1, T2>::load_params(
    const std::vector<const T1 *> &para_vec, int offset) { // for inference
  int size = 0;
  _attn_nw->set_value((char *)para_vec[offset + size]), size++;
  _attn_nw->set_shape({_hidden_size});
  _attn_nb->set_value((char *)para_vec[offset + size]), size++;
  _attn_nb->set_shape({_hidden_size});

  _attn_qw->set_value((char *)para_vec[offset + size]), size++;
  _attn_qw->set_shape({_hidden_size, _hidden_size});
  _attn_qb->set_value((char *)para_vec[offset + size]), size++;
  _attn_qb->set_shape({_hidden_size});

  _attn_ow->set_value((char *)para_vec[offset + size]), size++;
  _attn_ow->set_shape({_hidden_size, _hidden_size});
  _attn_ob->set_value((char *)para_vec[offset + size]), size++;
  _attn_ob->set_shape({_hidden_size});

  return size;
}

} // namespace lightseq
