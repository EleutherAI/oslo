#pragma once
#include "feed_forward_layer.h"
#include "layer.h"
#include "multihead_attention_layer.h"

namespace lightseq {

template <class T1, class T2> class TransformerEncoderLayer : public Layer {
private:
  MultiheadAttentionLayerPtr<T1, T2> _attn_layer;
  FeedForwardLayerPtr<T1, T2> _ffn_layer;

  int _layer_id;

public:
  TransformerEncoderLayer(int layer_id, int max_batch_tokens, int max_seq_len,
                          int hidden_size, int num_heads, int intermediate_size,
                          float attn_prob_dropout_ratio,
                          float activation_dropout_ratio,
                          float hidden_output_dropout_ratio, bool is_pre_ln,
                          std::string activation_fn, bool mask_future_tokens);
  virtual ~TransformerEncoderLayer() {}

  Variable *operator()(Variable *inp, Variable *inp_mask);

  void before_forward(int batch_size, int seq_len) {
    _attn_layer->before_forward(batch_size, seq_len);
    _ffn_layer->before_forward(batch_size, seq_len);
  }

  void before_backward() { return; }

  size_t load_para_and_grad(const T1 *para_ptr, T2 *grad_ptr);

  int load_params(const std::vector<const T1 *> &para_vec, int offset);
};

template class TransformerEncoderLayer<float, float>;
#ifdef LIGHTSEQ_cuda
template class TransformerEncoderLayer<__half, __half>;
#endif

template <class T1, class T2>
using TransformerEncoderLayerPtr =
    std::shared_ptr<TransformerEncoderLayer<T1, T2>>;

} // namespace lightseq
