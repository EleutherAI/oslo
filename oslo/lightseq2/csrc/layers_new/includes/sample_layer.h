#pragma once

#include "beam_search_topk.h"
#include "layer.h"

namespace lightseq {

template <class T> class SampleLayer : public Layer {
private:
  // operators
  BeamSearchTopOp<T> *_beam_search = nullptr;

  // parameters
  Variable *_logit_bias;
  size_t _trg_vocab_size;

public:
  SampleLayer(int nshared_layer, int max_batch_size, int max_step,
              int trg_vocab_size, int hidden_size, int max_thread_per_block,
              int beam_size, int diverse_lambda, int dim_per_head, int end_id,
              int head_num,
              float length_penalty); // for beam_search

  virtual ~SampleLayer() {}

  std::tuple<Variable *, Variable *> operator()(Variable *logits,
                                                Variable *alive_seq);

  void before_forward(int batch_size, int cur_step);

  int load_params(const std::vector<const T *> &para_vec, int offset);

  bool is_stop() { return _beam_search->is_stop(); }
};

template class SampleLayer<float>;
#ifdef LIGHTSEQ_cuda
template class SampleLayer<__half>;
#endif

template <typename T> using SampleLayerPtr = std::shared_ptr<SampleLayer<T>>;

} // namespace lightseq
