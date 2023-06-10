#pragma once
#include "layer_normalize.h"
#include "launch_dec_emb_op.h"
#include "layer.h"

namespace lightseq {

template <typename T>
class LaunchDecEmbLayer : public Layer {
 private:
  // operators
  LaunchDecEmbOp<T>* _launch_dec_op = nullptr;

  // parameters
  Variable* _token_emb;
  Variable* _pos_emb;
  Variable* _lang_emb;
  Variable* _lang_id;

 public:
  LaunchDecEmbLayer(size_t max_batch_tokens, size_t beam_size,
                    size_t hidden_size, size_t trg_vocab_size, size_t max_step,
                    int multilg_type)
      : Layer("LaunchDecEmbLayer"),
        _launch_dec_op(new LaunchDecEmbOp<T>(max_batch_tokens, beam_size,
                                             hidden_size, trg_vocab_size,
                                             max_step, multilg_type)) {
    _token_emb = new Variable("token_emb", g_dtype<T>());
    _pos_emb = new Variable("pos_emb", g_dtype<T>());
    _lang_emb = new Variable("lang_emb", g_dtype<T>());
    _lang_id = new Variable("lang_id", g_dtype<T>());

    this->_context_ptr->exit_layer();  // necessary
  }

  virtual ~LaunchDecEmbLayer() {}

  Variable* operator()(Variable* inp) {
    set_inputs({inp});

    Variable* out =
        (*_launch_dec_op)(inp, _token_emb, _pos_emb, _lang_emb, _lang_id);

    set_outputs({out});
    return out;
  }

  void before_forward(size_t batch_size, int cur_step) {
    _launch_dec_op->before_forward(batch_size, cur_step);
  }

  int load_params(const std::vector<const T*>& para_vec, int offset) {
    _token_emb->set_value((char*)para_vec[offset]);
    _pos_emb->set_value((char*)para_vec[offset + 1]);
    // _lang_emb->set_value((char*)para_vec[offset + 4]);
    return 0;
  }
};

template class LaunchDecEmbLayer<float>;
#ifdef LIGHTSEQ_cuda
template class LaunchDecEmbLayer<__half>;
#endif

template <class T>
using LaunchDecEmbLayerPtr = std::shared_ptr<LaunchDecEmbLayer<T>>;

}  // namespace lightseq
