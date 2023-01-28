#pragma once

#include <string>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <type_traits>

#include "cuda_util.h"
#include "dropout.h"
#include "feed_forward.h"
#include "normalize_layer.h"
#include "softmax.h"
#include "strided_batch_gemm.h"
#include "cublas_algo_map.h"

template <typename T>
class TransformerEncoderLayer {
 public:
  TransformerEncoderLayer(int layer_id, int max_batch_tokens, int _max_seq_len,
                          int hidden_size, int num_heads, int intermediate_size,
                          float attn_dropout_ratio,
                          float hidden_output_dropout_ratio,
                          float layer_norm_eps, bool pre_or_postLayerNorm,
                          std::string activation_fn,
                          bool mask_future_tokens = false);

  virtual ~TransformerEncoderLayer();

  void Forward(const T *input_ptr, const T *input_mask_ptr, T *out_ptr);

  void Backward(const T *grad_output_ptr, const T *input_ptr,
                const T *output_ptr, const T *input_mask_ptr,
                T *grad_input_ptr);

  void attn_layer_fw(const T *input_ptr, const T *input_mask_ptr, T *output_ptr,
                     T *buffer);

  void ffn_layer_fw(T *inp_ptr, T *out_ptr);

  void attn_layer_bw(const T *input_ptr, const T *input_mask_ptr,
                     const T *grad_output_ptr, T *grad_input_ptr, T *buffer);

  void ffn_layer_bw(const T *grad_output_ptr, const T *output_ptr,
                    T *grad_inp_ptr, T *buffer);

  void zero_mask_grad();

  void set_cur_batch_shape(int batch_size, int seq_len) {
    _batch_size = batch_size;
    _seq_len = seq_len;
    _batch_tokens = batch_size * seq_len;
    _batch_heads = batch_size * _heads;
    _batch_dim = _batch_tokens * _hidden_size;
    _attn_scores.SetConfig(_seq_len, _seq_len, _hidden_size / _heads);
    _attn_context.SetConfig(_hidden_size / _heads, _seq_len, _seq_len);
  }

  void SetTrainingMode(bool training);
  void SetQuantMode(bool enable_quant);
  inline bool IsTrainingMode() const { return _training; }

  void assign_weight_ptr(const T *weights_ptr) {
    const T *wptr = weights_ptr;
    // assign weights ptr
    _attn_qkvw_ptr = wptr;
    wptr += _hidden_size * _hidden_size * 3;
    _attn_qkvb_ptr = wptr;
    wptr += _hidden_size * 3;
    _attn_ow_ptr = wptr;
    wptr += _hidden_size * _hidden_size;
    _attn_ob_ptr = wptr;
    wptr += _hidden_size;
    _attn_nw_ptr = wptr;
    wptr += _hidden_size;
    _attn_nb_ptr = wptr;
    wptr += _hidden_size;

    _inter_w_ptr = wptr;
    wptr += _hidden_size * _intermediate_size;
    _inter_b_ptr = wptr;
    wptr += _intermediate_size;
    _output_w_ptr = wptr;
    wptr += _hidden_size * _intermediate_size;
    _output_b_ptr = wptr;
    wptr += _hidden_size;
    _ffn_nw_ptr = wptr;
    wptr += _hidden_size;
    _ffn_nb_ptr = wptr;
    wptr += _hidden_size;

    _attn_qkv_cmax_ptr = wptr;
    wptr += 3;
    _attn_out_cmax_ptr = wptr;
    wptr += 3;
    _inter_cmax_ptr = wptr;
    wptr += 3;
    _output_cmax_ptr = wptr;
    wptr += 2;
    _attn_qkv_cache_cmax_ptr = wptr;
    wptr += 1;
  }

  void assign_grad_ptr(T *grads_ptr) {
    T *gptr = grads_ptr;
    // assign grads ptr
    _grad_attn_qkvw_ptr = gptr;
    gptr += _hidden_size * _hidden_size * 3;
    _grad_attn_qkvb_ptr = gptr;
    gptr += _hidden_size * 3;
    _grad_attn_ow_ptr = gptr;
    gptr += _hidden_size * _hidden_size;
    _grad_attn_ob_ptr = gptr;
    gptr += _hidden_size;
    _grad_attn_nw_ptr = gptr;
    gptr += _hidden_size;
    _grad_attn_nb_ptr = gptr;
    gptr += _hidden_size;

    _grad_inter_w_ptr = gptr;
    gptr += _hidden_size * _intermediate_size;
    _grad_inter_b_ptr = gptr;
    gptr += _intermediate_size;
    _grad_output_w_ptr = gptr;
    gptr += _hidden_size * _intermediate_size;
    _grad_output_b_ptr = gptr;
    gptr += _hidden_size;
    _grad_ffn_nw_ptr = gptr;
    gptr += _hidden_size;
    _grad_ffn_nb_ptr = gptr;
    gptr += _hidden_size;

    _grad_attn_qkv_cmax_ptr = gptr;
    gptr += 3;
    _grad_attn_out_cmax_ptr = gptr;
    gptr += 3;
    _grad_inter_cmax_ptr = gptr;
    gptr += 3;
    _grad_output_cmax_ptr = gptr;
    gptr += 2;
    _grad_attn_qkv_cache_cmax_ptr = gptr;
    gptr += 1;
  }

 private:
  void allocate_mem_buffer() {
    // allocate local gpu memory
    allocate_layer_memory();

    // buffer size needed by ffn bw
    size_t sz_ffn_bw = 3 * _max_batch_tokens * _hidden_size +
                       _max_batch_tokens * _intermediate_size;
    // buffer size needed by attn bw
    size_t sz_attn_bw = 5 * _max_batch_tokens * _hidden_size +
                        std::max(3 * _max_batch_tokens * _hidden_size,
                                 _max_batch_tokens * _heads * _max_seq_len);
    size_t smem_size = std::max(sz_ffn_bw, sz_attn_bw);

    if (!_shared_mem_ptr) {
      cuda_free(_shared_mem_ptr);
      _shared_mem_ptr = cuda_malloc<T>(smem_size);
      std::cout << "Encoder layer #" << _layer_id
                << " allocate shared memory size: " << smem_size << std::endl;
    }
    if (!_shared_quant_mem_ptr) {
      cuda_free(_shared_quant_mem_ptr);
      size_t quant_smem_size =
          std::max(_max_batch_tokens * _hidden_size * 4 +
                       _hidden_size * _hidden_size * 3,
                   _max_batch_tokens * (_hidden_size + _intermediate_size) +
                       _hidden_size * _intermediate_size);
      _shared_quant_mem_ptr = cuda_malloc<int8_t>(quant_smem_size);
      std::cout << "Encoder layer #" << _layer_id
                << " allocate shared quant memory size: " << quant_smem_size
                << std::endl;
    }
  }

  void free_mem_buffer() {
    // free local gpu memory
    free_layer_memory();

    // free shared gpu memory between layers
    cuda_free(_shared_mem_ptr);
    cuda_free(_shared_quant_mem_ptr);
    _shared_mem_ptr = nullptr;
    _shared_quant_mem_ptr = nullptr;
  }

  void allocate_layer_memory() {
    // allocate local gpu memory
    _qkv_ptr = cuda_malloc<T>(_max_batch_tokens * _hidden_size * 3);
    _soft_out_ptr = cuda_malloc<T>(_max_batch_tokens * _heads * _max_seq_len);
    _ctx_bufB_ptr = cuda_malloc<T>(_max_batch_tokens * _heads * _max_seq_len);
    if (_pre_or_postLayerNorm) {
      _gemmQKV_inp_ptr = cuda_malloc<T>(_max_batch_tokens * _hidden_size);
    } else {
      _gemmQKV_inp_ptr = nullptr;
    }
    _attn_o_inp_ptr = cuda_malloc<T>(_max_batch_tokens * _hidden_size);
    _ff1_inp_ptr = cuda_malloc<T>(_max_batch_tokens * _hidden_size);
    _relu_inp_ptr = cuda_malloc<T>(_max_batch_tokens * _intermediate_size);
    _relu_inp_i8_ptr = reinterpret_cast<int8_t *>(_relu_inp_ptr);
    _ff2_inp_ptr = cuda_malloc<T>(_max_batch_tokens * _intermediate_size);
    _igemm_alpha_ptr = cuda_malloc<float>(_max_batch_tokens);
    _igemm_beta_ptr = cuda_malloc<float>(1);
    cuda_set<float>(_igemm_beta_ptr, 0, 1);
  }

  void free_layer_memory() {
    // free local gpu memory
    cuda_free(_igemm_alpha_ptr);
    cuda_free(_igemm_beta_ptr);
    cuda_free(_gemmQKV_inp_ptr);
    cuda_free(_qkv_ptr);
    cuda_free(_soft_out_ptr);
    cuda_free(_ctx_bufB_ptr);
    cuda_free(_attn_o_inp_ptr);
    cuda_free(_ff1_inp_ptr);
    cuda_free(_relu_inp_ptr);
    cuda_free(_ff2_inp_ptr);
  }

  const size_t _layer_id;
  const size_t _hidden_size;
  const size_t _heads;
  const size_t _intermediate_size;
  const size_t _max_batch_tokens;
  const size_t _max_seq_len;
  const bool _pre_or_postLayerNorm;
  const std::string _activation_fn;
  const bool _mask_future_tokens;
  // dynamic parameter between batch
  size_t _batch_size;
  size_t _seq_len;
  size_t _batch_tokens;
  size_t _batch_heads;
  size_t _batch_dim;
  bool _training;
  bool _enable_quant;

  cublasHandle_t _cublasHandle;
  cublasLtHandle_t _cublasLtHandle;
  cudaStream_t _stream;
  cublasAlgoMap _algo_map;

  // layers
  FeedForward<T> _qkv_linear;
  FeedForward<T> _attn_out_linear;
  Normalize_Layer<T> _attn_ln;
  Normalize_Layer<T> _ffn_ln;
  FeedForward<T> _ff1, _ff2;
  Softmax<T> _softmax;
  Dropout<T> _attn_prob_dropout;
  Dropout<T> _attn_dropout;
  Dropout<T> _ffn_activation_dropout;
  Dropout<T> _ffn_dropout;
  StridedBatchGemm<T> _attn_scores;
  StridedBatchGemm<T> _attn_context;

  // local GPU memory
  T *_gemmQKV_inp_ptr;
  T *_qkv_ptr;
  T *_soft_out_ptr;
  T *_ctx_bufB_ptr;
  T *_attn_o_inp_ptr;
  T *_ff1_inp_ptr;
  T *_relu_inp_ptr;
  T *_ff2_inp_ptr;
  int8_t *_relu_inp_i8_ptr;

  // local GPU memory for quant
  float *_igemm_alpha_ptr;
  float *_igemm_beta_ptr;

  // shared GPU memory between layer
  static T *_shared_mem_ptr;
  static int8_t *_shared_quant_mem_ptr;

  // weights ptr
  const T *_attn_qkvw_ptr;
  const T *_attn_qkvb_ptr;
  const T *_attn_ow_ptr;
  const T *_attn_ob_ptr;
  const T *_attn_nw_ptr;
  const T *_attn_nb_ptr;

  const T *_inter_w_ptr;
  const T *_inter_b_ptr;
  const T *_output_w_ptr;
  const T *_output_b_ptr;
  const T *_ffn_nw_ptr;
  const T *_ffn_nb_ptr;
  const T *_attn_qkv_cmax_ptr;
  const T *_attn_out_cmax_ptr;
  const T *_inter_cmax_ptr;
  const T *_output_cmax_ptr;
  const T *_attn_qkv_cache_cmax_ptr;

  // grads ptr
  T *_grad_attn_qkvw_ptr;
  T *_grad_attn_qkvb_ptr;
  T *_grad_attn_ow_ptr;
  T *_grad_attn_ob_ptr;
  T *_grad_attn_nw_ptr;
  T *_grad_attn_nb_ptr;

  T *_grad_inter_w_ptr;
  T *_grad_inter_b_ptr;
  T *_grad_output_w_ptr;
  T *_grad_output_b_ptr;
  T *_grad_ffn_nw_ptr;
  T *_grad_ffn_nb_ptr;
  T *_grad_attn_qkv_cmax_ptr;
  T *_grad_attn_out_cmax_ptr;
  T *_grad_inter_cmax_ptr;
  T *_grad_output_cmax_ptr;
  T *_grad_attn_qkv_cache_cmax_ptr;
};
