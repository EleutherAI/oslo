#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <string>

#include "context.h"
#include "cross_entropy_layer.h"

using namespace torch::indexing;

// x is torch::Tensor
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

static std::unordered_map<int, std::shared_ptr<void>> s_cross_entropy_layers;

template <typename T>
int create_cross_entropy_layer(const int layer_id, const float epsilon,
                               const int padding_idx,
                               const int max_batch_tokens) {
  auto layer = std::make_shared<CrossEntropyLayer<T>>(epsilon, padding_idx,
                                                      max_batch_tokens);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  Context::Instance().set_stream(stream);
  s_cross_entropy_layers[layer_id] = layer;

  std::string dtype = (std::is_same<T, __half>::value) ? "half" : "float";

  std::cout << "CrossEntropyLayer is created with date type [" << dtype << "]."
            << std::endl;

  return 0;
}

template <typename T>
std::vector<torch::Tensor> cross_entropy_layer_fw(
    const int layer_id, const torch::Tensor &inputs,
    const torch::Tensor &targets) {
  CHECK_INPUT(inputs);
  CHECK_INPUT(targets);
  AT_ASSERTM(targets.dtype() == torch::kInt32, "targets must be int32");

  const T *inputs_ptr = static_cast<const T *>(inputs.data_ptr());
  const int *targets_ptr = static_cast<const int *>(targets.data_ptr());

  int batch_size = inputs.size(0);
  int seq_len = inputs.size(1);
  int vocab_size = inputs.size(2);

  std::shared_ptr<CrossEntropyLayer<T>> layer =
      std::static_pointer_cast<CrossEntropyLayer<T>>(
          s_cross_entropy_layers[layer_id]);

  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .layout(torch::kStrided)
                     .device(torch::kCUDA, inputs.device().index());
  auto outputs = torch::zeros({1}, options);
  auto nll_loss = torch::zeros({1}, options);
  float *outputs_ptr = static_cast<float *>(outputs.data_ptr());
  float *nll_loss_ptr = static_cast<float *>(nll_loss.data_ptr());

  layer->set_cur_batch_shape(batch_size, seq_len, vocab_size);
  layer->Forward(inputs_ptr, targets_ptr, outputs_ptr, nll_loss_ptr);
  return {outputs, nll_loss};
}

template <typename T>
std::vector<torch::Tensor> cross_entropy_layer_bw(
    const int layer_id, const torch::Tensor &grad_outputs,
    const torch::Tensor &inputs, const torch::Tensor &targets) {
  CHECK_INPUT(grad_outputs);
  CHECK_INPUT(inputs);
  CHECK_INPUT(targets);
  AT_ASSERTM(targets.dtype() == torch::kInt32, "targets must be int32");

  const float *grad_outputs_ptr =
      static_cast<const float *>(grad_outputs.data_ptr());
  const T *inputs_ptr = static_cast<const T *>(inputs.data_ptr());
  const int *targets_ptr = static_cast<const int *>(targets.data_ptr());

  int batch_size = inputs.size(0);
  int seq_len = inputs.size(1);
  int vocab_size = inputs.size(2);

  auto grad_inputs = torch::zeros_like(inputs);
  T *grad_inputs_ptr = static_cast<T *>(grad_inputs.data_ptr());

  std::shared_ptr<CrossEntropyLayer<T>> layer =
      std::static_pointer_cast<CrossEntropyLayer<T>>(
          s_cross_entropy_layers[layer_id]);

  layer->set_cur_batch_shape(batch_size, seq_len, vocab_size);
  layer->Backward(grad_outputs_ptr, inputs_ptr, targets_ptr, grad_inputs_ptr);
  return {grad_inputs};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("create_cross_entropy_layer_fp32", &create_cross_entropy_layer<float>,
        "Create LightSeq Cross Entropy Layer with fp32 (CUDA)");
  m.def("create_cross_entropy_layer_fp16", &create_cross_entropy_layer<__half>,
        "Create LightSeq Cross Entropy Layer with fp16 (CUDA)");
  m.def("cross_entropy_layer_fw_fp32", &cross_entropy_layer_fw<float>,
        "LightSeq Cross Entropy forward with fp32 (CUDA)");
  m.def("cross_entropy_layer_fw_fp16", &cross_entropy_layer_fw<__half>,
        "LightSeq Cross Entropy forward with fp16 (CUDA)");
  m.def("cross_entropy_layer_bw_fp32", &cross_entropy_layer_bw<float>,
        "LightSeq Cross Entropy backward with fp32 (CUDA)");
  m.def("cross_entropy_layer_bw_fp16", &cross_entropy_layer_bw<__half>,
        "LightSeq Cross Entropy backward with fp16 (CUDA)");
}
