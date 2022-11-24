import copy
import math
from typing import Union

import torch
import torch.distributed as dist
import torch.nn as nn

from oslo.torch.distributed import ParallelMode
from oslo.torch.distributed._seed.helper import seed
from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.nn.parallel.expert_parallel.experts import Experts
from oslo.torch.nn.parallel.expert_parallel.layers import ExpertParallelBehindBlock
from oslo.torch.nn.parallel.expert_parallel.layers import ExpertParallelFrontBlock
from oslo.torch.nn.parallel.expert_parallel.layers import TopKGate
from oslo.torch.nn.parallel.expert_parallel._ops import AllReduce
from oslo.torch.nn.parallel.expert_parallel.mapping import ExpertParallelMapping
from oslo.torch.nn.parallel.utils import _update_module_arguments
from oslo.transformers.mapping_utils import _ExpertParallelMappingForHuggingFace


class _ExpertParallel(nn.Module):
    """
    A class to wrap the given model for expert parallelization

    Args:
        model: model to wrap for expert paralleization
        parallel_context: global parallel context
        num_experts: number of experts
        top_k: the number of experts for each token to be dispatched
        capacity_factor_train: capacity of each expert for training
        capacity_factor_eval: capacity of each expert for evaluation
        min_capacity: minimum capacity of each expert
        select_policy: policy to select top-1 expert ("first" or "random")
        noisy_policy: policy to generate and add noise ("Jitter" or "Gaussian")
        drop_tks: flag to drop tokens in the case that the number of dispatched tokens is larger than capacity
        use_residual: flag to use residual network proposed by
                      DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

    Notes:
        1. Similar design with `torch.nn.parallel.DistributedDataParallel`
        2. Support data parallel for non-expert paramete

    Examples:
        >>> from oslo.torch.nn.parallel.expert_parallel.expert_parallel import ExpertParallel
        >>> model = TransformersModel()
        >>> ep_wrapper = ExpertParallel(model, parallel_context=..., ...)
        >>> optimizer = AnyOptimizer(ep_wrapper.parameters(), lr=3e-5)
        >>> output = ep_wrapper(input_data)
        >>> output.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        parallel_context: ParallelContext,
        num_enc_experts: Union[int, dict] = None,
        num_dec_experts: Union[int, dict] = None,
        top_k: int = 2,
        capacity_factor_train: float = 1.0,
        capacity_factor_eval: float = 1.0,
        min_capacity: int = 4,
        select_policy: str = "first",
        noisy_policy: str = None,
        use_rts: bool = True,
        drop_tokens: bool = True,
        use_residual: bool = False,
    ):
        super().__init__()

        self.model = model
        self.parallel_context = parallel_context
        self.device = torch.cuda.current_device()

        self.use_residual = use_residual
        if use_residual is None:
            self.use_residual = True if top_k == 1 else False

        if noisy_policy is None:
            noisy_policy = "Jitter" if use_residual else "RSample"

        self.top_k = top_k
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        self.min_capacity = min_capacity
        self.noisy_policy = noisy_policy
        self.use_rts = use_rts
        self.drop_tokens = drop_tokens

        mapping = _ExpertParallelMappingForHuggingFace().get_mapping(model)
        self.expert_parallel_mapping = ExpertParallelMapping(mapping)
        self.link_info = dict()

        self.enc_layer_ids, self.dec_layer_ids = self._get_architecture_info()

        self.num_experts = dict()
        self.num_experts["enc"] = self._get_num_experts(
            num_enc_experts, self.enc_layer_ids
        )
        self.num_experts["dec"] = self._get_num_experts(
            num_dec_experts, self.dec_layer_ids
        )

        self._sanity_check()
        self._parallelize()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _sanity_check(self):
        if isinstance(self.parallel_context.expert_parallel_size, int):
            return None

        if "enc" in self.parallel_context.expert_parallel_size:
            # num_experts must be divisible by corresponding expert parallel size
            assert all(
                [
                    self.parallel_context.expert_parallel_size["enc"][k]
                    % self.num_experts["enc"][k]
                    == 0
                    for k in self.num_experts
                ]
            )

        if "dec" in self.parallel_context.expert_parallel_size:
            # num_experts must be divisible by corresponding expert parallel size
            assert all(
                [
                    self.parallel_context.expert_parallel_size["dec"][k]
                    % self.num_experts["dec"][k]
                    == 0
                    for k in self.num_experts
                ]
            )

    @torch.no_grad()
    def _parallelize(self):
        self._parallelize_module()
        self.to(self.device)
        self._synchronize_non_expert_params()
        self._add_allreduce_hook_for_non_expert_params()

    def _parallelize_module(self):
        to_parallelize = [
            (module_name, module) for module_name, module in self.model.named_modules()
        ]
        for module_name, module in to_parallelize:
            if self.expert_parallel_mapping.is_front_parallel(self.model, module_name):
                self._wrap_front(
                    module,
                    module_name,
                    reversed=self.expert_parallel_mapping.is_reversed_param(
                        self.model, module_name
                    ),
                )
                module.__class__ = ExpertParallelFrontBlock
            elif self.expert_parallel_mapping.is_behind_parallel(
                self.model, module_name
            ):
                self._wrap_behind(
                    module,
                    module_name,
                    reversed=self.expert_parallel_mapping.is_reversed_param(
                        self.model, module_name
                    ),
                )
                module.__class__ = ExpertParallelBehindBlock

    def _get_num_experts(self, num_experts, layer_ids):
        num_experts = (
            self.parallel_context.get_world_size(ParallelMode.GLOBAL)
            if num_experts is None
            else num_experts
        )

        if len(layer_ids) == 0:
            return None

        if type(num_experts) is int:
            assert num_experts > 0, "The Number of Experts must be Positive."
            num_experts = {cur_id: num_experts for cur_id in layer_ids}
        elif type(num_experts) is dict:
            assert (
                num_experts.keys() == layer_ids
            ), "The Keys of Experts Dictionary must be equal to the Set of Layer Ids"
        else:
            raise TypeError("num_enc_experts or num_dec_experts must be int or dict")

        return num_experts

    def _get_module_role(self, module_name):
        elem = self.expert_parallel_mapping.search(self.model, module_name)
        if elem is None:
            return None

        if elem.enc_name is not None and elem.enc_name in module_name:
            return "enc"

        if elem.dec_name is not None and elem.dec_name in module_name:
            return "dec"

    def _get_architecture_info(self):
        enc_layer_ids, dec_layer_ids = set(), set()
        for module_name, module in self.model.named_modules():
            role = self._get_module_role(module_name)
            if role is None:
                continue

            if role == "enc":
                enc_layer_ids.add(self._extract_layer_id(module_name))
            elif role == "dec":
                dec_layer_ids.add(self._extract_layer_id(module_name))
            else:
                raise ValueError(
                    "The mapping information about Encoder/Decoder is wrong."
                )

        return enc_layer_ids, dec_layer_ids

    def _extract_layer_id(self, module_name):
        layer_info = self.expert_parallel_mapping.get_layer_info(
            self.model, module_name
        )

        spl_modules = module_name.split(".")
        spl_layer_info = layer_info.split(".")

        layer_ids = list()

        for cur_layer_info in spl_layer_info:
            to_find = spl_modules.index(cur_layer_info)
            layer_ids.append(int(spl_modules[to_find + 1]))

        return tuple(layer_ids)

    def _extract_link_info_key(self, module_name):
        spl_modules = module_name.split(".")

        split_id = len(spl_modules)
        for i, cur_module in enumerate(spl_modules):
            if cur_module.isdigit():
                split_id = i + 1

        return ".".join(spl_modules[:split_id])

    def _wrap_front(self, module: nn.Module, module_name: str, reversed: bool):
        out_features, in_features = module.weight.size()
        if reversed:
            out_features, in_features = in_features, out_features

        layer_id = self._extract_layer_id(module_name)
        role = self._get_module_role(module_name)

        num_experts = self.num_experts[role][layer_id]
        if isinstance(self.parallel_context.expert_parallel_size, int):
            ep_size = self.parallel_context.expert_parallel_size
        else:
            ep_size = self.parallel_context.expert_parallel_size[role][layer_id]

        gate = TopKGate(
            in_features,
            num_experts,
            self.top_k,
            self.capacity_factor_train,
            self.capacity_factor_eval,
            self.min_capacity,
            self.noisy_policy,
            self.drop_tokens,
            self.use_rts,
        )

        expert_parallel_residual, expert_parallel_residual_mix = None, None
        if self.use_residual:
            expert_parallel_residual = copy.deepcopy(module)
            expert_parallel_residual_mix = nn.Linear(in_features, 2)

        if layer_id not in self.link_info:
            self.link_info[layer_id] = dict()

        num_local_experts = num_experts // ep_size
        experts = Experts(module, num_local_experts)

        ep_group = self.parallel_context.get_group(ParallelMode.EXPERT)

        _update_module_arguments(
            module=module,
            link_info=self.link_info[layer_id],
            gate=gate,
            in_features=in_features,
            out_features=out_features,
            front_experts=experts,
            ep_group=ep_group,
            ep_size=ep_size,
            num_local_experts=num_local_experts,
            use_residual=self.use_residual,
            expert_parallel_residual=expert_parallel_residual,
            expert_parallel_residual_mix=expert_parallel_residual_mix,
        )

        delattr(module, "weight")
        if getattr(module, "bias", None) is not None:
            delattr(module, "bias")

    def _wrap_behind(self, module, module_name: str, reversed: bool):
        out_features, in_features = module.weight.size()
        if reversed:
            out_features, in_features = in_features, out_features

        layer_id = self._extract_layer_id(module_name)
        role = self._get_module_role(module_name)

        num_experts = self.num_experts[role][layer_id]
        if isinstance(self.parallel_context.expert_parallel_size, int):
            ep_size = self.parallel_context.expert_parallel_size
        else:
            ep_size = self.parallel_context.expert_parallel_size[role][layer_id]

        expert_parallel_residual = None
        if self.use_residual:
            expert_parallel_residual = copy.deepcopy(module)

        if layer_id not in self.link_info:
            self.link_info[layer_id] = dict()

        num_local_experts = num_experts // ep_size
        experts = Experts(module, num_local_experts)

        ep_group = self.parallel_context.get_group(ParallelMode.EXPERT)

        _update_module_arguments(
            module,
            link_info=self.link_info[layer_id],
            in_features=in_features,
            out_features=out_features,
            behind_experts=experts,
            ep_size=ep_size,
            ep_group=ep_group,
            num_local_experts=num_local_experts,
            use_residual=self.use_residual,
            expert_parallel_residual=expert_parallel_residual,
        )

        delattr(module, "weight")
        if getattr(module, "bias", None) is not None:
            delattr(module, "bias")

    def _synchronize_non_expert_params(self):
        ep_group = self.parallel_context.get_group(ParallelMode.EXPERT)
        src_rank = self.parallel_context.get_ranks_in_group(ParallelMode.EXPERT)[0]

        for para_name, param in self.model.named_parameters():
            conditions = [
                "front_expert" not in para_name,
                "behind_expert" not in para_name,
            ]

            # Broadcast Non Expert Parameter
            if all(conditions):
                dist.broadcast(param, src_rank, group=ep_group)

    def _get_ep_size(self, name=None):
        if isinstance(self.parallel_context.expert_parallel_size, int):
            ep_size = self.parallel_context.expert_parallel_size
        else:
            assert (
                name is not None
            ), "If you use different ep_sizes for each layer, you need to pass the module/parameter name"
            role = self._get_module_role(name)
            layer_id = self._extract_layer_id(name)
            ep_size = self.parallel_context.expert_parallel_size[role][layer_id]

        return ep_size

    def _add_allreduce_hook_for_non_expert_params(self):
        ep_group = self.parallel_context.get_group(ParallelMode.EXPERT)

        for para_name, param in self.model.named_parameters():
            conditions = [
                "front_expert" not in para_name,
                "behind_expert" not in para_name,
            ]
            ep_size = self._get_ep_size(para_name)

            if all(conditions) and param.requires_grad:
                param.register_hook(AllReduce(ep_group, ep_size, para_name))

    @torch.no_grad()
    def deparallelize(self):
        dp_rank = self.parallel_context.get_local_rank(ParallelMode.DATA)
        ep_rank = self.parallel_context.get_local_rank(ParallelMode.EXPERT)

        if dp_rank == 0:
            for name, module in self.model.named_modules():
                if isinstance(module, Experts):
                    layer_id = self._extract_layer_id(name)
                    role = self._get_module_role(name)
                    num_experts = self.num_experts[role][layer_id]

                    depar_w, depar_b = self._deparallelize_experts(
                        name, module, num_experts
                    )

                    if ep_rank == 0:
                        base_expert = module.experts[0]
                        depar_experts = list()

                        for cur_w, cur_b in zip(depar_w, depar_b):
                            new_expert = copy.deepcopy(base_expert)
                            new_expert.weight = nn.parameter.Parameter(cur_w)

                            if cur_b is not None:
                                new_expert.bias = nn.parameter.Parameter(cur_b)

                            depar_experts.append(new_expert)

                        module.experts = nn.ModuleList(depar_experts)

    @torch.no_grad()
    def _deparallelize_experts(self, name, experts, num_experts):
        ep_size = self._get_ep_size(name)

        layer_id = self._extract_layer_id(name)
        role = self._get_module_role(name)
        num_experts = self.num_experts[role][layer_id]

        depar_w = [None for i in range(num_experts)]
        depar_b = [None for i in range(num_experts)]

        for local_expert_id, cur_expert in enumerate(experts.experts):
            ep_group = self.parallel_context.get_group(ParallelMode.EXPERT)

            experts_w = self._gather_tensors(cur_expert.weight, ep_group, ep_size)
            self._sort_by_global_expert_id(
                experts_w, depar_w, local_expert_id, experts.num_local_experts
            )

            if hasattr(cur_expert, "bias"):
                experts_b = self._gather_tensors(cur_expert.bias, ep_group, ep_size)
                self._sort_by_global_expert_id(
                    experts_b, depar_b, local_expert_id, experts.num_local_experts
                )

        return depar_w, depar_b

    @torch.no_grad()
    def _gather_tensors(self, base_tensor, process_group, ep_size):
        gathered = [torch.zeros_like(base_tensor) for _ in range(ep_size)]
        dist.all_gather(gathered, base_tensor, process_group)

        return gathered

    def _sort_by_global_expert_id(self, to_sort, dest, local_expert_id, num_experts):
        if self.parallel_context.get_local_rank(ParallelMode.EXPERT) == 0:

            for ep_rank, tensor in enumerate(to_sort):
                global_expert_id = ep_rank * num_experts + local_expert_id
                assert (
                    dest[global_expert_id] is None
                ), "Duplicate experts occurred during deparallelization"

                dest[global_expert_id] = tensor
