# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from model import QwenForCausalLM
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization import QuantMode



def extract_layer_idx(name):
    ss = name.split('.')
    for s in ss:
        if s.isdigit():
            return s
    return None


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx])
    else:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])


def load_from_hf_qwen(tensorrt_llm_model,
                             hf_model,
                             mapping=Mapping(),
                             dtype="float32"):
    tensorrt_llm.logger.info('Loading weights from HF Qwen...')
    time.time()

    quant_mode = getattr(tensorrt_llm_model, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()

    str_dtype_to_torch(dtype)

    if mapping.is_first_pp_rank():
        v = torch_to_numpy(hf_model.transformer.wte.weight.detach().cpu())
        if tensorrt_llm_model.use_parallel_embedding:
            v = split(v, mapping.tp_size, mapping.tp_rank, tensorrt_llm_model.embedding_sharding_dun)

        tensorrt_llm_model.vocab_embedding.weight.value = v

    if mapping.is_last_pp_rank():
        v = torch_to_numpy(hf_model.transformer.ln_f.weight.detach().cpu())
        tensorrt_llm_model.ln_f.weight.value = v

        v = torch_to_numpy(hf_model.lm_head.weight.detach().cpu())
        v = split(v, mapping.tp_size, mapping.tp_rank)
        tensorrt_llm_model.lm_head.weight.value = v


    def load_quant_weight(src, value_dst, scale_dst,
                          plugin_weight_only_quant_type):
        v = np.ascontiguousarray(src.transpose())
        processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
            torch.tensor(v), plugin_weight_only_quant_type)
        # workaround for trt not supporting int8 inputs in plugins currently
        value_dst.value = processed_torch_weights.view(
            dtype=torch.float32).numpy()
        scale_dst.value = torch_weight_scales.numpy()

    # 属于该pipeline的layer
    layers_per_pipeline_stage = hf_model.config.num_hidden_layers // mapping.pp_size
    layers_range = list(range(mapping.pp_rank * layers_per_pipeline_stage, (mapping.pp_rank + 1) * layers_per_pipeline_stage))
    for idx in range(hf_model.config.num_hidden_layers):
        if idx not in layers_range:
            continue
        i = idx - mapping.pp_rank * layers_per_pipeline_stage
        # layer norm 不做tensor parallel
        tensorrt_llm.logger.info(f'Loading weights from HF Qwen, layers_{i}...')
        tensorrt_llm_model.layers[i].input_layernorm.weight.value = torch_to_numpy(
            hf_model.transformer.h[idx].ln_1.weight.detach().cpu())
        tensorrt_llm_model.layers[i].post_layernorm.weight.value = torch_to_numpy(
            hf_model.transformer.h[idx].ln_2.weight.detach().cpu())
    
        # qkv_bias

        v = torch_to_numpy(hf_model.transformer.h[idx].attn.c_attn.bias.detach().cpu())
        v = v.reshape(3, -1)
        v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
        v = v.flatten()
        
        tensorrt_llm_model.layers[i].attention.qkv.bias.value = v

        # qkv_weight
        v = torch_to_numpy(hf_model.transformer.h[idx].attn.c_attn.weight.detach().cpu())
        out_dim = v.shape[0] // 3
        in_dim = v.shape[1]
        # 需要把q,k,v分别拆分
        v = v.reshape(3, out_dim, in_dim)
        v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
        v = v.reshape(-1, in_dim)

        if use_weight_only:
            load_quant_weight(
                src=v,
                value_dst=tensorrt_llm_model.layers[i].attention.qkv.weight,
                scale_dst=tensorrt_llm_model.layers[i].attention.qkv.per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
        else:
            tensorrt_llm_model.layers[i].attention.qkv.weight.value = v

        # atten dense
        v = torch_to_numpy(hf_model.transformer.h[idx].attn.c_proj.weight.detach().cpu())
        v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
        if use_weight_only:
            load_quant_weight(
                src=v,
                value_dst=tensorrt_llm_model.layers[i].attention.dense.weight,
                scale_dst=tensorrt_llm_model.layers[i].attention.dense.per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
        else:
            tensorrt_llm_model.layers[i].attention.dense.weight.value = v

        # mlp fc1
        v = torch_to_numpy(hf_model.transformer.h[idx].mlp.w1.weight.detach().cpu())
        v = split(v, mapping.tp_size, mapping.tp_rank)
        if use_weight_only:
            load_quant_weight(
                src=v,
                value_dst=tensorrt_llm_model.layers[i].mlp.fc1.weight,
                scale_dst=tensorrt_llm_model.layers[i].mlp.fc1.per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
        else:
            tensorrt_llm_model.layers[i].mlp.fc1.weight.value = v
        
        # mlp fc2
        v = torch_to_numpy(hf_model.transformer.h[idx].mlp.w2.weight.detach().cpu())
        v = split(v, mapping.tp_size, mapping.tp_rank)
        if use_weight_only:
            load_quant_weight(
                src=v,
                value_dst=tensorrt_llm_model.layers[i].mlp.fc2.weight,
                scale_dst=tensorrt_llm_model.layers[i].mlp.fc2.per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
        else:
            tensorrt_llm_model.layers[i].mlp.fc2.weight.value = v
        
        # mlp c_proj
        v = torch_to_numpy(hf_model.transformer.h[idx].mlp.c_proj.weight.detach().cpu())
        v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
        if use_weight_only: 
            load_quant_weight(
                src=v,
                value_dst=tensorrt_llm_model.layers[i].mlp.proj.weight,
                scale_dst=tensorrt_llm_model.layers[i].mlp.proj.per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
        else:
            tensorrt_llm_model.layers[i].mlp.proj.weight.value = v

    return tensorrt_llm_model


if __name__ == '__main__':
    from tensorrt_llm.layers.attention import PositionEmbeddingType
    from tensorrt_llm.models import weight_only_quantize
    from tensorrt_llm.quantization import QuantMode

    kv_dtype = 'float16'
    quant_mode = QuantMode.use_weight_only(False)
    tensorrt_llm_qwen = QwenForCausalLM(
        num_layers=28,
        num_heads=32,
        hidden_size=4096,
        inter_size=None,
        vocab_size=65024,
        hidden_act='swiglu',
        max_position_embeddings=4096,
        position_embedding_type=PositionEmbeddingType.learned_absolute,
        rotary_embedding_percentage=1.0,
        dtype=kv_dtype,
        tensor_parallel=1,  # TP only
        tensor_parallel_group=list(range(1)),  # TP only
        apply_query_key_layer_scaling=False,
        quant_mode=quant_mode,
        bias=False,
        multi_query_mode=False)
    tensorrt_llm_qwen = weight_only_quantize(
        tensorrt_llm_qwen, quant_mode)

    model_dir = './pyTorchModel'

    print(f'Loading HF Chat_GLM2 ... from {model_dir}')

    import transformers
    hf_model = transformers.AutoModel.from_pretrained(
        model_dir, trust_remote_code=True).cpu()

    load_from_hf_qwen(tensorrt_llm_qwen,
                             hf_model,
                             dtype='float16')
    del hf_model
