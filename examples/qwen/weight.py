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
import configparser
from pathlib import Path

import tensorrt_llm
from safetensors import safe_open
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
                value_dst=tensorrt_llm_model.layers[i].mlp.gate.weight,
                scale_dst=tensorrt_llm_model.layers[i].mlp.gate.per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
        else:
            tensorrt_llm_model.layers[i].mlp.gate.weight.value = v
        
        # mlp fc2
        v = torch_to_numpy(hf_model.transformer.h[idx].mlp.w2.weight.detach().cpu())
        v = split(v, mapping.tp_size, mapping.tp_rank)
        if use_weight_only:
            load_quant_weight(
                src=v,
                value_dst=tensorrt_llm_model.layers[i].mlp.fc.weight,
                scale_dst=tensorrt_llm_model.layers[i].mlp.fc.per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
        else:
            tensorrt_llm_model.layers[i].mlp.fc.weight.value = v
        
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

def load_from_gptq_qwen(tensorrt_llm_model,
                         quant_ckpt_path,
                         mapping=Mapping(),
                         dtype="float16"):
    """
    load from gptq quantianzed model
    """
    tensorrt_llm.logger.info(
        'Loading weights from groupwise GPTQ safetensors...')
    tik = time.time()

    # 导入模型
    if quant_ckpt_path.endswith(".safetensors"):
        groupwise_qweight_safetensors = safe_open(quant_ckpt_path,
                                                  framework="pt",
                                                  device=0)
        model_params = {
            key: groupwise_qweight_safetensors.get_tensor(key)
            for key in groupwise_qweight_safetensors.keys()
        }
    elif quant_ckpt_path.endswith(".pt"):
        model_params = torch.load(quant_ckpt_path,
                                  map_location=torch.device('cpu'))
    else:
        assert False, "Quantized checkpoint format not supported!"


    def unpack_int32_into_int8(w_packed):
        # Unpack inputs packed in int32/float32 into uint4 and store them in int8 format
        w_packed_int4x2 = w_packed.contiguous().view(torch.uint8)
        w_unpacked = torch.zeros(w_packed_int4x2.shape[0],
                                 w_packed_int4x2.shape[1] * 2,
                                 dtype=torch.int8)
        w_unpacked[:, ::2] = w_packed_int4x2 % 16
        w_unpacked[:, 1::2] = w_packed_int4x2 // 16
        return w_unpacked.contiguous()

    def preprocess_groupwise_weight_params(weight_name,
                                           qweight_int32=None,
                                           qzeros_int32=None,
                                           scales_fp16=None):
        if weight_name is not None:
            qweight_int32 = model_params[weight_name].cpu()
            qzeros_int32 = model_params[weight_name[:-7] + 'qzeros'].cpu()
            scales_fp16 = model_params[weight_name[:-7] + 'scales'].cpu()

        UINT4_TO_INT4_FLAG = 1
        GPTQ_FLAG = 1
        packer = torch.ops.fastertransformer.pack_int8_tensor_to_packed_int4
        preprocessor = torch.ops.fastertransformer.preprocess_weights_for_mixed_gemm

        qweight_unpacked_int8 = unpack_int32_into_int8(
            qweight_int32.T).T.contiguous() - 8
        qweight_interleaved = preprocessor(packer(qweight_unpacked_int8),
                                           torch.quint4x2).view(torch.float32)
        # zeros = zeros * scales
        qzeros_unpacked_int32 = unpack_int32_into_int8(qzeros_int32)
        zeros_x_scales_fp16 = (-qzeros_unpacked_int32 + 8 * UINT4_TO_INT4_FLAG -
                               GPTQ_FLAG) * scales_fp16
        zeros_x_scales_fp16 = zeros_x_scales_fp16.half()

        # return processed interleaved weight, original scales and zeros * scales
        return qweight_interleaved.contiguous(), zeros_x_scales_fp16.contiguous(),scales_fp16.contiguous().to(torch_dtype)

    layer_ids = [
        extract_layer_idx(key) for key in groupwise_qweight_safetensors.keys()
    ]
    layer_ids = [
        int(layer_idx) for layer_idx in layer_ids if layer_idx is not None
    ]
    num_hidden_layers = max(layer_ids) + 1
    suffixs = ['qweight', 'qzeros', 'scales']

    layers_per_pipeline_stage = num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))

    torch_dtype = str_dtype_to_torch(dtype)
    # 遍历所有参数
    for k, v in model_params.items():
        # transform to numpy
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(torch_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(torch_dtype).detach().cpu())

        # embedding & lm_head & layer_norm
        if 'transformer.wte.weight' in k:
            if mapping.is_first_pp_rank():
                tensorrt_llm_model.vocab_embedding.weight.value = v
        elif 'transformer.ln_f.weight' in k:
            if mapping.is_last_pp_rank():
                tensorrt_llm_model.ln_f.weight.value = v
        elif 'lm_head.weight' in k:
            if mapping.is_last_pp_rank():
                tensorrt_llm_model.lm_head.weight.value = split(v, mapping.tp_size, mapping.tp_rank)
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(layer_idx)
            if idx not in layers_range:
                continue
            idx = idx - mapping.pp_rank * layers_per_pipeline_stage

            # layernorm
            if 'ln_1.weight' in k:
                tensorrt_llm_model.layers[idx].input_layernorm.weight.value = v
            elif 'ln_2.weight' in k:
                tensorrt_llm_model.layers[idx].post_layernorm.weight.value = v
            elif 'attn.c_attn.bias' in k:
                v = v.reshape(3, -1)
                v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                v = v.flatten()
                tensorrt_llm_model.layers[idx].attention.qkv.bias.value = v
            elif 'attn.c_attn.qweight' in k:
                split_v_suf = []
                for suf in suffixs:
                    v = model_params[k[:-7] + suf].cpu()

                    out_dim = v.shape[1] // 3
                    in_dim = v.shape[0]

                    v = v.reshape(in_dim, out_dim, 3)
                    split_v = v.split(out_dim // mapping.tp_size, dim=1)[mapping.tp_rank]
                    split_v = split_v.reshape(in_dim, -1)
                    split_v_suf.append(split_v)
                th_qweight, th_zero, th_scale = preprocess_groupwise_weight_params(
                    None, split_v_suf[0], split_v_suf[1], split_v_suf[2])
                tensorrt_llm_model.layers[
                    idx].attention.qkv.qweight.value = th_qweight.numpy()
                tensorrt_llm_model.layers[
                    idx].attention.qkv.scale.value = th_scale.numpy()
                tensorrt_llm_model.layers[
                    idx].attention.qkv.zero.value = th_zero.numpy()
            # attention dense layer
            elif 'attn.c_proj.qweight' in k:
                split_v_suf = []
                for suf in suffixs:
                    v = model_params[k[:-7] + suf].cpu()
                    split_v = v.split(v.shape[0] // mapping.tp_size, dim=0)[mapping.tp_rank]
                    split_v_suf.append(split_v)
                th_qweight, th_zero, th_scale = preprocess_groupwise_weight_params(
                    None, split_v_suf[0], split_v_suf[1], split_v_suf[2])
                tensorrt_llm_model.layers[
                    idx].attention.dense.qweight.value = th_qweight.numpy()
                tensorrt_llm_model.layers[
                    idx].attention.dense.scale.value = th_scale.numpy()
                tensorrt_llm_model.layers[
                    idx].attention.dense.zero.value = th_zero.numpy()
            # mlp layer
            elif 'mlp.w1.qweight' in k:
                split_v_suf = []
                for suf in suffixs:
                    v = model_params[k[:-7] + suf].cpu()
                    split_v = v.split(v.shape[1] // mapping.tp_size, dim=1)[mapping.tp_rank]
                    split_v_suf.append(split_v)
                th_qweight, th_zero, th_scale = preprocess_groupwise_weight_params(
                    None, split_v_suf[0], split_v_suf[1], split_v_suf[2])
                tensorrt_llm_model.layers[
                    idx].mlp.gate.qweight.value = th_qweight.numpy()
                tensorrt_llm_model.layers[
                    idx].mlp.gate.scale.value = th_scale.numpy()
                tensorrt_llm_model.layers[
                    idx].mlp.gate.zero.value = th_zero.numpy()
            elif 'mlp.w2.qweight' in k:
                split_v_suf = []
                for suf in suffixs:
                    v = model_params[k[:-7] + suf].cpu()
                    split_v = v.split(v.shape[1] // mapping.tp_size, dim=1)[mapping.tp_rank]
                    split_v_suf.append(split_v)
                th_qweight, th_zero, th_scale = preprocess_groupwise_weight_params(
                    None, split_v_suf[0], split_v_suf[1], split_v_suf[2])
                tensorrt_llm_model.layers[
                    idx].mlp.fc.qweight.value = th_qweight.numpy()
                tensorrt_llm_model.layers[
                    idx].mlp.fc.scale.value = th_scale.numpy()
                tensorrt_llm_model.layers[
                    idx].mlp.fc.zero.value = th_zero.numpy()
            elif 'mlp.c_proj.qweight' in k:
                split_v_suf = []
                for suf in suffixs:
                    v = model_params[k[:-7] + suf].cpu()
                    split_v = v.split(v.shape[0] // mapping.tp_size, dim=0)[mapping.tp_rank]
                    split_v_suf.append(split_v)
                th_qweight, th_zero, th_scale = preprocess_groupwise_weight_params(
                    None, split_v_suf[0], split_v_suf[1], split_v_suf[2])
                tensorrt_llm_model.layers[
                    idx].mlp.proj.qweight.value = th_qweight.numpy()
                tensorrt_llm_model.layers[
                    idx].mlp.proj.scale.value = th_scale.numpy()
                tensorrt_llm_model.layers[
                    idx].mlp.proj.zero.value = th_zero.numpy()

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return


def gen_suffix(rank, use_smooth_quant, quant_per_channel):
    suffix = f"{rank}.bin"
    if use_smooth_quant:
        sq_prefix = "int8."
        if quant_per_channel:
            sq_prefix += "col."
        suffix = sq_prefix + suffix
    return suffix


def parse_ft_config(ini_file):
    gpt_config = configparser.ConfigParser()
    gpt_config.read(ini_file)

    n_embd = gpt_config.getint('qwen', 'hidden_size')
    n_head = gpt_config.getint('qwen', 'num_attention_heads')
    n_layer = gpt_config.getint('qwen', 'num_hidden_layers')
    n_positions = gpt_config.getint('qwen', 'max_position_embeddings')
    vocab_size = gpt_config.getint('qwen', 'vocab_size')
    inter_size = gpt_config.getint('qwen', 'intermediate_size', fallback=None)
    # n_kv_head = gpt_config.getint('qwen', 'num_key_value_heads', fallback=None)

    if inter_size is None:
        inter_size = 4 * n_embd

    return n_embd, n_head, n_layer, n_positions, vocab_size,  inter_size, n_head


def load_from_binary(tensorrt_llm_model,
                     dir_path,
                     mapping=Mapping(),
                     fp16=False,
                     multi_query_mode=False):
    tensorrt_llm.logger.info('Loading weights from FT...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_model, 'quant_mode', QuantMode(0))

    n_embd, n_head, n_layer, n_positions, vocab_size, inter_size, n_kv_head = parse_ft_config(Path(dir_path) / 'config.ini')
    np_dtype = np.float16 if fp16 else np.float32

    def fromfile(dir_path, name, shape=None, dtype=None):
        dtype = np_dtype if dtype is None else dtype
        p = dir_path + '/' + name
        if Path(p).exists():
            t = np.fromfile(p, dtype=dtype)
            if shape is not None:
                t = t.reshape(shape)
            return t
        return None

    def set_smoothquant_scale_factors(module,
                                      pre_scale_weight,
                                      dir_path,
                                      basename,
                                      shape,
                                      per_tok_dyn,
                                      per_channel,
                                      is_qkv=False,
                                      rank=None):
        suffix = "bin"

        # scale，如果是per_channel则为向量
        if per_channel:
            if rank is not None:
                suffix = f"{rank}." + suffix
            suffix = "col." + suffix

        col_shape = shape if (per_channel or is_qkv) else [1, 1]

        if per_tok_dyn:
            # dynamic模式不需要记录activation原来的scale factor
            if pre_scale_weight is not None:
                pre_scale_weight.value = np.array([1.0], dtype=np.float32)
            if is_qkv and not per_channel:
                t = fromfile(dir_path,
                             f"{basename}scale_w_quant_orig.{rank}.{suffix}",
                             col_shape, np.float32)
            else:
                t = fromfile(dir_path, f"{basename}scale_w_quant_orig.{suffix}",
                             col_shape, np.float32)
            module.per_channel_scale.value = t
        else:
            t = fromfile(dir_path, f"{basename}scale_x_orig_quant.bin", [1],
                         np.float32)
            pre_scale_weight.value = t
            if is_qkv:
                t = fromfile(dir_path,
                             f"{basename}scale_y_accum_quant.{rank}.{suffix}",
                             col_shape, np.float32)
            else:
                t = fromfile(dir_path,
                             f"{basename}scale_y_accum_quant.{suffix}",
                             col_shape, np.float32)
            module.per_channel_scale.value = t
            t = fromfile(dir_path, f"{basename}scale_y_quant_orig.bin", [1, 1],
                         np.float32)
            module.act_scale.value = t

    def set_smoother(module, dir_path, base_name, shape, rank):
        suffix = f"{rank}.bin"
        t = fromfile(dir_path, f"{base_name}.smoother.{suffix}", shape,
                     np.float32)
        module.smoother.value = t

    # Determine the quantization mode.
    quant_mode = getattr(tensorrt_llm_model, "quant_mode", QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    # Do we use SmoothQuant?
    use_smooth_quant = quant_mode.has_act_and_weight_quant()
    # Do we use quantization per token?
    quant_per_token_dyn = quant_mode.has_per_token_dynamic_scaling()
    # Do we use quantization per channel?
    quant_per_channel = quant_mode.has_per_channel_scaling()

    # Do we use INT4/INT8 weight-only?
    use_weight_only = quant_mode.is_weight_only()

    # Int8 KV cache
    use_int8_kv_cache = quant_mode.has_int8_kv_cache()

    def sq_trick(x):
        return x.view(np.float32) if use_smooth_quant else x

    # Debug
    suffix = gen_suffix(mapping.tp_rank, use_smooth_quant, quant_per_channel)
    # The type of weights.
    w_type = np_dtype if not use_smooth_quant else np.int8

    # embedding, output layernorm and lm_head
    if mapping.is_first_pp_rank():
        tensorrt_llm_model.vocab_embedding.weight.value = (fromfile(
            dir_path, 'vocab_embedding.weight.bin', [vocab_size, n_embd]))

    if mapping.is_last_pp_rank():
        tensorrt_llm_model.ln_f.weight.value = (fromfile(
            dir_path, 'ln_f.weight.bin'))
    # share input embedding
    lm_head_weight = fromfile(dir_path, 'lm_head.weight.bin',
                              [vocab_size, n_embd])

    if vocab_size % mapping.tp_size != 0:
        # padding
        vocab_size_padded = tensorrt_llm_model.lm_head.out_features * mapping.tp_size
        pad_width = vocab_size_padded - vocab_size
        lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)),
                                'constant',
                                constant_values=0)
    if mapping.is_last_pp_rank():
        tensorrt_llm_model.lm_head.weight.value = np.ascontiguousarray(
            split(lm_head_weight, mapping.tp_size, mapping.tp_rank))

    layers_range = list(
        range(mapping.pp_rank * tensorrt_llm_model.num_layers,
              (mapping.pp_rank + 1) * tensorrt_llm_model.num_layers, 1))

    for i in layers_range:
        n_groups = n_head // n_kv_head
        c_attn_out_dim = (
            3 * n_embd // mapping.tp_size) if not multi_query_mode else (
                n_embd // mapping.tp_size +
                (n_embd // n_head * n_groups) // mapping.tp_size * 2)
        idx = i - mapping.pp_rank * tensorrt_llm_model.num_layers
        tensorrt_llm_model.layers[idx].input_layernorm.weight.value = (fromfile(
            dir_path, 'model.layers.' + str(i) + '.ln_1.weight.bin'))
    
        t = fromfile(dir_path, 'model.layers.' + str(i) + '.attn.c_attn.weight.' + suffix,
            [n_embd, c_attn_out_dim], w_type)
        dst = tensorrt_llm_model.layers[idx].attention.qkv.weight
        if use_smooth_quant:
            dst.value = sq_trick(np.ascontiguousarray(np.transpose(t, [1, 0])))
            set_smoothquant_scale_factors(
                tensorrt_llm_model.layers[idx].attention.qkv,
                tensorrt_llm_model.layers[idx].input_layernorm.scale_to_int,
                dir_path,
                'model.layers.' + str(i) + '.attn.c_attn.',
                [1, c_attn_out_dim],
                quant_per_token_dyn,
                quant_per_channel,
                rank=mapping.tp_rank,
                is_qkv=True)
        else:
            dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))


        t = fromfile(dir_path, 'model.layers.' + str(i) + '.attn.c_attn.bias.' + f'{mapping.rank}.bin', [c_attn_out_dim])
        dst = tensorrt_llm_model.layers[idx].attention.qkv.bias

        dst.value = np.ascontiguousarray(t)

        dst = tensorrt_llm_model.layers[idx].attention.dense.weight
        t = fromfile(
            dir_path,
            'model.layers.' + str(i) + '.attn.c_proj.weight.' + suffix,
            [n_embd // mapping.tp_size, n_embd], w_type)
        if use_smooth_quant:
            dst.value = sq_trick(np.ascontiguousarray(np.transpose(t, [1, 0])))
            dense_scale = getattr(tensorrt_llm_model.layers[idx].attention,
                                  "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                tensorrt_llm_model.layers[idx].attention.dense, dense_scale,
                dir_path, 'model.layers.' + str(i) + '.attn.c_proj.',
                [1, n_embd], quant_per_token_dyn, quant_per_channel)
            set_smoother(tensorrt_llm_model.layers[idx].attention.dense,
                         dir_path,
                         'model.layers.' + str(i) + '.attn.c_proj',
                         [1, n_embd // mapping.tp_size], mapping.tp_rank)
        else:
            dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

        dst = tensorrt_llm_model.layers[idx].post_layernorm.weight
        dst.value = fromfile(
            dir_path, 'model.layers.' + str(i) + '.ln_2.weight.bin')

        t = fromfile(dir_path,
                     'model.layers.' + str(i) + '.mlp.w1.weight.' + suffix,
                     [n_embd, inter_size // mapping.tp_size // 2], w_type)

        if use_smooth_quant:
            tensorrt_llm_model.layers[idx].mlp.gate.weight.value = sq_trick(
                np.ascontiguousarray(np.transpose(t, [1, 0])))
            set_smoothquant_scale_factors(
                tensorrt_llm_model.layers[idx].mlp.gate,
                tensorrt_llm_model.layers[idx].post_layernorm.scale_to_int,
                dir_path,
                'model.layers.' + str(i) + '.mlp.w1.',
                [1, inter_size // mapping.tp_size // 2],
                quant_per_token_dyn,
                quant_per_channel,
                rank=mapping.tp_rank)
        else:
            tensorrt_llm_model.layers[
                idx].mlp.gate.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))

        t = fromfile(dir_path,
                     'model.layers.' + str(i) + '.mlp.w2.weight.' + suffix,
                     [n_embd, inter_size // mapping.tp_size // 2], w_type)
        if use_smooth_quant:
            tensorrt_llm_model.layers[idx].mlp.fc.weight.value = sq_trick(
                np.ascontiguousarray(np.transpose(t, [1, 0])))
            set_smoothquant_scale_factors(
                tensorrt_llm_model.layers[idx].mlp.fc,
                tensorrt_llm_model.layers[idx].post_layernorm.scale_to_int,
                dir_path,
                'model.layers.' + str(i) + '.mlp.w2.',
                [1, inter_size // mapping.tp_size // 2],
                quant_per_token_dyn,
                quant_per_channel,
                rank=mapping.tp_rank)
        else:
            tensorrt_llm_model.layers[
                idx].mlp.gate.weight.value = np.ascontiguousarray(
                    np.transpose(t, [1, 0]))

        t = fromfile(dir_path,
                     'model.layers.' + str(i) + '.mlp.c_proj.weight.' + suffix,
                     [inter_size // mapping.tp_size // 2, n_embd], w_type)
        if use_smooth_quant:
            tensorrt_llm_model.layers[idx].mlp.proj.weight.value = sq_trick(
                np.ascontiguousarray(np.transpose(t, [1, 0])))
            proj_scale = getattr(tensorrt_llm_model.layers[idx].mlp,
                                 "quantization_scaling_factor", None)
            set_smoothquant_scale_factors(
                tensorrt_llm_model.layers[idx].mlp.proj, proj_scale, dir_path,
                'model.layers.' + str(i) + '.mlp.c_proj.', [1, n_embd],
                quant_per_token_dyn, quant_per_channel)
            set_smoother(tensorrt_llm_model.layers[idx].mlp.proj, dir_path,
                         'model.layers.' + str(i) + '.mlp.c_proj',
                         [1, inter_size // mapping.tp_size // 2], mapping.tp_rank)
        else:
            tensorrt_llm_model.layers[idx].mlp.proj.weight.value = (
                np.ascontiguousarray(np.transpose(t, [1, 0])))

        if use_int8_kv_cache:
            t = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.attention.query_key_value.scale_y_quant_orig.bin', [1],
                np.float32)
            tensorrt_llm_model.layers[
                idx].attention.kv_orig_quant_scale.value = 1.0 / t
            tensorrt_llm_model.layers[
                idx].attention.kv_quant_orig_scale.value = t

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')



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
