from tensorrt_llm.layers.attention import AttentionMaskType, PositionEmbeddingType
from tensorrt_llm.quantization.layers import SmoothQuantRmsNorm, SmoothQuantAttention, SmoothQuantGatedMLP
import copy
import functools
from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D
from tensorrt_llm.quantization.mode import QuantMode
from utils.utils import make_context
import numpy as np

# utils for smooth quant
@torch.no_grad()
def apply_smoothing(scales,
                    gemm_weights,
                    layernorm_weights=None,
                    layernorm_bias=None,
                    dtype=torch.float32,
                    layernorm_1p=False):
    if not isinstance(gemm_weights, list):
        gemm_weights = [gemm_weights]

    if layernorm_weights is not None:
        assert layernorm_weights.numel() == scales.numel()
        layernorm_weights.div_(scales).to(dtype)
    if layernorm_bias is not None:
        assert layernorm_bias.numel() == scales.numel()
        layernorm_bias.div_(scales).to(dtype)
    if layernorm_1p:
        layernorm_weights += (1 / scales) - 1

    for gemm in gemm_weights:
        gemm.mul_(scales.view(1, -1)).to(dtype)


@torch.no_grad()
def smooth_gemm(gemm_weights,
                act_scales,
                layernorm_weights=None,
                layernorm_bias=None,
                alpha=0.5,
                weight_scales=None):
    if not isinstance(gemm_weights, list):
        gemm_weights = [gemm_weights]
    orig_dtype = gemm_weights[0].dtype

    for gemm in gemm_weights:
        # gemm_weights are expected to be transposed
        assert gemm.shape[1] == act_scales.numel()

    if weight_scales is None:
        weight_scales = torch.cat(
            [gemm.abs().max(dim=0, keepdim=True)[0] for gemm in gemm_weights],
            dim=0)
        weight_scales = weight_scales.max(dim=0)[0]
    weight_scales.to(float).clamp(min=1e-5)
    scales = (act_scales.to(gemm_weights[0].device).to(float).pow(alpha) /
              weight_scales.pow(1 - alpha)).clamp(min=1e-5)

    apply_smoothing(scales, gemm_weights, layernorm_weights, layernorm_bias,
                    orig_dtype)

    return scales


@torch.no_grad()
def smooth_gemm_fc1_gate(fc1_weights,
                         gate_weights,
                         act_scales,
                         layernorm_weights=None,
                         layernorm_bias=None,
                         alpha=0.5,
                         weight_scales=None):
    gemm_weights = []
    if not isinstance(fc1_weights, list):
        fc1_weights = [fc1_weights]
    if not isinstance(gate_weights, list):
        gate_weights = [gate_weights]

    for i in range(len(fc1_weights)):
        gemm_weight = torch.cat([fc1_weights[i], gate_weights[i]], dim=0)
        gemm_weights.append(gemm_weight)

    orig_dtype = gemm_weights[0].dtype

    for gemm in gemm_weights:
        # gemm_weights are expected to be transposed
        assert gemm.shape[1] == act_scales.numel()

    if weight_scales is None:
        weight_scales = torch.cat(
            [gemm.abs().max(dim=0, keepdim=True)[0] for gemm in gemm_weights],
            dim=0)
        weight_scales = weight_scales.max(dim=0)[0]
    weight_scales.to(float).clamp(min=1e-5)
    scales = (act_scales.to(gemm_weights[0].device).to(float).pow(alpha) /
              weight_scales.pow(1 - alpha)).clamp(min=1e-5)

    apply_smoothing(scales, fc1_weights + gate_weights, layernorm_weights,
                    layernorm_bias, orig_dtype)

    return scales


@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (act_scales.pow(alpha) /
              weight_scales.pow(1 - alpha)).clamp(min=1e-5).to(device).to(dtype)

    if ln is not None:
        ln.weight.div_(scales)
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
    return scales


@torch.no_grad()
def capture_activation_range(model, tokenizer, num_samples=512, seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = defaultdict(lambda: {"x": None, "y": None, "w": None})

    tokenizer.pad_token = tokenizer.im_end_id


    def stat_tensor(name, tensor, act_scales, key):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float()

        if act_scales[name][key] is None:
            act_scales[name][key] = comming_max
        else:
            act_scales[name][key] = torch.max(act_scales[name][key],
                                              comming_max)

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        
        # 输入
        stat_tensor(name, x, act_scales, "x")
        # 输出
        stat_tensor(name, y, act_scales, "y")

        # 权重
        if act_scales[name]["w"] is None:
            act_scales[name]["w"] = m.weight.abs().clip(1e-8,
                                                        None).max(dim=1)[0]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, Conv1D):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    from datasets import load_dataset
    dataset_cnn = load_dataset("ccdv/cnn_dailymail", '3.0.0')

    system_prompt = "You are a useful assistant, please directly output the corresponding summary according to the article entered by the user."

    for i in tqdm(range(num_samples), desc="calibrating model"):
        datapoint = dataset_cnn['train'][i]
        line = copy.copy(datapoint['article'])
        line = line + ' TL;DR: '
        line = line.strip()
        line = line.replace(" n't", "n't")
        
        _, input_id_list = make_context(
            tokenizer=tokenizer,
            query=line,
            history=[],
            system=system_prompt,
            max_input_length=seq_len
        )
        line_encoded = torch.from_numpy(
            np.array(input_id_list, dtype=np.int32)
        ).type(torch.int32).unsqueeze(0)
        line_encoded = line_encoded.to(device)
        model(line_encoded)

    for h in hooks:
        h.remove()

    return act_scales

# transform trt model to smooth quant model
def smooth_quant_qwen(model, quant_mode):
    assert quant_mode.has_act_and_weight_quant()
    for layer in model.layers:
        assert hasattr(layer,
                       "input_layernorm"), "The layer has no input_layernorm"
        layer.input_layernorm = SmoothQuantRmsNorm(
            normalized_shape=layer.hidden_size,
            dtype=layer.dtype,
            quant_mode=quant_mode)
        assert hasattr(layer, "attention"), "The layer has no attention"
        layer.attention = SmoothQuantAttention(
            layer.hidden_size,
            num_attention_heads=layer.num_attention_heads,
            num_kv_heads=layer.num_kv_heads,
            max_position_embeddings=layer.max_position_embeddings,
            num_layers=model.num_layers,
            dtype=layer.dtype,
            attention_mask_type=layer.attention_mask_type,
            position_embedding_type=layer.position_embedding_type,
            tp_group=layer.tp_group,
            tp_size=layer.tp_size,
            quant_mode=quant_mode,
            bias=True)
        # a hack way
        layer.attention.dense.bias = None
        layer.attention.dense._parameters.pop('bias')
        assert hasattr(layer, "mlp"), "The layer has no mlp"
        layer.mlp = SmoothQuantGatedMLP(hidden_size=model.hidden_size,
                                        ffn_hidden_size=layer.mlp_hidden_size // 2,
                                        hidden_act=layer.hidden_act,
                                        dtype=layer.dtype,
                                        tp_group=layer.tp_group,
                                        tp_size=layer.tp_size,
                                        quant_mode=quant_mode,
                                        bias=False)
        assert hasattr(
            layer,
            "post_layernorm"), "The layer has no post_rmspost_layernormnorm"
        layer.post_layernorm = SmoothQuantRmsNorm(
            normalized_shape=layer.hidden_size,
            dtype=layer.dtype,
            quant_mode=quant_mode)

    setattr(model, 'quant_mode', quant_mode)
    return model