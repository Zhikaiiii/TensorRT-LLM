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
'''
Convert huggingface GPT model. Use https://huggingface.co/gpt2 as demo.
'''
import argparse
import configparser
import os
from pathlib import Path

import torch
import torch.multiprocessing as multiprocessing
from convert import split_and_save_weight, str_to_np_dtype
from smooth_quant import (capture_activation_range, smooth_gemm,
                         smooth_gemm_fc1_gate)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


def merge_qkv_scales(q_name, hf_model, scales, llama_qkv_para):
    layer_name_q = q_name.replace(".weight", "")
    layer_name_k = layer_name_q.replace("q_proj", "k_proj")
    layer_name_v = layer_name_q.replace("q_proj", "v_proj")
    layer_name_qkv = layer_name_q.replace("q_proj", "qkv_proj")

    q = hf_model.state_dict()[layer_name_q + ".weight"]
    k = hf_model.state_dict()[layer_name_k + ".weight"]
    v = hf_model.state_dict()[layer_name_v + ".weight"]

    weight = torch.cat([q, k, v], dim=0)

    scales[layer_name_qkv]["x"] = scales[layer_name_q]["x"]
    scales[layer_name_qkv]["w"] = weight.abs().max(dim=1)[0]
    print(scales[layer_name_q])
    scales[layer_name_qkv]["y"] = torch.cat([
        scales[layer_name_q]["y"], scales[layer_name_k]["y"],
        scales[layer_name_v]["y"]
    ],
                                            dim=0)

    llama_qkv_para[layer_name_qkv] = weight.transpose(0, 1)


@torch.no_grad()
def smooth_qwen_model(model, scales, alpha, qwen_smoother):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        if not str(type(module)).endswith("QWenBlock'>"):
            continue
        # qkv_proj
        layer_name = name + ".attn.c_attn"
        # smooth：计算scale后把权重进行平滑
        smoother = smooth_gemm(module.attn.c_attn.weight,
                               scales[layer_name]["x"], module.ln_1.weight, None, alpha)
        # 记录激活值量化的最大值，用于量化
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.attn.c_attn.weight.abs().max(
            dim=1)[0]


        # =================================================================
        layer_name = name + ".attn.c_proj"
        # smooth：计算scale后把权重进行平滑
        smoother = smooth_gemm(module.attn.c_proj.weight,
                               scales[layer_name]["x"], None, None, alpha)
        # 记录smooth系数
        qwen_smoother[layer_name] = smoother.float()

        # 记录激活值量化的最大值，用于量化
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.attn.c_proj.weight.abs().max(
            dim=1)[0]

        # ==================================================================
        w1_layer_name = name + ".mlp.w1"
        w2_layer_name = name + ".mlp.w2"

        smoother = smooth_gemm_fc1_gate(module.mlp.w1.weight,
                                        module.mlp.w2.weight,
                                        scales[w1_layer_name]["x"],
                                        module.ln_2.weight,
                                        None, alpha)

        scales[w1_layer_name]["x"] = scales[w1_layer_name]["x"] / smoother
        scales[w1_layer_name]["w"] = module.mlp.w1.weight.abs().max(
            dim=1)[0]

        scales[w2_layer_name]["x"] = scales[w2_layer_name]["x"] / smoother
        scales[w2_layer_name]["w"] = module.mlp.w2.weight.abs().max(
            dim=1)[0]

        # ==================================================================
        layer_name = name + ".mlp.c_proj"
        smoother = smooth_gemm(module.mlp.c_proj.weight,
                               scales[layer_name]["x"], None, None, alpha)
        qwen_smoother[layer_name] = smoother.float()
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.c_proj.weight.abs().max(
            dim=1)[0]


def qwen_to_ft_name(orig_name):
    global_ft_weights = {
        "transformer.wte.weight": 'vocab_embedding.weight',
        "transformer.ln_f.weight": 'ln_f.weight',
        "lm_head.weight": 'lm_head.weight',
    }

    if orig_name in global_ft_weights:
        return global_ft_weights[orig_name]

    _, _, layer_id, *weight_name = orig_name.split(".")

    layer_id = int(layer_id)
    weight_name = ".".join(weight_name)


    # per_layer_weights = {
    #     "ln_1.weight": "ln_1.weight",
    #     "ln_2.weight": "ln_2.weight",
    #     "attn.c_attn.weight": "attention.dense.weight",
    #     "mlp.gate_proj.weight": "mlp.fc.weight",
    #     "mlp.down_proj.weight": "mlp.proj.weight",
    #     "mlp.up_proj.weight": "mlp.gate.weight",
    #     "post_attention_layernorm.weight": "post_layernorm.weight",
    # }

    return f"layers.{layer_id}.{weight_name}"




# LLaMA uses nn.Linear for these following ops whose weight matrix is transposed compared to gpt2.
# In order to use the preprocess codes of gpt2, we transpose them firstly.
def transpose_weights(hf_name, param):
    weight_to_transpose = ["c_proj", "w1", "w2", "c_attn"]
    if any([k in hf_name for k in weight_to_transpose]):
        if len(param.shape) == 2:
            param = param.transpose(0, 1)
    return param


def hf_qwen_converter(args):
    infer_tp = args.tensor_parallelism
    saved_dir = Path(args.out_dir) / f"{infer_tp}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(args.in_file, device_map="auto", trust_remote_code=True, fp16=True)

    act_range = {}
    # smoother for inputs of self_attn.o_proj and mlp.down_proj
    qwen_smoother = {}

    # 做smoothquant
    if args.smoothquant is not None or args.calibrate_kv_cache:
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false")
        alpha = args.smoothquant
        act_range = capture_activation_range(
            model,
            AutoTokenizer.from_pretrained(args.in_file, padding_side='left', trust_remote_code=True))
        if args.smoothquant is not None:
            smooth_qwen_model(model, act_range, alpha, qwen_smoother)

    config = configparser.ConfigParser()
    config["qwen"] = {}
    for key in vars(args):
        config["qwen"][key] = f"{vars(args)[key]}"
    for k, v in vars(model.config).items():
        config["qwen"][k] = f"{v}"
    config["qwen"]["weight_data_type"] = args.storage_type
    with open(saved_dir / "config.ini", 'w') as configfile:
        config.write(configfile)

    storage_type = str_to_np_dtype(args.storage_type)

    global_ft_weights = [
        'vocab_embedding.weight', 'ln_f.weight', 'lm_head.weight'
    ]

    int8_outputs = None
    if args.calibrate_kv_cache:
        int8_outputs = "kv_cache_only"
    if args.smoothquant is not None:
        int8_outputs = "all"

    starmap_args = []
    for name, param in model.named_parameters():
        if "weight" not in name and "bias" not in name:
            continue
        ft_name = qwen_to_ft_name(name)
        # 存储smoother
        if name.replace(".weight", "") in qwen_smoother.keys():
            smoother = qwen_smoother[name.replace(".weight", "")]
            smoother = smoother.detach().cpu().numpy()
            starmap_args.append(
                (0, saved_dir, infer_tp,
                 f"{ft_name}.smoother".replace(".weight", ""), smoother, None, {
                     "int8_outputs": int8_outputs,
                     "local_dim": None,
                 }))
        # 存储原参数
        param = transpose_weights(name, param)
        param = param.detach().cpu().numpy().astype(storage_type)

        if ft_name in global_ft_weights:
            param.tofile(saved_dir / f"{ft_name}.bin")
        else:
            starmap_args.append((0, saved_dir, infer_tp, ft_name, param,
                                 act_range.get(name.replace(".weight", "")), {
                                     "int8_outputs": int8_outputs,
                                     "local_dim": None,
                                 }))

    starmap_args = tqdm(starmap_args, desc="saving weights")
    if args.processes > 1:
        with multiprocessing.Pool(args.processes) as pool:
            pool.starmap(split_and_save_weight, starmap_args)
    else:
        # simpler for debug situations
        for starmap_arg in starmap_args:
            split_and_save_weight(*starmap_arg)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--out-dir',
                        '-o',
                        type=str,
                        help='file name of output directory',
                        required=True)
    parser.add_argument('--in-file',
                        '-i',
                        type=str,
                        help='file name of input checkpoint file',
                        required=True)
    parser.add_argument('--tensor-parallelism',
                        '-tp',
                        type=int,
                        help='Requested tensor parallelism for inference',
                        default=1)
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        help="How many processes to spawn for conversion (default: 4)",
        default=4)
    parser.add_argument(
        "--calibrate-kv-cache",
        "-kv",
        action="store_true",
        help=
        "Generate scaling factors for KV cache. Used for storing KV cache in int8."
    )
    parser.add_argument(
        "--smoothquant",
        "-sq",
        type=float,
        default=0.5,
        help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
        " to Smoothquant the model, and output int8 weights."
        " A good first try is 0.5. Must be in [0, 1]")
    parser.add_argument("--storage-type",
                        "-t",
                        type=str,
                        default="fp32",
                        choices=["fp32", "fp16"])
    parser.add_argument("--multi-query-mode",
                        action="store_true",
                        help="Use multi-query-attention.")

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))
    print("========================================")

    assert (args.calibrate_kv_cache or args.smoothquant), \
        "Either INT8 kv cache or SmoothQuant must be enabled for this script. Otherwise you can directly build engines from HuggingFace checkpoints, no need to do this FT-format conversion. "

    hf_qwen_converter(args)
