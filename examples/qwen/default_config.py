import os


class DefaultConfig:
    now_dir = os.path.dirname(os.path.abspath(__file__))
    hf_model_dir = os.path.join(now_dir, "qwen_7b_chat")
    tokenizer_dir = os.path.join(now_dir, "qwen_7b_chat")
    int4_gptq_model_dir = os.path.join(now_dir, "qwen_7b_chat_int4")
    ft_dir_path = os.path.join(now_dir, "c-model", "qwen_7b_chat")
    engine_dir=os.path.join(now_dir, "trt_engines", "fp16", "1-gpu")

    # Maximum batch size for HF backend.
    hf_max_batch_size = 1

    # Maximum batch size for TRT-LLM backend.
    trt_max_batch_size = 4

    # choice the model format, base or chat
    #  choices=["chatml", "raw"],
    chat_format = "chatml"

    # Maximum input length.
    max_input_len = 256

    # Maximum number of generate new tokens.
    max_new_tokens = 512

    # Maximum sequence length.
    # for Qwen-7B-Chat V1.0, the seq_length is 2048
    # for Qwen-7B-Chat V1.1, the seq_length is 8192
    # for Qwen-14B-Chat, the seq_length is 2048
    seq_length = 2048

    # Top p for sampling.
    top_p = 0.8


    # Top k for sampling.
    top_k = 0

    # Temperature for sampling.
    temperature = 1.0


default_config = DefaultConfig()