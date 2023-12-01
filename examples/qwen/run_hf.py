from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


if __name__ == '__main__':

    # Note: The default behavior now has injection attack prevention off.
    model_dir = './qwen_7b_chat'

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


    # use auto mode, automatically select precision based on the device.
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()

    # Specify hyperparameters for generation
    model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参

    # 第一轮对话 1st dialogue turn
    response, history = model.chat(tokenizer, "你好，请介绍一下清华大学？", history=None)

    print(response)