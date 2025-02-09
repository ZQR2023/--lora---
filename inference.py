import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed

# 初始化 DeepSpeed 推理配置
model_name = "./output/Qwen1.5"
deepspeed_config = {
    "tensor_parallel": {"tp_size": 1},  # 单卡推理保持 tp_size=1
    "dtype": "bfloat16",                # 使用 bfloat16 加速推理
    "replace_with_kernel_inject": True,  # 启用内核注入优化
}

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 使用 DeepSpeed 初始化推理引擎
model = deepspeed.init_inference(
    model=model,
    config=deepspeed_config,
    checkpoint=None  # 如果使用分片检查点可指定路径
)

# 定义生成配置
generation_config = {
    "max_new_tokens": 50,
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.7,
    "pad_token_id": tokenizer.pad_token_id
}

# 定义推理函数
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.module.device)
    outputs = model.generate(**inputs, **generation_config)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 示例推理
prompt = "<|im_start|>system\n现在你要扮演皇帝身边的女人--甄嬛<|im_end|>\n<|im_start|>user\n你父亲是谁？<|im_end|>\n<|im_start|>assistant\n"
response = generate_response(prompt)
print("Generated Response:", response) 