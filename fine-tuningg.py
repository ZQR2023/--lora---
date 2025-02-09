from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 只使用单卡

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 读取数据
df = pd.read_json(r'/root/lanyun-fs/lora_tuning/dataset/dataset/huanhuan.json')
ds = Dataset.from_pandas(df)
print(ds[:3])


# 加载模型
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat", use_fast=False, trust_remote_code=True)

# 处理数据
def process_func(example):
    MAX_LENGTH = 200    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    # 构建输入部分，包含系统提示、用户输入、助手特殊标记
    instruction = tokenizer(f"<|im_start|>system\n现在你要扮演皇帝身边的女人--甄嬛<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  
    # 构建输出部分，包含助手回复
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    # 构建input_ids 将文本转换为模型可以理解的数字序列、attention_mask 需要关注的部分、labels
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为EOS token 是序列结束的标志 
    # 构建labels 将文本转换为模型可以理解的数字序列 ;-100 是一个特殊值，表示在计算损失时忽略这些位置
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]

        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

# 加载模型后直接移动到设备
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-7B-Chat", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()
model = model.to(device)  # 直接移动到默认设备（GPU0）

# 在初始化 Trainer 前应用 LoRA 包装（重要！否则 LoRA 参数不会被训练）
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, #指定任务类型为因果语言模型（Causal Language Model），模型只能看到前面的 token，不能看到后面的
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
model = get_peft_model(model, config)  # 提前应用 LoRA

# 修改 TrainingArguments 添加 DeepSpeed 配置
args = TrainingArguments(
    output_dir="./output/Qwen1.5",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    gradient_checkpointing=True,
    bf16=True,
    remove_unused_columns=False,
    deepspeed="ds_config.json",  # 添加 DeepSpeed 配置文件路径
)
trainer = Trainer(
    model=model,

    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()

model.save_pretrained("./output/Qwen1.5")
tokenizer.save_pretrained("./output/Qwen1.5")

