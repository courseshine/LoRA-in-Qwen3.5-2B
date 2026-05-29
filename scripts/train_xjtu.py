import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Qwen3.5 2B - XJTU 知识微调")
print("=" * 60, flush=True)

model_path = os.path.join(base_dir, "qwen3.5 2B")
output_dir = os.path.join(base_dir, "qwen3.5 2B-xjtu")

device = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# 1. 加载 tokenizer
print("加载 tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 加载模型 + 4-bit QLoRA
print("加载模型 (4-bit)...", flush=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)

# 3. 配置 LoRA
print("配置 LoRA...", flush=True)
lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"     可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)", flush=True)


# 4. 加载 数据
print("加载数据...", flush=True)
system_prompt = (
    "你是一位西安交通大学（XJTU）的校史专家与资深教师，"
    "对交大的历史沿革、西迁精神、学科建设、书院制、校园文化、"
    "招生培养等方方面面都了如指掌。"
    "请你以热情专业的口吻，结合具体年代、人物和事件，"
    "为提问者详细讲解，处处体现对交大的深厚感情和"
    "「西迁精神」「为世界之光」等交大文化。"
)

data_file = os.path.join(base_dir, "xjtu data.jsonl")
raw_data = []
with open(data_file, "r", encoding="utf-8") as f:
    for line in f:
        raw_data.append(json.loads(line))
print(f"数据集加载完成: {len(raw_data)} 条", flush=True)

response_template_ids = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)

class XJTUDataset(Dataset):
    def __init__(self, raw_data, tokenizer, system_prompt, max_length=512):
        self.input_ids_list = []
        self.labels_list = []
        for item in raw_data:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["completion"]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
            input_ids = enc["input_ids"][0]
            labels = input_ids.clone()

            seq = input_ids.tolist()
            for j in range(len(seq) - len(response_template_ids), -1, -1):
                if seq[j:j + len(response_template_ids)] == response_template_ids:
                    labels[:j] = -100
                    break

            self.input_ids_list.append(input_ids)
            self.labels_list.append(labels)

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids_list[idx],
            "labels": self.labels_list[idx],
        }

dataset = XJTUDataset(raw_data, tokenizer, system_prompt)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda batch: {
    "input_ids": torch.nn.utils.rnn.pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id),
    "labels": torch.nn.utils.rnn.pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100),
})
print(f"数据集构建完成: {len(dataset)} 条", flush=True)


# 5. 优化器

print("配置优化器...", flush=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# 手动余弦学习率 + warmup
total_gradient_steps = len(dataloader) // 4  # gradient_accumulation_steps = 4
warmup_steps = min(10, total_gradient_steps)

def get_lr(step):
    if step < warmup_steps:
        return 2e-4 * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_gradient_steps - warmup_steps)
    return 2e-4 * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

print(f"     训练参数配置完成", flush=True)
print(f"     批次: 2 x 4 = 有效 8", flush=True)
print(f"     学习率: 2e-4", flush=True)
print(f"     轮数: 1", flush=True)
print(f"     精度: bf16", flush=True)
print(f"     总梯度步数: {total_gradient_steps}", flush=True)


# 6. 训练循环

print("=" * 60, flush=True)
print("开始训练...", flush=True)
print("=" * 60, flush=True)

model.train()
losses = []
global_gradient_step = 0
accumulation_steps = 4

for epoch in range(1):
    optimizer.zero_grad()
    running_loss = 0.0

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss = loss / accumulation_steps
        loss.backward()

        running_loss += loss.item() * accumulation_steps

        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            lr = get_lr(global_gradient_step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()
            optimizer.zero_grad()

            global_gradient_step += 1
            losses.append(running_loss)
            running_loss = 0.0

            if global_gradient_step % 1 == 0:
                print(f"  Step {global_gradient_step}/{total_gradient_steps}, Loss: {losses[-1]:.4f}, LR: {lr:.2e}", flush=True)

   

print(flush=True)
print("=" * 60, flush=True)
print("训练完成，保存模型...", flush=True)
print("=" * 60, flush=True)


# 7. 保存模型

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f" LoRA 权重已保存到: {output_dir}", flush=True)


# 8. 绘制 loss 曲线
print("=" * 60, flush=True)
print("绘制 Loss 曲线...", flush=True)
print("=" * 60, flush=True)

plt.figure(figsize=(10, 5))
plt.plot(
    range(1, len(losses) + 1),
    losses,
    marker="o",
    linestyle="-",
    color="b",
    markersize=2,
)
plt.xlabel("Gradient Step")
plt.ylabel("Loss")
plt.title("XJTU Finetuning Loss Curve (1 epoch)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
if losses:
    print(f"Loss 曲线已保存到: loss_curve.png", flush=True)
    print(f"     最终 loss: {losses[-1]:.4f}", flush=True)

print("=" * 60, flush=True)
print("全部完成！", flush=True)
print("=" * 60, flush=True)
