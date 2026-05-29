import os
import torch
import safetensors.torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

base_path = os.path.join(base_dir, "qwen3.5 2B")
lora_path = os.path.join(base_dir, "qwen3.5 2B-xjtu")
output_path = os.path.join(base_dir, "qwen3.5 2B-xjtu-merged")

print("加载原模型...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    base_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print("加载 LoRA 权重...", flush=True)
lora_state = safetensors.torch.load_file(
    os.path.join(lora_path, "adapter_model.safetensors")
)

print("构建参数名映射...", flush=True)
model_params = dict(model.named_parameters())

merged = 0
for lora_key, lora_value in lora_state.items():
    if "lora_A" in lora_key:
        continue

    lora_A_key = lora_key.replace("lora_B", "lora_A")
    if lora_A_key not in lora_state:
        continue

    lora_B = lora_state[lora_key]
    lora_A = lora_state[lora_A_key]
    delta = (lora_B @ lora_A).to(torch.bfloat16)

    target = lora_key.replace("base_model.model.", "").replace(".lora_B.weight", ".weight")

    if target in model_params:
        param = model_params[target]
        param.data += delta.to(param.device, dtype=param.dtype)
        merged += 1
    else:
        print(f"  ⚠ 未找到: {target}", flush=True)

print(f"合并了 {merged} 个模块", flush=True)

print("保存...", flush=True)
model.save_pretrained(output_path)
tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
tokenizer.save_pretrained(output_path)
print(f"完成 → {output_path}", flush=True)
