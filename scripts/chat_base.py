import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRITON_CUDA_DISABLE"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "qwen3.5 2B")

print("=" * 60)
print("加载基座模型 (Qwen3.5-2B)")
print("=" * 60, flush=True)

print("加载 tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print("tokenizer 加载完成", flush=True)

print("加载模型", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print("模型加载完成", flush=True)

print("=" * 60)
print("输入问题开始对话，输入 exit 或 quit 退出")
print("=" * 60, flush=True)

while True:
    try:
        user_input = input("\n>>> ")
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
        break
    if not user_input.strip():
        continue

    messages = [
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": user_input},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True,
    )

    print(response, flush=True)
