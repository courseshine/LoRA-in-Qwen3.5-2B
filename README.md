# Qwen3.5-2B-XJTU

用 QLoRA 在西安交通大学相关数据上微调 Qwen3.5-2B，得到一个懂交大的 AI 助手。

## 目录结构

```
.
├── qwen3.5 2B/                  # 基座模型（Qwen3.5-2B 原始权重）
├── qwen3.5 2B-xjtu/             # LoRA 适配器（训练产出）
├── qwen3.5 2B-xjtu-merged/      # LoRA 合并后的完整模型
│
├── GGUF model/                  # GGUF 格式模型（可直接给 Ollama 用）
│   ├── blobs/                   # Ollama 的内容寻址存储
│   └── manifests/               # Ollama 模型清单
│
├── scripts/
│   ├── train_xjtu.py            # QLoRA 微调训练
│   ├── merge_lora.py            # LoRA 权重合并
│   ├── chat_base.py             # 基座模型推理
│   ├── chat_xjtu.py             # 微调模型推理
│   ├── analyze_data.py          # 数据集分析
│   └── convert_to_gguf.py       # safetensors → GGUF 转换
│
├── xjtu data.jsonl              # 训练数据集
├── data_gen_prompt.txt          # 数据生成提示词模版
├── loss_curve.png               # 训练 Loss 曲线
├── requirements.txt             # Python 依赖
└── README.md
```

## 环境要求

### 硬件

- 训练：建议 6GB+ 显存的 NVIDIA GPU（我的4050显卡，训练耗时65分钟）
- 推理：没有显存需求

### 依赖

```bash
pip install -r requirements.txt
```

## 使用流程

### 1. 训练

确保 `qwen3.5 2B/` 和 `xjtu data.jsonl` 都在项目根目录下，然后：

```bash
python scripts/train_xjtu.py
```

脚本会自动完成：加载 tokenizer → 4-bit 量化加载模型 → 注入 LoRA → 训练 1 轮 → 保存适配器 → 绘制 Loss 曲线。

关键参数：lr=2e-4, LoRA rank=8, 有效 batch size=8, bf16 精度, 最大长度 512 tokens。

训练完成后，LoRA 权重保存在 `qwen3.5 2B-xjtu/` 目录。

### 2. 合并 LoRA 权重

```bash
python scripts/merge_lora.py
```

把 LoRA 适配器合并回基座模型，得到完整的 `qwen3.5 2B-xjtu-merged/`，可以直接用 transformers 加载推理。

### 3. 推理

```bash
# 基座模型对话
python scripts/chat_base.py

# 微调模型对话
python scripts/chat_xjtu.py
```

两个脚本都是交互式对话，输入 `exit` 或 `quit` 退出。

### 4. 数据分析

```bash
python scripts/analyze_data.py
```

统计数据集中回答的开头风格分布，检查数据多样性。

### 5. 转换为 GGUF 格式（可选）

如果你想把模型用在 Ollama、LM Studio 这类工具上，需要转成 GGUF 格式：

```bash
python scripts/convert_to_gguf.py
```

脚本会自动下载 llama.cpp 的转换工具，把基座和微调两个模型都转成 GGUF，输出到 `GGUF model/` 目录。

### 6. 导入 Ollama

转换完成后，导入Ollama还需要一步转换，用 Modelfile 导入 Ollama：

```bash
# 微调模型
ollama create qwen3.5-2b-xjtu -f Modelfile

# 基座模型
ollama create qwen3.5-2b -f Modelfile-base

# 运行
ollama run qwen3.5-2b-xjtu
```

## 数据集

`xjtu data.jsonl` 包含约 1000 条问答对，覆盖西安交通大学十大主题：

- 历史沿革、西迁精神、校园地理、书院制、学科建设
- 知名人物、招生培养、校园文化、附属机构、报考指南

每条数据格式：

```json
{"prompt": "用户问题", "completion": "助手的详细回答"}
```

数据使用 ChatML 格式模板，训练时只对 assistant 部分计算损失。

## 事后思考

1. **数据量的问题**。对于 2B 模型来说，1000 条数据有点多了，当前参数条件下 800 条可能比较合适。更多的数据不一定带来更好的效果，反而可能让模型过拟合。

2. **为什么转 GGUF**。两个 chat 脚本本身就能对话，但在 Python 里跑比较慢，因为每次推理都要经过 Python→C++ 的上下文切换。转成 GGUF 导入 Ollama 之后，推理在 C++ 原生层执行，速度快很多，而且用起来也方便。

3. **为什么选 2B 模型**。2B 参数训练起来快，4050 显卡一个小时左右就跑完了。如果换用 4B 模型的 Q4 量化版本，应该也能训，而且模型能力更强，数据也可以多准备一些。

4. **换个领域也能做**。修改 `data_gen_prompt.txt` 里的提示词模板，找一个强大的 AI 帮忙生成数据集，就可以把模型训练成其他领域的chatbot。也可以考虑自制数据集，因为只有1000条数据，自己搜一搜当然也可以左大培。

5. **微调的实用性问题**。这种微调感觉实用性有限。如果想达到类似的目的，更高效的做法是用一个规模较大的模型（比如同系列的Qwen3.5 27B）建一个知识库（RAG），配合一些提示词工程，效果可能更好，而且不需要重新训练模型。

## 附录：Qwen3.5-2B 模型结构详解

### 一、整体架构

Qwen3.5-2B 是一个混合架构的语言模型，结合了 Mamba（线性注意力）和标准 Transformer 注意力，同时原生支持图像和视频多模态输入。

```
┌──────────────────────────────────────────────────┐
│            Qwen3_5ForConditionalGeneration         │
│                                                    │
│  ┌──────────────────────────────────────┐         │
│  │  文本模型（Language Model）           │         │
│  │  ├─ Embedding (248,320 × 2048)       │         │
│  │  ├─ 24 层混合注意力                  │         │
│  │  │   ├─ 18 层 Mamba（线性注意力）     │         │
│  │  │   └─ 6 层 标准注意力（每4层一次）  │         │
│  │  ├─ MTP 头（多 token 预测，训练用）   │         │
│  │  └─ LM Head（与 Embedding 共享权重）  │         │
│  ├──────────────────────────────────────┤         │
│  │  视觉编码器（Vision Encoder）         │         │
│  │  ├─ 24 层 ViT（patch_size=16）       │         │
│  │  └─ 投影层 1024 → 2048              │         │
│  └──────────────────────────────────────┘         │
└──────────────────────────────────────────────────┘
```

### 二、核心参数

| 参数 | 值 | 说明 |
|---|---|---|
| `hidden_size` | 2048 | 隐藏层维度 |
| `intermediate_size` | 6144 | MLP 中间维度（3× expansion） |
| `num_hidden_layers` | 24 | 总层数 |
| `num_attention_heads` | 8 | 注意力头数 |
| `num_key_value_heads` | 2 | KV 头数（GQA，4:1 分组） |
| `head_dim` | 256 | 每个注意力头的维度 |
| `vocab_size` | 248,320 | 词表大小 |
| `max_position_embeddings` | 262,144 | 最大上下文长度 |
| `tie_word_embeddings` | true | 输入输出 embedding 共享权重 |
| `hidden_act` | silu | MLP 激活函数（SwiGLU） |
| `rope_theta` | 10,000,000 | RoPE 基频 |
| `partial_rotary_factor` | 0.25 | 部分 RoPE 比例 |

### 三、参数总量

**总参数量：2,015,246,336（约 20.15 亿）**

| 组件 | 参数量 | 占比 |
|---|---|---|
| Embedding（与 LM Head 共享） | 508,559,360 | 25.2% |
| MLP（24 层 SwiGLU，gate/up/down） | 905,969,664 | 45.0% |
| 标准 Attention 层（6 层，q/k/v/o） | 62,914,560 | 3.1% |
| Mamba 层（18 层，SSM 投影） | 226,750,464 | 11.3% |
| 视觉编码器（24 层 ViT） | ~304,000,000 | 15.1% |
| RMSNorm 归一化层 | ~98,304 | <0.01% |

### 四、混合注意力机制

24 层的排列模式：

```
层 0:  linear_attention (Mamba)
层 1:  linear_attention (Mamba)
层 2:  linear_attention (Mamba)
层 3:  full_attention   (标准 Transformer) ← 每4层一次
层 4:  linear_attention (Mamba)
...重复...
层 23: full_attention   (标准 Transformer)
```

- **Mamba（线性注意力）**：计算复杂度 O(n)，推理快、显存低
- **标准注意力**：计算复杂度 O(n²)，精确信息建模
- 混合策略：18 层 Mamba 保效率 + 6 层标准注意力保精度

### 五、LoRA 更新的层

本项目中 LoRA 对 **全部 24 层的 7 个线性投影模块** 注入低秩适配器：

```
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```

#### 各模块形状

| 模块 | 原始权重形状 | LoRA 参数量（每层） | 用途 |
|---|---|---|---|
| `q_proj` | 2048 × 2048 | 32,768 | 生成 Query |
| `k_proj` | 2048 × 512 | 20,480 | 生成 Key（GQA） |
| `v_proj` | 2048 × 512 | 20,480 | 生成 Value（GQA） |
| `o_proj` | 2048 × 2048 | 32,768 | 合并多头输出 |
| `gate_proj` | 2048 × 6144 | 65,536 | SwiGLU 门控 |
| `up_proj` | 2048 × 6144 | 65,536 | SwiGLU 上投影 |
| `down_proj` | 6144 × 2048 | 65,536 | SwiGLU 下投影 |

#### LoRA 配置

| 参数 | 值 |
|---|---|
| `r`（秩） | 8 |
| `lora_alpha` | 8 |
| `lora_dropout` | 0 |
| `bias` | none |

#### 可训练参数量

```
单层 7 个模块: 303,104 参数
24 层总计:     7,274,496 参数（约 7.27M）
占总参数量:    0.36%
```

#### 未更新的部分

- Embedding 层和 LM Head
- RMSNorm 归一化层
- Mamba 的 SSM 内部参数（A_log, D, dt_bias 等）
- 视觉编码器所有参数


