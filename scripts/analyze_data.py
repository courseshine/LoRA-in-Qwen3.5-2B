import json
import os
import collections
import re

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, 'xjtu data.jsonl')

data = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

print("=" * 60)
print("数据集分析报告")
print("=" * 60, flush=True)

print("\n【基础统计】")
print("  总条目数: {}".format(len(data)))

prompt_lens = [len(d['prompt']) for d in data]
completion_lens = [len(d['completion']) for d in data]

print("  提示词长度:")
print("    平均: {:.0f} 字符, 最短: {}, 最长: {}".format(
    sum(prompt_lens) / len(prompt_lens), min(prompt_lens), max(prompt_lens)))
print("  回答长度:")
print("    平均: {:.0f} 字符, 最短: {}, 最长: {}".format(
    sum(completion_lens) / len(completion_lens), min(completion_lens), max(completion_lens)))

total_tokens_est = sum(prompt_lens + completion_lens) // 2
print("  预估总 token 数: ~{} (按 2 字符/token 估算)".format(total_tokens_est))

print("\n【长度分布 - 提示词】")
bins = [(0, 20), (20, 50), (50, 100), (100, 200)]
bin_counts = [0] * (len(bins) + 1)
for l in prompt_lens:
    placed = False
    for i, (lo, hi) in enumerate(bins):
        if lo <= l < hi:
            bin_counts[i] += 1
            placed = True
            break
    if not placed:
        bin_counts[-1] += 1
for i, (lo, hi) in enumerate(bins):
    pct = bin_counts[i] / len(prompt_lens) * 100
    bar = "#" * int(pct / 2)
    print("  {:>3}-{:<3} 字: {:>3} 条 ({:>4.1f}%) {}".format(lo, hi, bin_counts[i], pct, bar))
pct = bin_counts[-1] / len(prompt_lens) * 100
bar = "#" * int(pct / 2)
print("  >200  字: {:>3} 条 ({:>4.1f}%) {}".format(bin_counts[-1], pct, bar))

print("\n【长度分布 - 回答】")
bins = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 500)]
bin_counts = [0] * (len(bins) + 1)
for l in completion_lens:
    placed = False
    for i, (lo, hi) in enumerate(bins):
        if lo <= l < hi:
            bin_counts[i] += 1
            placed = True
            break
    if not placed:
        bin_counts[-1] += 1
for i, (lo, hi) in enumerate(bins):
    pct = bin_counts[i] / len(completion_lens) * 100
    bar = "#" * int(pct / 2)
    print("  {:>3}-{:<3} 字: {:>3} 条 ({:>4.1f}%) {}".format(lo, hi, bin_counts[i], pct, bar))
pct = bin_counts[-1] / len(completion_lens) * 100
bar = "#" * int(pct / 2)
print("  >500  字: {:>3} 条 ({:>4.1f}%) {}".format(bin_counts[-1], pct, bar))

print("\n【问题类型分布】")
type_patterns = [
    ("什么/什么是", r'(什么|啥|哪些|哪一)'),
    ("为什么/为何", r'(为什么|为何|怎么|如何|怎样)'),
    ("是谁/谁", r'(谁是|是谁|哪位)'),
    ("在哪里", r'(在哪里|在哪|何处)'),
    ("什么时候", r'(什么时候|何时|哪一年|哪年)'),
    ("请介绍/请说明", r'(请介绍|请说明|请讲|请解释)'),
    ("是否/能不能", r'(是否|能不能|有没有|会不会|是不是)'),
]
type_counts = collections.Counter()
other_count = 0
for d in data:
    matched = False
    for label, pattern in type_patterns:
        if re.search(pattern, d['prompt']):
            type_counts[label] += 1
            matched = True
            break
    if not matched:
        other_count += 1
for label, count in type_counts.most_common():
    print("  {}: {} 条 ({:.1f}%)".format(label, count, count / len(data) * 100))
print("  其他: {} 条 ({:.1f}%)".format(other_count, other_count / len(data) * 100))

print("\n【重复检查】")
prompt_set = set()
dup_count = 0
for d in data:
    if d['prompt'] in prompt_set:
        dup_count += 1
    prompt_set.add(d['prompt'])
print("  重复问题: {} 条".format(dup_count))



print("\n" + "=" * 60)
print("分析完成", flush=True)
