import os
import sys
import subprocess
import urllib.request
import zipfile
import io
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LLAMA_CPP_DIR = BASE_DIR / "llama.cpp-temp"
OUTPUT_DIR = BASE_DIR / "GGUF model"

print("=" * 60)
print("Qwen3.5-2B -> GGUF \u683c\u5f0f\u8f6c\u6362")
print("=" * 60, flush=True)

OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. \u4e0b\u8f7d llama.cpp
# ============================================================
print()
print("[1/4] \u4e0b\u8f7d llama.cpp...")

if LLAMA_CPP_DIR.exists():
    shutil.rmtree(LLAMA_CPP_DIR)

ZIP_URL = "https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.zip"
print("  \u4e0b\u8f7d zip \u5305...", flush=True)
req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    zip_data = resp.read()

print("  \u89e3\u538b...", flush=True)
with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
    # \u67e5\u627e\u6839\u76ee\u5f55\u540d
    root_dirs = set()
    for name in zf.namelist():
        parts = name.split("/")
        if parts and parts[0]:
            root_dirs.add(parts[0])
    if not root_dirs:
        raise RuntimeError("Cannot find root directory in zip")
    root_name = sorted(root_dirs)[0]

    # \u53ea\u63d0\u53d6\u9700\u8981\u7684\u6587\u4ef6
    needed_prefixes = [
        root_name + "/gguf-py/",
        root_name + "/conversion/",
        root_name + "/convert_hf_to_gguf.py",
    ]
    for name in zf.namelist():
        should_extract = any(name.startswith(p) or name == p for p in needed_prefixes)
        if should_extract:
            rel_path = name[len(root_name)+1:]
            if not rel_path:
                continue
            target = LLAMA_CPP_DIR / rel_path
            if name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

print("  llama.cpp \u4e0b\u8f7d\u5b8c\u6210", flush=True)

# ============================================================
# 2. \u5b89\u88c5\u4f9d\u8d56
# ============================================================
print()
print("[2/4] \u5b89\u88c5\u4f9d\u8d56...")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", str(LLAMA_CPP_DIR / "gguf-py")],
    check=True,
)
print("[OK] \u4f9d\u8d56\u5b89\u88c5\u5b8c\u6210", flush=True)

# ============================================================
# 3. \u8f6c\u6362\u57fa\u7840\u6a21\u578b
# ============================================================
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

print()
print("[3/4] \u8f6c\u6362\u57fa\u7840\u6a21\u578b qwen3.5 2B...")
base_model_dir = BASE_DIR / "qwen3.5 2B"
output_base = OUTPUT_DIR / "qwen3.5-2b.gguf"

if output_base.exists():
    print("  [\u8df3\u8fc7] \u5df2\u5b58\u5728", flush=True)
else:
    subprocess.run(
        [
            sys.executable, str(CONVERT_SCRIPT),
            str(base_model_dir),
            "--outfile", str(output_base),
            "--outtype", "bf16",
            "--no-mtp",
        ],
        check=True,
    )
    print("  [OK] \u57fa\u7840\u6a21\u578b \u8f6c\u6362\u5b8c\u6210", flush=True)

# ============================================================
# 4. \u8f6c\u6362\u5fae\u8c03\u6a21\u578b
# ============================================================
print()
print("[4/4] \u8f6c\u6362\u5fae\u8c03\u6a21\u578b qwen3.5 2B-xjtu-merged...")
merged_model_dir = BASE_DIR / "qwen3.5 2B-xjtu-merged"
output_merged = OUTPUT_DIR / "qwen3.5-2b-xjtu.gguf"

if output_merged.exists():
    print("  [\u8df3\u8fc7] \u5df2\u5b58\u5728", flush=True)
else:
    subprocess.run(
        [
            sys.executable, str(CONVERT_SCRIPT),
            str(merged_model_dir),
            "--outfile", str(output_merged),
            "--outtype", "bf16",
            "--no-mtp",
        ],
        check=True,
    )
    print("  \u5fae\u8c03\u6a21\u578b \u8f6c\u6362\u5b8c\u6210", flush=True)

# ============================================================
# \u5b8c\u6210
# ============================================================
print()
print("=" * 60)
print("\u5168\u90e8\u5b8c\u6210\uff01")
print("\u8f93\u51fa\u76ee\u5f55: {}".format(OUTPUT_DIR), flush=True)
for f in sorted(OUTPUT_DIR.iterdir()):
    if f.suffix == ".gguf":
        size_gb = f.stat().st_size / (1024**3)
        print("  {}  ({:.2f} GB)".format(f.name, size_gb), flush=True)
print("=" * 60, flush=True)

# \u6e05\u7406\u4e34\u65f6\u6587\u4ef6
print("\n\u6e05\u7406\u4e34\u65f6\u6587\u4ef6...", flush=True)
shutil.rmtree(LLAMA_CPP_DIR, ignore_errors=True)
print("[OK]", flush=True)
