# -*- coding: utf-8 -*-
"""
H8 · MusicGen-small 文本到音乐推理
针对 4GB 显存（实际可用约 2.8GB）保守配置：fp16 + 短时长 + 显存监控

用法：
    set HF_HOME=D:\\hf_cache
    python musicgen_infer.py

产出：
    outputs/*.wav              生成的音频
    h8_benchmark.md            实测记录表（prompt / 时长 / 耗时 / 显存）
"""
import os
import sys
import time

# 强制把 HuggingFace 缓存放到 D 盘（C 盘只剩 18GB）
os.environ.setdefault("HF_HOME", r"D:\hf_cache")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
import soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_ID = "facebook/musicgen-small"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTDIR, exist_ok=True)

# 4GB 显存下的保守配置
MAX_NEW_TOKENS = 200      # 约 4 秒（50 tokens/秒）
GUIDANCE_SCALE = 3.0      # 默认 3.0，越高越贴合 prompt 但更耗算力

PROMPTS = [
    "lo-fi hip hop with warm piano, soft vinyl crackle, mellow drums",
    "cinematic orchestral strings with slow build up and deep bass",
    "gentle acoustic guitar fingerpicking, calm and intimate",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print("=" * 64)
print(f"device        : {DEVICE}")
if DEVICE == "cuda":
    print(f"gpu           : {torch.cuda.get_device_name(0)}")
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"vram free     : {free_b/1024**3:.2f} GB / {total_b/1024**3:.2f} GB")
print(f"dtype         : {DTYPE}")
print(f"max_new_tokens: {MAX_NEW_TOKENS} (~{MAX_NEW_TOKENS/50:.1f} s)")
print(f"HF_HOME       : {os.environ.get('HF_HOME')}")
print("=" * 64)

# ---------------- 加载模型 ----------------
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE)
model.eval()
load_s = time.time() - t0

n_params = sum(p.numel() for p in model.parameters())
print(f"\n[load] 耗时 {load_s:.1f}s | 参数量 {n_params/1e6:.1f}M ({n_params:,})")

if DEVICE == "cuda":
    print(f"[load] 显存占用 {torch.cuda.memory_allocated()/1024**3:.2f} GB "
          f"| 峰值 {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

SAMPLE_RATE = model.config.audio_encoder.sampling_rate
print(f"[load] 输出采样率 {SAMPLE_RATE} Hz")

# ---------------- 逐条生成 ----------------
rows = []
for i, prompt in enumerate(PROMPTS):
    torch.cuda.reset_peak_memory_stats() if DEVICE == "cuda" else None

    inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(DEVICE)
    if DEVICE == "cuda" and "input_values" in inputs:
        inputs["input_values"] = inputs["input_values"].to(DTYPE)

    t1 = time.time()
    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            guidance_scale=GUIDANCE_SCALE,
        )
    gen_s = time.time() - t1

    audio = audio_values[0, 0].float().cpu().numpy()
    duration = len(audio) / SAMPLE_RATE
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))

    path = os.path.join(OUTDIR, f"h8_prompt{i+1}.wav")
    sf.write(path, audio, SAMPLE_RATE)

    vram = torch.cuda.max_memory_allocated() / 1024**3 if DEVICE == "cuda" else 0.0
    rtf = gen_s / duration if duration > 0 else 0

    rows.append({
        "id": i + 1,
        "prompt": prompt,
        "tokens": MAX_NEW_TOKENS,
        "duration": duration,
        "gen_s": gen_s,
        "rtf": rtf,
        "vram": vram,
        "peak": peak,
        "rms": rms,
        "file": os.path.basename(path),
    })

    print(f"\n[{i+1}/{len(PROMPTS)}] {prompt}")
    print(f"  时长 {duration:.2f}s | 生成耗时 {gen_s:.1f}s | RTF {rtf:.2f}x | "
          f"显存峰值 {vram:.2f} GB | 峰值 {peak:.3f} | RMS {rms:.4f}")
    print(f"  -> {path}")

# ---------------- 写实测记录表 ----------------
md = ["# H8 实测记录表 · MusicGen-small", "",
      f"- 设备：{torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}（显存 4 GB，实际可用约 2.8 GB）",
      f"- dtype：{DTYPE}",
      f"- 模型参数量：{n_params/1e6:.1f}M",
      f"- 模型加载耗时：{load_s:.1f}s",
      f"- 输出采样率：{SAMPLE_RATE} Hz",
      f"- 生成配置：max_new_tokens={MAX_NEW_TOKENS}（约 4 秒），guidance_scale={GUIDANCE_SCALE}",
      "",
      "| # | Prompt | 时长(s) | 生成耗时(s) | RTF | 显存峰值(GB) | 峰值 | RMS | 文件 |",
      "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['id']} | {r['prompt']} | {r['duration']:.2f} | {r['gen_s']:.1f} | "
              f"{r['rtf']:.2f} | {r['vram']:.2f} | {r['peak']:.3f} | {r['rms']:.4f} | {r['file']} |")
md += ["",
       "> RTF（real-time factor）= 生成耗时 / 音频时长。RTF < 1 表示比实时快。",
       "> 生成配置属于**保守设置**：受 4 GB 显存限制，仅生成约 4 秒片段；",
       "> MusicGen-small 论文与官方 demo 通常在更长时长（10–30 秒）下评估。"]

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "h8_benchmark.md"),
          "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print("\n" + "=" * 64)
print(f"[done] 生成 {len(rows)} 段音频 -> {OUTDIR}")
print(f"[done] 实测记录表 -> h8_benchmark.md")
if DEVICE == "cuda":
    print(f"[done] 全程显存峰值 {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
print("=" * 64)
