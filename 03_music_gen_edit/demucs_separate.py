# -*- coding: utf-8 -*-
"""
H9-1 · 音乐源分离（Demucs v4 / htdemucs）
把混音拆成 4 条 stem：vocals / drums / bass / other

用法：
    set HF_HOME=D:\\hf_cache
    python demucs_separate.py                       # 用 librosa 自带示例（drum+bass）
    python demucs_separate.py "your_song.wav"

产出：
    outputs/stems/<name>/{vocals,drums,bass,other}.wav
    outputs/stems/<name>/mixture.wav
    h9_separation_report.md
"""
import os
import sys
import time
import json

os.environ.setdefault("HF_HOME", r"D:\hf_cache")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
import soundfile as sf
import librosa

MODEL_NAME = "htdemucs"
STEMS = ["drums", "bass", "other", "vocals"]   # demucs 的固定输出顺序
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 44100

HERE = os.path.dirname(os.path.abspath(__file__))
OUTROOT = os.path.join(HERE, "outputs", "stems")

# ---------------- 输入 ----------------
if len(sys.argv) > 1:
    audio_path = sys.argv[1]
    if not os.path.isfile(audio_path):
        print(f"[错误] 找不到文件：{audio_path}"); sys.exit(1)
    name = os.path.splitext(os.path.basename(audio_path))[0]
else:
    audio_path = librosa.ex("choice")
    name = "choice_drumbass"

print("=" * 66)
print(f"输入      : {audio_path}")
print(f"模型      : {MODEL_NAME}")
print(f"设备      : {DEVICE}  ({torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'})")
if DEVICE == "cuda":
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"显存可用  : {free_b/1024**3:.2f} GB / {total_b/1024**3:.2f} GB")
print("=" * 66)

# ---------------- 加载音频 ----------------
wav_np, sr_in = librosa.load(audio_path, sr=SR, mono=False)
if wav_np.ndim == 1:
    wav_np = np.stack([wav_np, wav_np])          # 单声道复制为双声道
duration = wav_np.shape[-1] / SR
print(f"\n[input] 采样率 {sr_in} Hz | 声道 {wav_np.shape[0]} | 时长 {duration:.2f}s")

wav = torch.from_numpy(np.ascontiguousarray(wav_np)).float().unsqueeze(0)   # (1, 2, T)

# ---------------- 加载 Demucs ----------------
from demucs.pretrained import get_model
from demucs.apply import apply_model

t0 = time.time()
model = get_model(MODEL_NAME)
model.eval()
load_s = time.time() - t0
n_params = sum(p.numel() for p in model.parameters())
print(f"[model] 加载 {load_s:.1f}s | 参数量 {n_params/1e6:.1f}M ({n_params:,})")

# ---------------- 分离 ----------------
if DEVICE == "cuda":
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

t1 = time.time()
with torch.no_grad():
    sources = apply_model(
        model, wav.to(DEVICE),
        device=DEVICE,
        shifts=1,           # 不做 shift 平均，省算力
        split=True,
        overlap=0.25,
        progress=True,
    )[0]
sep_s = time.time() - t1

sources = sources.cpu()
vram_peak = torch.cuda.max_memory_allocated() / 1024**3 if DEVICE == "cuda" else 0.0

# ---------------- 保存 ----------------
outdir = os.path.join(OUTROOT, name)
os.makedirs(outdir, exist_ok=True)

stem_audio = {}
for i, s in enumerate(STEMS):
    arr = sources[i].numpy().T                     # (T, C)
    # 峰值归一化到 0.95，避免写文件削波
    pk = float(np.max(np.abs(arr))) or 1.0
    arr = arr / pk * 0.95
    stem_audio[s] = arr
    sf.write(os.path.join(outdir, f"{s}.wav"), arr, SR)

# 原始混音也存一份（便于对比）
mix = wav_np.T
sf.write(os.path.join(outdir, "mixture.wav"), mix / (np.max(np.abs(mix)) or 1.0) * 0.95, SR)

# ---------------- 质量指标 ----------------
# 1) 重建保真度：各 stem 之和 与 原始混音 的差异
recon = np.stack([stem_audio[s] for s in STEMS]).sum(axis=0)
mix_aligned = mix[:recon.shape[0]]
# 两条信号独立归一化后比较形状相似度
a = recon / (np.max(np.abs(recon)) or 1.0)
b = mix_aligned / (np.max(np.abs(mix_aligned)) or 1.0)
mae = float(np.mean(np.abs(a - b)))
snr = float(10 * np.log10((np.sum(b ** 2) + 1e-12) / (np.sum((b - a) ** 2) + 1e-12)))

# 2) 每条 stem 的能量占比
energies = {}
for s in STEMS:
    energies[s] = float(np.mean(stem_audio[s] ** 2))
tot = sum(energies.values()) or 1.0
ratios = {s: v / tot for s, v in energies.items()}

rows = []
for s in STEMS:
    y = stem_audio[s].mean(axis=1)
    rows.append({
        "stem": s,
        "rms": float(np.sqrt(np.mean(y ** 2))),
        "peak": float(np.max(np.abs(y))),
        "energy_ratio": ratios[s],
    })

# ---------------- 报告 ----------------
print(f"\n[separate] 耗时 {sep_s:.2f}s | 音频时长 {duration:.2f}s | "
      f"实时倍速 {duration/sep_s:.2f}x | 显存峰值 {vram_peak:.2f} GB")

print("\n各 stem 统计：")
print(f"  {'stem':8s} {'RMS':>10s} {'Peak':>8s} {'能量占比':>10s}")
for r in rows:
    print(f"  {r['stem']:8s} {r['rms']:10.5f} {r['peak']:8.3f} {r['energy_ratio']*100:9.1f}%")

print(f"\n重建检查（4 条 stem 之和 vs 原始混音）：")
print(f"  MAE = {mae:.5f} | SNR = {snr:.2f} dB")

report = [
    "# H9 音乐源分离实测报告 · Demucs v4 (htdemucs)",
    "",
    f"- 输入：{os.path.basename(audio_path)}（{duration:.2f}s @ {SR} Hz）",
    f"- 模型：`{MODEL_NAME}`，参数量 **{n_params/1e6:.1f}M**",
    f"- 设备：{torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}"
    + (f"（可用显存 {free_b/1024**3:.2f} GB / 共 {total_b/1024**3:.2f} GB）" if DEVICE == 'cuda' else ""),
    f"- 加载耗时：{load_s:.1f}s",
    f"- **分离耗时：{sep_s:.2f}s（音频时长 {duration:.2f}s，约 {duration/sep_s:.2f}× 实时）**",
    f"- **显存峰值：{vram_peak:.2f} GB**",
    "",
    "## 各 stem 统计",
    "",
    "| stem | RMS | Peak | 能量占比 |",
    "|---|---|---|---|",
] + [f"| {r['stem']} | {r['rms']:.5f} | {r['peak']:.3f} | {r['energy_ratio']*100:.1f}% |" for r in rows] + [
    "",
    "## 重建保真度",
    "",
    f"把 4 条 stem 相加与原始混音比较：**MAE = {mae:.5f}，SNR = {snr:.2f} dB**。",
    "",
    "> 源分离是**近似分解**：stems 之和 ≈ 原始混音但不严格相等（存在分离残差），",
    "> 这正是后续做 stem 级编辑时会产生**相位/抵消问题**的根源。",
    "",
    "## 输出文件",
    "",
    "```",
    f"outputs/stems/{name}/",
    "├── mixture.wav   （原始混音，归一化）",
    "├── vocals.wav",
    "├── drums.wav",
    "├── bass.wav",
    "└── other.wav",
    "```",
]

with open(os.path.join(HERE, "h9_separation_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))

# 机器可读的指标
with open(os.path.join(HERE, "h9_metrics.json"), "w", encoding="utf-8") as f:
    json.dump({
        "model": MODEL_NAME, "params": n_params, "duration_s": duration,
        "sep_s": sep_s, "realtime_x": duration / sep_s, "vram_peak_gb": vram_peak,
        "mae": mae, "snr_db": snr, "stems": rows,
    }, f, ensure_ascii=False, indent=2)

print(f"\n[done] stems -> {outdir}")
print(f"[done] 报告 -> h9_separation_report.md | 指标 -> h9_metrics.json")
print("=" * 66)
