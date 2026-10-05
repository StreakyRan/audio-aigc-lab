# -*- coding: utf-8 -*-
"""
H5-1 · 生成受控"脏数据"测试集（ground truth）

思路：真实数据里没人知道"哪条被注入过什么缺陷"，因此无法评估检测器。
本脚本把干净音频按**已知的缺陷清单**做受控劣化，产出一份 ground-truth 标签，
用于后续验证清洗流水线的检测准确率（precision / recall）。

用法：
    python make_messy_audio.py
"""
import os
import sys
import json

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import soundfile as sf
import librosa

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
os.makedirs(RAW, exist_ok=True)

SR = 44100
rng = np.random.default_rng(42)

# ---------------- 干净素材 ----------------
SOURCES = {
    "music_choice": "choice",        # 鼓+贝斯
    "music_waltz": "sweetwaltz",     # 圆舞曲
    "music_ragtime": "pistachio",    # 拉格泰姆
}

def load_clean(key, dur=20.0):
    y, _ = librosa.load(librosa.ex(key), sr=SR, mono=False, duration=dur)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y * (0.7 / (np.max(np.abs(y)) + 1e-9))     # 统一峰值 0.7，避免起始就削波

# ---------------- 缺陷注入函数 ----------------
def inject_clipping(y, ratio=0.02, gain=6.0):
    """硬削波：把一段信号大幅放大后 clip 到 ±1.0，形成持续触顶的平顶段"""
    out = y.copy()
    n = out.shape[1]
    k = max(512, int(n * ratio))
    start = int(n * 0.30)
    seg = out[:, start:start + k] * gain
    out[:, start:start + k] = np.clip(seg, -1.0, 1.0)
    return out

def inject_silence(y, ratio=0.35):
    """大段静音：把 35% 的时长置零"""
    out = y.copy()
    n = out.shape[1]
    k = int(n * ratio)
    start = int(n * 0.5)
    out[:, start:start + k] = 0.0
    return out

def inject_dc_offset(y, offset=0.05):
    return y + offset

def inject_lowpass(y, cutoff=3000):
    """带宽受限：低通到 3 kHz（模拟低质量来源 / 被升采样伪装的高采样率文件）"""
    from scipy.signal import butter, sosfilt
    sos = butter(8, cutoff, btype="low", fs=SR, output="sos")
    out = np.empty_like(y)
    for c in range(y.shape[0]):
        out[c] = sosfilt(sos, y[c])
    return out

def inject_noise(y, snr_db=12.0):
    """加高斯噪声到指定 SNR"""
    p_sig = np.mean(y ** 2)
    p_noise = p_sig / (10 ** (snr_db / 10))
    n = rng.normal(0, np.sqrt(p_noise), y.shape)
    return y + n

def inject_short(y, keep_sec=0.4):
    """超短文件（0.4 秒），应被时长门槛淘汰"""
    return y[:, :int(SR * keep_sec)]

# ---------------- 构造数据集 ----------------
PLAN = [
    # (文件名, 源, 缺陷列表, 说明)
    ("s01_clean_choice.wav",      "music_choice",  [],                                  "干净"),
    ("s02_clean_waltz.wav",       "music_waltz",   [],                                  "干净"),
    ("s03_clean_ragtime.wav",     "music_ragtime", [],                                  "干净"),
    ("s04_clipped_choice.wav",    "music_choice",  ["clipping"],                        "硬削波"),
    ("s05_silence_waltz.wav",     "music_waltz",   ["silence"],                         "大段静音"),
    ("s06_dc_ragtime.wav",        "music_ragtime", ["dc_offset"],                       "直流偏移"),
    ("s07_lowpass_choice.wav",    "music_choice",  ["lowpass"],                         "带宽受限 3kHz"),
    ("s08_noisy_waltz.wav",       "music_waltz",   ["noise"],                           "SNR 12dB 噪声"),
    ("s09_short_ragtime.wav",     "music_ragtime", ["short"],                           "超短 0.4s"),
    ("s10_multi_choice.wav",      "music_choice",  ["clipping", "silence", "noise"],    "多重缺陷"),
]

# 原始采样率故意不统一（模拟多来源），清洗阶段需统一到 44.1k
SR_VARIANTS = {3: 22050, 5: 48000, 8: 16000}

ground_truth = []
print("=" * 70)
print("生成受控脏数据 ->", RAW)
print("=" * 70)
print(f"{'文件':28s} {'注入缺陷':28s} {'采样率':>7s} {'时长':>7s}")
for i, (fname, src, defects, note) in enumerate(PLAN, start=1):
    y = load_clean(SOURCES[src])
    # 注入顺序很重要：削波放最后，否则会被后续的静音注入覆盖掉
    ORDER = ["dc_offset", "lowpass", "noise", "short", "silence", "clipping"]
    for d in [x for x in ORDER if x in defects]:
        if d == "clipping":  y = inject_clipping(y)
        elif d == "silence": y = inject_silence(y)
        elif d == "dc_offset": y = inject_dc_offset(y)
        elif d == "lowpass": y = inject_lowpass(y)
        elif d == "noise":   y = inject_noise(y)
        elif d == "short":   y = inject_short(y)

    out_sr = SR_VARIANTS.get(i, SR)
    y_out = y
    if out_sr != SR:
        y_out = np.stack([librosa.resample(y[c], orig_sr=SR, target_sr=out_sr) for c in range(y.shape[0])])

    # 保证不因写文件被削波（保持注入的削波特征：直接 clip 写盘）
    peak = np.max(np.abs(y_out))
    if peak > 0.999 and "clipping" not in defects:
        y_out = y_out / (peak + 1e-9) * 0.95

    path = os.path.join(RAW, fname)
    sf.write(path, y_out.T, out_sr)

    dur = y_out.shape[1] / out_sr
    ground_truth.append({
        "file": fname, "source": src, "injected_defects": defects,
        "note": note, "sample_rate": out_sr, "duration_sec": round(dur, 3),
    })
    print(f"{fname:28s} {','.join(defects) or 'none':28s} {out_sr:>7d} {dur:>6.2f}s")

with open(os.path.join(HERE, "h5_ground_truth.json"), "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, ensure_ascii=False, indent=2)

print()
print(f"[done] 共 {len(ground_truth)} 个文件")
print(f"[done] ground truth -> h5_ground_truth.json")
print("=" * 70)
