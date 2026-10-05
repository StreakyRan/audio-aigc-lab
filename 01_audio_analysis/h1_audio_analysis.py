# -*- coding: utf-8 -*-
"""
H1 动手实验 v2：音频信号分析（修正 MFCC 可视化 + 精确 LUFS）
输出：波形图 / log-Mel 谱 / MFCC(去c0) / 响度(LUFS或RMS)
用法：
    python h1_audio_analysis.py "你的音频路径.wav"
    python h1_audio_analysis.py            # 不给参数则用 librosa 示例音频
"""
import sys
import os

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display

for f in ["Microsoft YaHei", "SimHei", "SimSun"]:
    try:
        matplotlib.rcParams["font.sans-serif"] = [f]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        pass

# ---------------- 参数（面试要能说出这三个数） ----------------
SR       = 44100     # 目标采样率
N_FFT    = 2048      # FFT 点数  -> 频率 bin = N_FFT/2 + 1 = 1025
HOP      = 512       # 帧移      -> 帧率 = SR/HOP ≈ 86.13 fps
N_MELS   = 128       # 梅尔滤波器数
N_MFCC   = 13        # MFCC 静态系数（含 c0）
DURATION = 30.0

# ---------------- 读音频 ----------------
if len(sys.argv) > 1:
    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"[错误] 找不到文件：{path}")
        sys.exit(1)
    source_name = os.path.basename(path)
else:
    path = librosa.ex("trumpet")
    source_name = "librosa example: trumpet"

y, sr = librosa.load(path, sr=SR, mono=True, duration=DURATION)
dur = len(y) / sr

print("=" * 62)
print(f"音频来源 : {source_name}")
print(f"时长     : {dur:.2f} s")
print(f"采样率   : {sr} Hz")
print(f"采样点数 : {len(y)}")
print(f"峰值     : {np.max(np.abs(y)):.4f}")
print("=" * 62)

# ---------------- 1. STFT ----------------
D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
n_bins, n_frames = D.shape
print("\n【STFT 矩阵】")
print(f"  形状            : ({n_bins}, {n_frames})   <- (频率bins, 帧数)")
print(f"  频率 bins 数     : {n_bins}   = n_fft/2 + 1")
print(f"  帧数             : {n_frames}")
print(f"  帧率             : {sr}/{HOP} = {sr/HOP:.2f} fps")
print(f"  验证 时长x帧率   : {dur:.2f} x {sr/HOP:.2f} = {dur*sr/HOP:.1f} (~ {n_frames})")
print(f"  频率分辨率 df    : {sr/N_FFT:.2f} Hz")
print(f"  时间分辨率 dt    : {N_FFT/sr*1000:.2f} ms")

# ---------------- 2. log-Mel ----------------
M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmax=sr//2)
M_db = librosa.power_to_db(M, ref=np.max)
print("\n【log-Mel 频谱】")
print(f"  n_mels           : {N_MELS}")
print(f"  形状             : {M.shape}   <- ({N_MELS}, {n_frames})")
print(f"  动态范围         : {M_db.min():.1f} ~ {M_db.max():.1f} dB")

# ---------------- 3. MFCC ----------------
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
delta  = librosa.feature.delta(mfcc)
delta2 = librosa.feature.delta(mfcc, order=2)
mfcc39 = np.vstack([mfcc, delta, delta2])
print("\n【MFCC】")
print(f"  静态系数(含c0)   : {mfcc.shape}")
print(f"  +Delta +Delta2   : {mfcc39.shape}   <- 39 维")
print(f"  c0 范围          : {mfcc[0].min():.1f} ~ {mfcc[0].max():.1f}   <- 量级压倒性大")
print(f"  c1~c12 范围      : {mfcc[1:].min():.1f} ~ {mfcc[1:].max():.1f}")
print(f"  => c0 与其他系数量级差 {abs(mfcc[0]).mean()/max(abs(mfcc[1:]).mean(),1e-9):.1f} 倍，所以通常丢弃 c0")

# ---------------- 4. 响度 ----------------
lufs_val = None
try:
    import pyloudnorm as pyln
    meter = pyln.Meter(sr)
    lufs_val = meter.integrated_loudness(y)
    print("\n【响度 LUFS (ITU-R BS.1770)】")
    print(f"  Integrated       : {lufs_val:.2f} LUFS")
    print(f"  与流媒体目标差   : {lufs_val - (-14):+.2f} LU (相对 -14 LUFS)")
except ImportError:
    print("\n【响度 RMS 近似】")
    print("  [提示] 装 pyloudnorm 可得精确 LUFS: pip install pyloudnorm")

rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
rms_db = 20 * np.log10(rms + 1e-9)
print(f"  平均 RMS         : {rms_db.mean():.1f} dBFS")
print(f"  RMS 动态范围     : {rms_db.max()-rms_db.min():.1f} dB")

# ---------------- 画图 ----------------
fig, ax = plt.subplots(5, 1, figsize=(13, 15))

librosa.display.waveshow(y, sr=sr, ax=ax[0])
ax[0].set_title(f"1) Waveform - {source_name} ({dur:.1f}s @ {sr}Hz)", fontsize=12)

img1 = librosa.display.specshow(M_db, sr=sr, hop_length=HOP, x_axis="time", y_axis="mel",
                                fmax=sr//2, ax=ax[1], cmap="magma")
ax[1].set_title(f"2) log-Mel Spectrogram ({N_MELS} mels, n_fft={N_FFT}, hop={HOP} -> {sr/HOP:.1f} fps)", fontsize=12)
fig.colorbar(img1, ax=ax[1], format="%+2.0f dB")

# MFCC 含 c0（说明为什么 c0 要丢）
img2 = librosa.display.specshow(mfcc, sr=sr, hop_length=HOP, x_axis="time", ax=ax[2], cmap="coolwarm")
ax[2].set_title(f"3) MFCC 13-D (WITH c0) - 注意最底下一行 c0 量级压倒一切", fontsize=12)
fig.colorbar(img2, ax=ax[2])

# MFCC 去掉 c0 + 归一化色标
mfcc_drop = mfcc[1:]
img3 = librosa.display.specshow(mfcc_drop, sr=sr, hop_length=HOP, x_axis="time", ax=ax[3],
                                cmap="coolwarm", vmin=-np.abs(mfcc_drop).max(), vmax=np.abs(mfcc_drop).max())
ax[3].set_title("4) MFCC 12-D (c0 DROPPED) - 这才是能看清结构的画法", fontsize=12)
fig.colorbar(img3, ax=ax[3])

t = librosa.times_like(rms_db, sr=sr, hop_length=HOP)
ax[4].plot(t, rms_db, color="tab:green", label="RMS (dBFS)")
if lufs_val is not None:
    ax[4].axhline(lufs_val, color="tab:red", ls="--", label=f"Integrated LUFS = {lufs_val:.1f}")
    ax[4].axhline(-14, color="tab:orange", ls=":", label="Streaming target -14 LUFS")
ax[4].set_title("5) Loudness contour", fontsize=12)
ax[4].set_xlabel("Time (s)"); ax[4].set_ylabel("dBFS / LUFS"); ax[4].grid(alpha=0.3); ax[4].legend()

plt.tight_layout()
outdir = os.path.dirname(os.path.abspath(__file__))
out_png = os.path.join(outdir, "h1_audio_analysis.png")
plt.savefig(out_png, dpi=125)
print(f"\n[OK] 图已保存: {out_png}")

np.save(os.path.join(outdir, "h1_mfcc39.npy"), mfcc39)
print(f"[OK] 特征已保存: h1_mfcc39.npy  shape {mfcc39.shape}")
