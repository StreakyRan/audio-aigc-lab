# -*- coding: utf-8 -*-
"""诊断：打印各文件的高频能量比，用于科学设定带宽检测阈值。"""
import os, sys, json
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np, soundfile as sf, librosa

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")

def hf_ratio(y, sr, split_hz=8000.0):
    yp = librosa.effects.preemphasis(y.astype(np.float32), coef=0.97)
    S = np.abs(librosa.stft(yp, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    e = (S ** 2).sum(axis=1)
    tot = e.sum()
    return float(e[freqs >= split_hz].sum() / tot) if tot > 0 else 0.0

def bw99(y, sr):
    S = np.abs(librosa.stft(y.astype(np.float32), n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    e = (S ** 2).sum(axis=1); cum = np.cumsum(e)
    if cum[-1] <= 0: return 0.0
    return float(freqs[min(int(np.searchsorted(cum, 0.99*cum[-1])), len(freqs)-1)])

gt = {r["file"]: r for r in json.load(open(os.path.join(HERE, "h5_ground_truth.json"), encoding="utf-8"))}

print(f"{'file':26s} {'注入了什么':22s} {'sr':>6s} {'BW99':>7s} {'HFratio':>10s}")
print("-" * 80)
rows = []
for f in sorted(os.listdir(RAW)):
    if not f.lower().endswith(".wav"): continue
    y, sr = sf.read(os.path.join(RAW, f), always_2d=True)
    ym = y.mean(axis=1).astype(np.float32)
    r = hf_ratio(ym, sr); b = bw99(ym, sr)
    d = ",".join(gt.get(f, {}).get("injected_defects", [])) or "none"
    rows.append((f, d, sr, b, r))
    print(f"{f:26s} {d:22s} {sr:>6d} {b:>7.0f} {r:>10.5f}")

clean = [r for f, d, sr, b, r in rows if d == "none"]
lp = [r for f, d, sr, b, r in rows if "lowpass" in d]
print()
print(f"干净文件的 HFratio 范围 : {min(clean):.5f} ~ {max(clean):.5f}  (n={len(clean)})")
print(f"低通文件的 HFratio      : {lp}")
if clean and lp:
    print(f"干净最小值 / 低通最大值 = {min(clean)/max(lp):.1f}x")
    print(f"→ 建议阈值取几何中点附近: {np.sqrt(min(clean)*max(lp)):.5f}")
