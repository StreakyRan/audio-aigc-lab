# -*- coding: utf-8 -*-
"""
H12 · 音频效果生成与音色处理（失真、混响、空间音效）

实现 6 个可参数化的音频效果，并量化每个效果对信号的实际改变：

  ① 失真 Distortion       —— 波形整形（soft/hard clip + 驱动 + 音色）
  ② 混响 Reverb           —— 合成脉冲响应 + 卷积
  ③ 延迟 Delay            —— 带反馈的延迟线
  ④ 合唱 Chorus           —— 调制延迟（多声部加厚）
  ⑤ 空间声像 Panning      —— 声像定位 + 立体声展宽
  ⑥ 吉他完整效果链        —— 噪声门 → 失真 → 三段 EQ → 合唱 → 延迟 → 混响 → 归一化

量化手段（每步都验证"效果真的生效了"）：
  · 波形统计：峰值 / RMS / 峰均比（crest factor）/ 削波率
  · 频谱：谐波是否增加（失真使总谐波失真 THD 上升）
  · 时间：混响衰减时间 RT60、延迟回声间隔
  · 空间：左右声道电平差与相关性
  · 对比：处理前后差异度量

用法：
    python audio_effects_chain.py
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
import scipy.signal as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 图表中文字体（Windows 常见字体，找不到则回退英文标签）
for _f in ("Microsoft YaHei", "SimHei", "DengXian"):
    matplotlib.rcParams["font.sans-serif"] = [_f]
    matplotlib.rcParams["axes.unicode_minus"] = False
    break

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
SR = 44100
rng = np.random.default_rng(42)

def title(s):
    print("\n" + "=" * 80)
    print(s)
    print("=" * 80)

# =====================================================================
# 度量工具
# =====================================================================
def stats(y, sr=SR):
    """波形统计"""
    p = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)))
    crest = p / (rms + 1e-12)                       # 峰均比
    clipped = float(np.mean(np.abs(y) >= 0.999))    # 削波率
    return {"peak": round(p, 4), "rms": round(rms, 5),
            "crest_factor": round(crest, 3), "clipped_ratio": round(clipped, 4)}

def thd_estimate(y, sr=SR, f0=110.0, n_harm=10):
    """总谐波失真（THD）：谐波能量之和 / 基频能量。
    用「频带能量求和」而非峰值，且排除基频附近的泄漏。"""
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    def band_energy(f, w):
        m = (freqs >= f - w) & (freqs <= f + w)
        return float(np.sum(Y[m] ** 2)) if m.any() else 0.0
    bw = max(3.0, f0 * 0.06)                    # 带宽随基频缩放，避免漏掉谐波
    f_energy = band_energy(f0, bw)
    h_energy = sum(band_energy(f0 * k, bw) for k in range(2, n_harm + 1))
    if f_energy <= 0:
        return None
    return round(float(np.sqrt(h_energy / f_energy)), 4)

def rt60(y, sr=SR):
    """用能量衰减曲线（Schroeder 积分）估计 RT60"""
    e = y ** 2
    cum = np.cumsum(e[::-1])[::-1]
    if cum[0] <= 0:
        return None
    db = 10 * np.log10(cum / cum[0] + 1e-12)
    def t_at(level):
        idx = np.where(db <= level)[0]
        return idx[0] / sr if len(idx) else None
    t5, t25 = t_at(-5), t_at(-25)
    if t5 is None or t25 is None or t25 <= t5:
        return None
    return round(float((t25 - t5) * 60 / 20), 3)      # 用 -5~-25dB 段外推

def stereo_stats(y):
    """立体声：左右电平差与声道相关性"""
    if y.ndim == 1 or y.shape[1] < 2:
        return {"lr_level_diff_db": None, "correlation": None}
    L, R = y[:, 0].astype(np.float64), y[:, 1].astype(np.float64)
    rl = np.sqrt(np.mean(L ** 2)); rr = np.sqrt(np.mean(R ** 2))
    ld = 20 * np.log10(rl + 1e-12); rd = 20 * np.log10(rr + 1e-12)
    # 任一声道为静音时，相关性无定义（避免 std=0 导致 nan 警告）
    if rl < 1e-9 or rr < 1e-9:
        corr = None
    else:
        corr = round(float(np.corrcoef(L, R)[0, 1]), 4)
    return {"lr_level_diff_db": round(ld - rd, 2) if (rl > 1e-9 and rr > 1e-9) else None,
            "correlation": corr}

def diff_metric(a, b):
    """两个信号的整体差异（归一化后）"""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    an = a / (np.max(np.abs(a)) + 1e-12)
    bn = b / (np.max(np.abs(b)) + 1e-12)
    return round(float(np.sqrt(np.mean((an - bn) ** 2))), 5)

# =====================================================================
title("1. 准备输入素材（用真实吉他/合成信号）")

def make_guitar_like(freq, dur, sr=SR):
    """合成类吉他信号：基频 + 谐波 + 拨弦瞬态（便于验证失真产生的谐波）"""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    y = np.zeros_like(t)
    # 吉他音色：奇次谐波较强
    for k, amp in enumerate([1.0, 0.15, 0.45, 0.08, 0.22, 0.05, 0.12], start=1):
        y += amp * np.sin(2 * np.pi * freq * k * t)
    # 拨弦包络：快速起音 + 缓慢衰减
    env = np.exp(-t * 2.5)
    e = np.ones_like(t)
    a = int(0.005 * sr)
    e[:a] = np.linspace(0, 1, a)
    y = y * env * e
    return (y / (np.max(np.abs(y)) + 1e-12) * 0.8).astype(np.float32)

# 主素材：A2 音符（110Hz），4 秒，单次拨弦 + 和弦
NOTE = 110.0
dry_note = make_guitar_like(NOTE, 2.0)
sf.write(os.path.join(OUT, "h12_input_note.wav"), dry_note, SR)

dry_chord = sum(make_guitar_like(f, 2.0) for f in (NOTE, NOTE * 5 / 4, NOTE * 3 / 2)) / 3
dry_chord = (dry_chord / np.max(np.abs(dry_chord)) * 0.8).astype(np.float32)
sf.write(os.path.join(OUT, "h12_input_chord.wav"), dry_chord, SR)

print(f"输入素材: A2 音符（110 Hz）+ A 大三和弦（A-E-C#），各 2.0s @ {SR} Hz")
print(f"干声统计: {stats(dry_note)}")
print(f"干声 THD 估计: {thd_estimate(dry_note, f0=NOTE)}")

# =====================================================================
title("2. ① 失真（Distortion）—— 波形整形")

# ⚠️ 度量说明：THD 必须在「纯正弦」上测，否则输入信号本身已有的谐波会干扰结果。
#    本例：类吉他信号本身含 7 个谐波（THD≈0.54），失真后再测 THD 反而下降
#    （因为 tanh 压缩让基频相对谐波的占比变化被归一化掉了）。
#    因此 THD 用纯正弦测试，音频演示仍用吉他信号（可听感对比）。
_t = np.linspace(0, 2.0, int(SR * 2.0), endpoint=False)
SINE_TEST = (np.sin(2 * np.pi * NOTE * _t) * 0.8).astype(np.float32)

def soft_clip(x, drive=1.0):
    """软削波：tanh，产生偶次+奇次谐波，听感温暖"""
    return np.tanh(x * drive)

def hard_clip(x, drive=1.0):
    """硬削波：直接截断，产生强烈奇次谐波，听感尖锐"""
    return np.clip(x * drive, -1.0, 1.0)

def distortion(x, drive=1.0, mode="soft", tone=0.5):
    """
    drive: 驱动量（越大失真越强）
    mode : soft / hard
    tone : 音色亮度（对失真后信号做倾斜滤波）
    """
    y = soft_clip(x, drive) if mode == "soft" else hard_clip(x, drive)
    # tone：一阶倾斜滤波（高频增益）
    if tone != 0.5:
        b, a = sps.butter(1, 2000 / (SR / 2), btype="low")
        low = sps.lfilter(b, a, y)
        high = y - low
        y = low + high * (tone * 2)          # tone>0.5 提亮
    return (y / (np.max(np.abs(y)) + 1e-12) * 0.9).astype(np.float32)

DIST_TABLE = []
for mode in ("soft", "hard"):
    for drive in (1.0, 4.0, 12.0):
        y = distortion(dry_note, drive=drive, mode=mode)          # 音频演示（吉他信号）
        y_sine = distortion(SINE_TEST, drive=drive, mode=mode)    # 度量用（纯正弦）
        s = stats(y)
        thd = thd_estimate(y_sine, f0=NOTE)
        DIST_TABLE.append({"mode": mode, "drive": drive, "thd": thd, **s})
        sf.write(os.path.join(OUT, f"h12_dist_{mode}_drive{int(drive)}.wav"), y, SR)

print(f"{'模式':>6} {'驱动':>6} {'峰值':>8} {'RMS':>9} {'峰均比':>8} {'THD':>9}")
print("-" * 52)
_base_thd = thd_estimate(SINE_TEST, f0=NOTE)
print(f"{'干声':>6} {'—':>6} {stats(dry_note)['peak']:>8.4f} {stats(dry_note)['rms']:>9.5f} "
      f"{stats(dry_note)['crest_factor']:>8.3f} {_base_thd:>9.4f}")
for r in DIST_TABLE:
    print(f"{r['mode']:>6} {r['drive']:>6.1f} {r['peak']:>8.4f} {r['rms']:>9.5f} "
          f"{r['crest_factor']:>8.3f} {r['thd']:>9.4f}")
print()
print("注：THD 在纯正弦上测量（吉他和弦本身含谐波，会干扰失真度量）。")

# =====================================================================
title("3. ② 混响（Reverb）—— 合成脉冲响应 + 卷积")

def make_ir(rt60_sec=1.5, sr=SR, pre_delay=0.02, damping=0.4, stereo=True):
    """
    合成脉冲响应（IR）：
      · 指数衰减噪声模型（简化版：真实混响用射线追踪/测量，这里用统计模型）
      · pre_delay: 直达声后的初始延迟
      · damping  : 高频衰减更快（模拟空气与材料吸收）
    """
    n = int(sr * (rt60_sec + 0.2))
    t = np.arange(n) / sr
    # 指数衰减包络
    env = np.exp(-6.9 * t / max(rt60_sec, 1e-3))        # -60dB 处约 rt60
    ir = rng.normal(0, 1, n) * env
    # 预延迟
    pd = int(pre_delay * sr)
    if pd > 0:
        ir[:pd] = 0
    # 阻尼：随时间增强的低通（高频先衰减）
    b, a = sps.butter(1, 6000 / (sr / 2), btype="low")
    lp = sps.lfilter(b, a, ir)
    ir = (1 - damping) * ir + damping * lp
    ir = ir / (np.max(np.abs(ir)) + 1e-12)
    if stereo:
        ir2 = np.roll(ir, int(0.003 * sr))               # 左右微小时间差 → 空间感
        return np.stack([ir, ir2], axis=1).astype(np.float32)
    return ir.astype(np.float32)

def reverb(x, rt60_sec=1.5, wet=0.35, damping=0.4):
    ir = make_ir(rt60_sec, damping=damping, stereo=True)
    x2 = np.stack([x, x], axis=1) if x.ndim == 1 else x
    n = len(x2) + len(ir) - 1
    y = np.stack([
        sps.fftconvolve(x2[:, 0], ir[:, 0])[:n],
        sps.fftconvolve(x2[:, 1], ir[:, 1])[:n],
    ], axis=1)
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
    dry = np.concatenate([x2, np.zeros((n - len(x2), 2))], axis=0)
    out = dry * (1 - wet) + y * wet
    return (out / (np.max(np.abs(out)) + 1e-12) * 0.9).astype(np.float32)

REV_TABLE = []
for rt in (0.4, 1.5, 3.0):
    for wet in (0.2, 0.5):
        ir_check = make_ir(rt, damping=0.4, stereo=False)     # 从 IR 本身测 RT60（准确做法）
        y = reverb(dry_chord, rt60_sec=rt, wet=wet)
        REV_TABLE.append({"target_rt60": rt, "wet": wet,
                          "measured_rt60_from_ir": rt60(ir_check),
                          "stereo": stereo_stats(y), **stats(y)})
        sf.write(os.path.join(OUT, f"h12_reverb_rt{rt}_wet{int(wet*100)}.wav"), y, SR)

print(f"{'目标 RT60':>10} {'湿声比':>7} {'IR 实测 RT60':>13} {'左右电平差':>11} {'声道相关':>10}")
print("-" * 58)
for r in REV_TABLE:
    lr = r['stereo']['lr_level_diff_db']
    co = r['stereo']['correlation']
    print(f"{r['target_rt60']:>10.1f} {r['wet']:>7.1f} "
          f"{str(r['measured_rt60_from_ir']):>13} "
          f"{(f'{lr:.2f}' if lr is not None else 'n/a'):>11} "
          f"{(f'{co:.4f}' if co is not None else 'n/a'):>10}")

# =====================================================================
title("4. ③④ 延迟与合唱（Delay / Chorus）")

def delay(x, delay_ms=250.0, feedback=0.35, mix=0.3, pingpong=True):
    """反馈延迟线；pingpong=True 时回声左右交替（空间效果）"""
    d = int(delay_ms / 1000 * SR)
    x2 = np.stack([x, x], axis=1) if x.ndim == 1 else x
    n = len(x2) + d * 4
    y = np.zeros((n, 2), dtype=np.float64)
    y[:len(x2)] += x2
    buf = np.zeros((d, 2))
    for i in range(len(x2)):
        bi = i % d
        e = buf[bi].copy()
        buf[bi] = x2[i] + e * feedback
        if pingpong:
            y[i + d, 0] += e[0]; y[i + d, 1] += e[1]
        else:
            y[i + d] += e
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
    dry = np.zeros_like(y); dry[:len(x2)] = x2
    out = dry * (1 - mix) + y * mix
    return (out / (np.max(np.abs(out)) + 1e-12) * 0.9).astype(np.float32)

DELAY_TABLE = []
for dms in (120.0, 250.0, 450.0):
    for fb in (0.2, 0.5):
        y = delay(dry_note, delay_ms=dms, feedback=fb, mix=0.35)
        DELAY_TABLE.append({"delay_ms": dms, "feedback": fb, **stats(y)})
        sf.write(os.path.join(OUT, f"h12_delay_{int(dms)}ms_fb{int(fb*100)}.wav"), y, SR)

def chorus(x, rate_hz=0.6, depth_ms=6.0, voices=3, mix=0.4):
    """合唱：多个 LFO 调制的短延迟叠加（加厚音色）"""
    x2 = np.stack([x, x], axis=1) if x.ndim == 1 else x
    t = np.arange(len(x2)) / SR
    out = np.zeros_like(x2, dtype=np.float64)
    for v in range(voices):
        phase = 2 * np.pi * v / voices
        lfo = np.sin(2 * np.pi * rate_hz * t + phase)
        d_samples = (depth_ms * (1 + lfo) / 2) / 1000 * SR     # 0~depth_ms
        idx = np.arange(len(x2)) - d_samples
        idx0 = np.clip(np.floor(idx).astype(int), 0, len(x2) - 1)
        frac = np.clip(idx - idx0, 0, 1)
        idx1 = np.clip(idx0 + 1, 0, len(x2) - 1)
        for c in range(2):
            out[:, c] += x2[idx0, c] * (1 - frac) + x2[idx1, c] * frac
    out /= voices
    out = x2 * (1 - mix) + out * mix
    return (out / (np.max(np.abs(out)) + 1e-12) * 0.9).astype(np.float32)

cho = chorus(dry_chord)
sf.write(os.path.join(OUT, "h12_chorus.wav"), cho, SR)

print("延迟：")
print(f"{'延迟(ms)':>9} {'反馈':>6} {'峰值':>8} {'RMS':>9} {'峰均比':>8}")
for r in DELAY_TABLE:
    print(f"{r['delay_ms']:>9.1f} {r['feedback']:>6.1f} {r['peak']:>8.4f} {r['rms']:>9.5f} {r['crest_factor']:>8.3f}")
c_st = stereo_stats(cho)
print()
print(f"合唱：与干声差异 {diff_metric(dry_chord, cho[:,0])}；声道相关性 {c_st['correlation']}")

# =====================================================================
title("5. ⑤ 空间声像与立体声展宽（Panning / Stereo Widening）")

def pan(x, pos=-0.6):
    """声像定位：pos ∈ [-1,1]，-1=全左，0=居中，1=全右（等功率定律）"""
    x2 = np.stack([x, x], axis=1) if x.ndim == 1 else x
    theta = (pos + 1) * np.pi / 4                         # 0 ~ pi/2
    gl, gr = np.cos(theta), np.sin(theta)
    return (np.stack([x2[:, 0] * gl, x2[:, 1] * gr], axis=1)).astype(np.float32)

def widen(x, width=1.5):
    """立体声展宽（Mid/Side 处理）：width>1 变宽，<1 变窄"""
    if x.ndim == 1:
        return x
    mid = (x[:, 0] + x[:, 1]) / 2
    side = (x[:, 0] - x[:, 1]) / 2 * width
    L = mid + side; R = mid - side
    y = np.stack([L, R], axis=1)
    return (y / (np.max(np.abs(y)) + 1e-12) * 0.9).astype(np.float32)

PAN_TABLE = []
for pos in (-1.0, -0.5, 0.0, 0.5, 1.0):
    y = pan(dry_chord, pos=pos)
    PAN_TABLE.append({"pan": pos, **stereo_stats(y)})
    sf.write(os.path.join(OUT, f"h12_pan_{int((pos+1)*50):02d}.wav"), y, SR)

WID_TABLE = []
stereo_in = np.stack([dry_chord, np.roll(dry_chord, 200)], axis=1)
for w in (0.0, 0.5, 1.0, 1.5, 2.0):
    y = widen(stereo_in, width=w)
    WID_TABLE.append({"width": w, **stereo_stats(y)})
    sf.write(os.path.join(OUT, f"h12_width_{int(w*100)}.wav"), y, SR)

print("声像定位（等功率定律，-1 全左 / 1 全右）：")
print(f"{'pan':>6} {'左右电平差(dB)':>14} {'声道相关':>10}")
print("-" * 34)
for r in PAN_TABLE:
    lr, co = r['lr_level_diff_db'], r['correlation']
    s_lr = f"{lr:>14.2f}" if lr is not None else "n/a".rjust(14)
    s_co = f"{co:>10.4f}" if co is not None else "n/a".rjust(10)
    print(f"{r['pan']:>6.1f} {s_lr} {s_co}")

print()
print("立体声展宽（Mid/Side）：")
print(f"{'width':>6} {'左右电平差(dB)':>14} {'声道相关':>10}")
print("-" * 34)
for r in WID_TABLE:
    lr, co = r['lr_level_diff_db'], r['correlation']
    s_lr = f"{lr:>14.2f}" if lr is not None else "n/a".rjust(14)
    s_co = f"{co:>10.4f}" if co is not None else "n/a".rjust(10)
    print(f"{r['width']:>6.1f} {s_lr} {s_co}")

# =====================================================================
title("6. ⑥ 吉他完整效果链（噪声门 → 失真 → EQ → 合唱 → 延迟 → 混响 → 归一化）")

def noise_gate(x, threshold=0.01, attack_ms=1.0, release_ms=50.0):
    """噪声门：低于阈值时衰减（去底噪/手指杂音）"""
    env = np.abs(sps.lfilter(*sps.butter(1, 20 / (SR / 2), btype="low"), np.abs(x)))
    gate = (env > threshold).astype(np.float32)
    # 平滑（attack/release）
    a = np.exp(-1 / (SR * attack_ms / 1000))
    r = np.exp(-1 / (SR * release_ms / 1000))
    sm = np.zeros_like(gate)
    prev = 0.0
    for i, g in enumerate(gate):
        coef = a if g > prev else r
        prev = coef * prev + (1 - coef) * g
        sm[i] = prev
    return x * sm

def three_band_eq(x, low_gain_db=0.0, mid_gain_db=0.0, high_gain_db=0.0,
                  lo=250.0, hi=4000.0):
    """三段 EQ：低/中/高独立增益（用带通拆分后加权求和）"""
    def db2lin(d): return 10 ** (d / 20)
    sos_lo = sps.butter(2, lo / (SR / 2), btype="low", output="sos")
    sos_hi = sps.butter(2, hi / (SR / 2), btype="high", output="sos")
    low = sps.sosfilt(sos_lo, x)
    high = sps.sosfilt(sos_hi, x)
    mid = x - low - high
    y = low * db2lin(low_gain_db) + mid * db2lin(mid_gain_db) + high * db2lin(high_gain_db)
    return y

def guitar_chain(x, cfg):
    """完整吉他效果链（输出单声道；立体声版另行展宽处理）"""
    y = x.astype(np.float64)
    stages = []
    to_mono = lambda a: a.mean(axis=1) if (a.ndim == 2) else a
    if cfg.get("gate"):
        y = noise_gate(y, threshold=cfg["gate"])
        stages.append(("noise_gate", stats(y), None))
    if cfg.get("drive"):
        y = distortion(y, drive=cfg["drive"], mode=cfg.get("dist_mode", "soft"),
                       tone=cfg.get("tone", 0.5))
        stages.append(("distortion", stats(y), thd_estimate(y, f0=NOTE)))
    if cfg.get("eq"):
        lg, mg, hg = cfg["eq"]
        y = three_band_eq(y, lg, mg, hg)
        stages.append(("eq", stats(y), None))
    if cfg.get("chorus"):
        y = to_mono(chorus(y, mix=cfg["chorus"]))
        stages.append(("chorus", stats(y), None))
    if cfg.get("delay"):
        y = to_mono(delay(y, delay_ms=cfg["delay"][0], feedback=cfg["delay"][1], mix=cfg["delay"][2]))
        stages.append(("delay", stats(y), None))
    if cfg.get("reverb"):
        rt, wet = cfg["reverb"]
        y = to_mono(reverb(y, rt60_sec=rt, wet=wet))
        stages.append(("reverb", stats(y), None))
    # 输出归一化（防削波）
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.95
    return y.astype(np.float32), stages

CFG = {
    "gate": 0.02,
    "drive": 8.0, "dist_mode": "soft", "tone": 0.62,
    "eq": (-2.0, 1.5, 3.0),          # 低频收、中频微提、高频提亮
    "chorus": 0.25,
    "delay": (320.0, 0.30, 0.20),
    "reverb": (1.8, 0.28),
}

wet, stages = guitar_chain(dry_note, CFG)
sf.write(os.path.join(OUT, "h12_guitar_chain_wet.wav"), wet, SR)
# 同时导出立体声版（把链式输出做展宽）
wet_st = widen(np.stack([wet, np.roll(wet, 320)], axis=1), width=1.3)
sf.write(os.path.join(OUT, "h12_guitar_chain_wet_stereo.wav"), wet_st, SR)

print(f"{'阶段':>14} {'峰值':>8} {'RMS':>9} {'峰均比':>8} {'THD':>9}")
print("-" * 52)
print(f"{'干声':>14} {stats(dry_note)['peak']:>8.4f} {stats(dry_note)['rms']:>9.5f} "
      f"{stats(dry_note)['crest_factor']:>8.3f} {thd_estimate(dry_note, f0=NOTE):>9.4f}")
for name, s, thd in stages:
    print(f"{name:>14} {s['peak']:>8.4f} {s['rms']:>9.5f} {s['crest_factor']:>8.3f} "
          f"{(thd if thd is not None else '—'):>9}")
print()
print(f"最终输出: 峰值 {stats(wet)['peak']:.4f} | 与干声差异 {diff_metric(dry_note, wet)}")
print(f"立体声版: 声道相关性 {stereo_stats(wet_st)['correlation']}")

# =====================================================================
# =====================================================================
title("7. 可视化：效果量化曲线")

# ---- 图 1：失真 —— THD 与峰均比随驱动量的变化（双 Y 轴）----
fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=150)
ax2 = ax1.twinx()
drives = sorted(set(r["drive"] for r in DIST_TABLE))
for mode, color in (("soft", "#1f77b4"), ("hard", "#d62728")):
    thds = [next(r["thd"] for r in DIST_TABLE if r["mode"] == mode and r["drive"] == d) for d in drives]
    crests = [next(r["crest_factor"] for r in DIST_TABLE if r["mode"] == mode and r["drive"] == d) for d in drives]
    ax1.plot(drives, thds, "o-", color=color, lw=1.8, label=f"THD（{mode} clip）")
    ax2.plot(drives, crests, "s--", color=color, lw=1.5, alpha=0.6, label=f"峰均比（{mode} clip）")
ax1.set_xlabel("驱动量 (drive)")
ax1.set_ylabel("THD（总谐波失真）", color="#333333")
ax2.set_ylabel("峰均比 crest factor", color="#666666")
ax1.set_title("① 失真：驱动量 ↑  →  THD ↑（谐波增加）、峰均比 ↓（波形被压平）")
ax1.grid(alpha=0.3)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "h12_plot_distortion.png"))
plt.close(fig)
print(f"  -> h12_plot_distortion.png")

# ---- 图 2：混响 —— Schroeder 能量衰减曲线（RT60 测量标准做法）----
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
for rt, color in ((0.4, "#2ca02c"), (1.5, "#ff7f0e"), (3.0, "#9467bd")):
    ir = make_ir(rt, damping=0.4, stereo=False).astype(np.float64)
    e = ir ** 2
    cum = np.cumsum(e[::-1])[::-1]                       # Schroeder 反向积分
    db = 10 * np.log10(cum / (cum[0] + 1e-12) + 1e-12)
    tt = np.arange(len(db)) / SR
    ax.plot(tt, db, color=color, lw=1.8, label=f"目标 RT60 = {rt}s")
ax.axhline(-60, color="gray", ls=":", lw=1)
ax.text(0.05, -58, "−60 dB（RT60 定义点）", fontsize=8, color="gray")
ax.set_xlabel("时间 (s)")
ax.set_ylabel("残余能量 (dB)")
ax.set_xlim(0, 3.5)
ax.set_ylim(-75, 3)
ax.set_title("② 混响：Schroeder 能量衰减曲线（斜率对应 RT60）")
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "h12_plot_reverb_decay.png"))
plt.close(fig)
print(f"  -> h12_plot_reverb_decay.png")

# ---- 图 3：声像与立体声展宽 —— 声道相关性 ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
pan_v = [r["pan"] for r in PAN_TABLE if r["lr_level_diff_db"] is not None]
pan_lr = [r["lr_level_diff_db"] for r in PAN_TABLE if r["lr_level_diff_db"] is not None]
axes[0].plot(pan_v, pan_lr, "o-", color="#1f77b4")
axes[0].axhline(0, color="gray", ls=":", lw=1)
axes[0].set_xlabel("声像位置 pan（-1 全左 → +1 全右）")
axes[0].set_ylabel("左右电平差 (dB)")
axes[0].set_title("③ 等功率声像定律")
axes[0].grid(alpha=0.3)

wid_v = [r["width"] for r in WID_TABLE]
wid_c = [r["correlation"] for r in WID_TABLE]
axes[1].plot(wid_v, wid_c, "o-", color="#d62728")
axes[1].axhline(0, color="gray", ls=":", lw=1)
axes[1].set_xlabel("展宽 width（Mid/Side）")
axes[1].set_ylabel("左右声道相关性")
axes[1].set_title("④ 立体声展宽：width ↑ → 相关性 ↓")
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "h12_plot_stereo.png"))
plt.close(fig)
print(f"  -> h12_plot_stereo.png")

# =====================================================================
# 上表即为最终结果；以下写入 JSON / Markdown 报告

effects_summary = {
    "distortion": DIST_TABLE,
    "reverb": REV_TABLE,
    "delay": DELAY_TABLE,
    "pan": PAN_TABLE,
    "width": WID_TABLE,
    "chorus_stereo": stereo_stats(cho),
    "chain_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()},
    "chain_stages": [{"stage": n, **s, "thd": t} for n, s, t in stages],
    "mix_metric_dry_vs_wet": diff_metric(dry_note, wet),
}
with open(os.path.join(HERE, "h12_effects_report.json"), "w", encoding="utf-8") as f:
    json.dump(effects_summary, f, ensure_ascii=False, indent=2)


md = [
    "# H12 音频效果生成与音色处理 · 实测报告", "",
    "实现 6 个可参数化音频效果，并**量化每个效果对信号的实际改变**。", "",
    "## ① 失真（波形整形）", "",
    "> THD 在**纯正弦**上测量（吉他和弦本身含谐波，会干扰失真度量）。", "",
    "| 模式 | 驱动 | 峰值 | RMS | 峰均比 | THD |",
    "|---|---|---|---|---|---|",
]
for r in DIST_TABLE:
    md.append(f"| {r['mode']} | {r['drive']:.1f} | {r['peak']:.4f} | {r['rms']:.5f} | "
              f"{r['crest_factor']:.3f} | {r['thd']:.4f} |")
md += ["",
       "> 驱动量↑ → **峰均比下降**（波形被压平）、**THD 上升** → 这是「失真变脏」的量化表现。", "",
       "## ② 混响（合成 IR + 卷积）", "",
       "| 目标 RT60 | 湿声比 | 实测 RT60 | 左右电平差(dB) | 声道相关性 |",
       "|---|---|---|---|---|"]
for r in REV_TABLE:
    md.append(f"| {r['target_rt60']:.1f} | {r['wet']:.1f} | {r['measured_rt60_from_ir']} | "
              f"{r['stereo']['lr_level_diff_db']} | {r['stereo']['correlation']} |")
md += ["", "> 湿声比↑ → **声道相关性下降**（左右 IR 有 3ms 时间差 → 空间感增强）。", "",
       "## ③ 延迟（反馈延迟线）", "",
       "| 延迟(ms) | 反馈 | 峰值 | RMS | 峰均比 |", "|---|---|---|---|---|"]
for r in DELAY_TABLE:
    md.append(f"| {r['delay_ms']:.0f} | {r['feedback']:.1f} | {r['peak']:.4f} | {r['rms']:.5f} | {r['crest_factor']:.3f} |")
md += ["", "## ④ 声像与立体声展宽", "",
       "声像（等功率定律）：", "",
       "| pan | 左右电平差(dB) | 声道相关性 |", "|---|---|---|"]
for r in PAN_TABLE:
    lr, co = r['lr_level_diff_db'], r['correlation']
    s_lr = f"{lr:.2f}" if lr is not None else "n/a"
    s_co = f"{co:.4f}" if co is not None else "n/a"
    md.append(f"| {r['pan']:.1f} | {s_lr} | {s_co} |")
md += ["", "Mid/Side 展宽：", "",
       "| width | 左右电平差(dB) | 声道相关性 |", "|---|---|---|"]
for r in WID_TABLE:
    lr, co = r['lr_level_diff_db'], r['correlation']
    s_lr = f"{lr:.2f}" if lr is not None else "n/a"
    s_co = f"{co:.4f}" if co is not None else "n/a"
    md.append(f"| {r['width']:.1f} | {s_lr} | {s_co} |")
md += ["",
       "> **width=0 时相关性 = 1.0（完全单声道）**；width 增大 → 相关性下降 → 空间感增强。", "",
       "## ⑤ 完整吉他效果链", "",
       "配置：噪声门 → 失真(drive=8, soft, tone=0.62) → 三段EQ(-2/+1.5/+3 dB) → 合唱(0.25) → "
       "延迟(320ms, fb=0.30, mix=0.20) → 混响(RT60=1.8s, wet=0.28) → 归一化", "",
       "| 阶段 | 峰值 | RMS | 峰均比 | THD |", "|---|---|---|---|---|",
       f"| 干声 | {stats(dry_note)['peak']:.4f} | {stats(dry_note)['rms']:.5f} | "
       f"{stats(dry_note)['crest_factor']:.3f} | {thd_estimate(dry_note, f0=NOTE):.4f} |"]
for name, s, thd in stages:
    md.append(f"| {name} | {s['peak']:.4f} | {s['rms']:.5f} | {s['crest_factor']:.3f} | {thd if thd is not None else '—'} |")
md += ["",
       "## 量化曲线", "",
       "| 图 | 内容 |",
       "|---|---|",
       "| `h12_plot_distortion.png` | ① 失真：drive ↑ → THD ↑、峰均比 ↓（双 Y 轴） |",
       "| `h12_plot_reverb_decay.png` | ② 混响：Schroeder 能量衰减曲线，−60 dB 处即 RT60 |",
       "| `h12_plot_stereo.png` | ③ 等功率声像定律　④ 立体声展宽 vs 声道相关性 |",
       "",
       "## 产出", "",
       "```",
       "outputs/h12_input_note.wav / h12_input_chord.wav     输入素材",
       "outputs/h12_dist_{soft,hard}_drive{1,4,12}.wav       失真对比",
       "outputs/h12_reverb_rt{0.4,1.5,3.0}_wet{20,50}.wav    混响对比",
       "outputs/h12_delay_{120,250,450}ms_fb{20,50}.wav      延迟对比",
       "outputs/h12_chorus.wav / h12_pan_*.wav / h12_width_*.wav",
       "outputs/h12_guitar_chain_wet{,_stereo}.wav           完整效果链输出",
       "outputs/h12_plot_*.png                               量化曲线图",
       "h12_effects_report.json / .md                        全部指标",
       "```",
       ]
with open(os.path.join(HERE, "h12_effects_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print("=" * 80)
print(f"[done] 音频效果 -> {OUT}")
print(f"[done] 报告 -> h12_effects_report.md / h12_effects_report.json")
print("=" * 80)
