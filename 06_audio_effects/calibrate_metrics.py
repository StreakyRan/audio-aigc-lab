# -*- coding: utf-8 -*-
"""诊断：验证 THD 与 RT60 的正确测量方式。

问题 1：失真后 THD 不升反降 —— 用纯正弦测试，排除输入信号本身谐波的干扰
问题 2：混响实测 RT60 与目标无关（都≈2.7s）—— 因为测量时"干声"占比过大，
        应从 IR 本身或"纯湿声"（wet=1.0）测量
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import scipy.signal as sps

SR = 44100
rng = np.random.default_rng(42)


def thd(y, sr=SR, f0=110.0, n_harm=10):
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    bw = max(3.0, f0 * 0.06)
    def be(f):
        m = (freqs >= f - bw) & (freqs <= f + bw)
        return float(np.sum(Y[m] ** 2)) if m.any() else 0.0
    f_e = be(f0)
    h_e = sum(be(f0 * k) for k in range(2, n_harm + 1))
    return None if f_e <= 0 else round(float(np.sqrt(h_e / f_e)), 4)


def rt60_schroeder(y, sr=SR):
    e = (y ** 2).astype(np.float64)
    cum = np.cumsum(e[::-1])[::-1]
    if cum[0] <= 0:
        return None
    db = 10 * np.log10(cum / cum[0] + 1e-12)
    def t(level):
        idx = np.where(db <= level)[0]
        return idx[0] / sr if len(idx) else None
    t5, t25 = t(-5), t(-25)
    if t5 is None or t25 is None or t25 <= t5:
        return None
    return round(float((t25 - t5) * 60 / 20), 3)


print("=" * 70)
print("诊断 1：THD 是否随驱动量上升（用纯正弦，排除输入谐波干扰）")
print("=" * 70)
t = np.linspace(0, 2.0, int(SR * 2.0), endpoint=False)
sine = np.sin(2 * np.pi * 110.0 * t) * 0.8

print(f"{'驱动':>6} {'soft-THD':>10} {'hard-THD':>10}")
print("-" * 30)
print(f"{'0(纯正弦)':>6} {thd(sine):>10} {thd(sine):>10}")
for drive in (1.0, 2.0, 4.0, 8.0, 16.0):
    soft = np.tanh(sine * drive)
    soft = soft / (np.max(np.abs(soft)) + 1e-12) * 0.9
    hard = np.clip(sine * drive, -1, 1)
    hard = hard / (np.max(np.abs(hard)) + 1e-12) * 0.9
    print(f"{drive:>6.1f} {thd(soft):>10} {thd(hard):>10}")

print("""
预期：驱动量↑ → THD↑（失真确实引入了更多谐波）
若纯正弦的 THD 随驱动上升，说明**是输入信号本身的谐波干扰了之前的测量**。
""")

print("=" * 70)
print("诊断 2：RT60 应该从脉冲响应本身测量（而非含干声的混合输出）")
print("=" * 70)


def make_ir(rt60_sec, sr=SR, pre_delay=0.02, damping=0.4):
    n = int(sr * (rt60_sec + 0.2))
    tt = np.arange(n) / sr
    env = np.exp(-6.9 * tt / max(rt60_sec, 1e-3))
    ir = rng.normal(0, 1, n) * env
    pd = int(pre_delay * sr)
    if pd > 0:
        ir[:pd] = 0
    b, a = sps.butter(1, 6000 / (sr / 2), btype="low")
    lp = sps.lfilter(b, a, ir)
    ir = (1 - damping) * ir + damping * lp
    return (ir / (np.max(np.abs(ir)) + 1e-12)).astype(np.float32)


print(f"{'目标 RT60':>10} {'IR 直接实测':>12}")
print("-" * 26)
for target in (0.4, 1.0, 1.5, 3.0):
    ir = make_ir(target)
    print(f"{target:>10.1f} {str(rt60_schroeder(ir)):>12}")

print("""
预期：IR 直接测量应接近目标值（±20% 内）。
之前测的是「干声 + 湿声」的混合信号，干声能量占主导 → 测出的是整段长度而非混响尾音。
正确做法：从 IR 本身测量 RT60，或对纯湿声（wet=1.0）测量。
""")

print("=" * 70)
print("诊断 3：pan 到极限位置时，相关性与电平差应显式处理 None")
print("=" * 70)
print("pan=-1.0（全左）时右声道为静音 → 相关性无定义，需返回 None 并跳过格式化")
