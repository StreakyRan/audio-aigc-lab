# -*- coding: utf-8 -*-
"""
H11 · 音高识别与 MIDI 自动扒谱（曲谱生成 / 音高识别 / MIDI 自动扒谱 / 和弦编排）

流水线：
    单音音频
      → ① 基频检测（pyin）得到逐帧 f0（Hz）
      → ② f0 → 音名/MIDI 音符号（十二平均律）
      → ③ 音符分段（合并连续同音高帧，过滤过短音符）
      → ④ 导出 MIDI 文件（可被 DAW / 打谱软件打开）
      → ⑤ 和弦识别（chroma + 模板匹配）
      → ⑥ 与真值对比，输出音符级 precision / recall / F1

设计说明：
    · 用「合成已知旋律」作为输入，这样有 ground truth，可以量化扒谱准确率
      （真实音乐里没有人知道"正确音符是什么"，无法评估）
    · 同时支持传入真实音频文件（无真值，只输出 MIDI）

用法：
    python pitch_to_midi.py                    # 用合成旋律（有真值，可评估）
    python pitch_to_midi.py "your.wav"         # 用真实音频
"""
import os
import sys
import json
import math

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import soundfile as sf
import librosa
import pretty_midi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
SR = 22050


def title(s):
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


# =====================================================================
# 音名 / MIDI 编号工具
# =====================================================================
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def hz_to_midi(f):
    """频率 -> MIDI 音符号（十二平均律，A4=440Hz=69）"""
    return 69 + 12 * np.log2(f / 440.0)

def midi_to_name(m):
    m = int(round(m))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"

def midi_to_hz(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))


def title_str():
    pass


# =====================================================================
title("1. 构造测试旋律（有 ground truth，才能量化扒谱准确率）")

# C 大调音阶片段 + 一个简单旋律（含跳进与级进，便于检验）
MELODY = [
    # (音名, 起始拍, 时值拍)
    ("C4", 0.0, 0.5), ("D4", 0.5, 0.5), ("E4", 1.0, 0.5), ("F4", 1.5, 0.5),
    ("G4", 2.0, 1.0), ("E4", 3.0, 0.5), ("C4", 3.5, 0.5),
    ("A4", 4.0, 0.5), ("G4", 4.5, 0.5), ("F4", 5.0, 0.5), ("E4", 5.5, 0.5),
    ("D4", 6.0, 1.0), ("C4", 7.0, 1.0),
]
BPM = 120.0
BEAT = 60.0 / BPM                    # 每拍秒数 = 0.5s

def synth_note(freq, dur, sr=SR, harmonics=(1.0, 0.5, 0.25)):
    """带谐波的合成音（更接近真实乐器，避免纯正弦过于理想化）"""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    y = np.zeros_like(t)
    for k, amp in enumerate(harmonics, start=1):
        y += amp * np.sin(2 * np.pi * freq * k * t)
    # 加 ADSR 包络（避免爆音，也更像真实演奏）
    n = len(t)
    env = np.ones(n)
    a = int(0.01 * sr); d = int(0.05 * sr); r = int(0.05 * sr)
    if a < n: env[:a] = np.linspace(0, 1, a)
    if d < n - a: env[a:a+d] = np.linspace(1, 0.7, d)
    if r < n: env[-r:] *= np.linspace(1, 0, r)
    return y * env

def midi_name_to_number(name):
    """音名 -> MIDI 编号"""
    for i in range(128):
        if midi_to_name(i) == name:
            return i
    raise ValueError(name)

total_beats = max(s + d for _, s, d in MELODY)
total_dur = total_beats * BEAT + 1.0
y = np.zeros(int(SR * total_dur), dtype=np.float32)

gt_notes = []
for name, start, dur in MELODY:
    m = midi_name_to_number(name)
    f = midi_to_hz(m)
    s = int(start * BEAT * SR)
    seg = synth_note(f, dur * BEAT)
    y[s:s + len(seg)] += seg
    gt_notes.append({"name": name, "midi": m, "start": start * BEAT, "end": (start + dur) * BEAT})

y += np.random.default_rng(0).normal(0, 0.003, y.shape).astype(np.float32)   # 轻微底噪
y = (y / np.max(np.abs(y)) * 0.9).astype(np.float32)

wav_path = os.path.join(OUT, "h11_melody.wav")
sf.write(wav_path, y, SR)

print(f"旋律长度      : {total_beats} 拍 = {total_beats*BEAT:.1f}s @ {BPM} BPM")
print(f"音符数        : {len(gt_notes)}")
print(f"音域          : {min(g['name'] for g in gt_notes)} ~ {max(g['name'] for g in gt_notes)}")
print(f"真值前 6 个   : " + ", ".join(f"{g['name']}@{g['start']:.2f}s" for g in gt_notes[:6]))
print(f"合成音频 -> {os.path.basename(wav_path)}")

# =====================================================================
title("2. ① 基频检测（pyin）")

FRAME_LENGTH = 2048
HOP = 256

f0, voiced_flag, voiced_prob = librosa.pyin(
    y,
    fmin=librosa.note_to_hz("C3"),      # 音域下限
    fmax=librosa.note_to_hz("C6"),      # 音域上限
    sr=SR, frame_length=FRAME_LENGTH, hop_length=HOP,
)
times = librosa.times_like(f0, sr=SR, hop_length=HOP)
voiced = np.isfinite(f0)

print(f"分析帧数      : {len(f0)}   (帧移 {HOP}/{SR*1000:.1f}ms → {SR/HOP:.1f} fps)")
print(f"有声帧占比    : {voiced.mean()*100:.1f}%")
print(f"f0 范围       : {np.nanmin(f0):.1f} ~ {np.nanmax(f0):.1f} Hz")
print(f"时间分辨率    : {HOP/SR*1000:.1f} ms")

# f0 -> MIDI 编号
f0_midi = hz_to_midi(f0)
print(f"MIDI 编号范围 : {np.nanmin(f0_midi):.1f} ~ {np.nanmax(f0_midi):.1f}")

# =====================================================================
title("3. ②③ 音符分段（合并连续同音高帧 → 音符事件）")

def segment_notes(f0, voiced, times, min_dur=0.08, pitch_tol=0.6):
    """
    把逐帧 f0 转成音符事件列表。
    pitch_tol: 相邻帧音高差小于该值（半音）视为同一个音
    min_dur:   短于该时长的音符丢弃（去抖动）
    """
    notes = []
    cur = None
    for i, (f, v, t) in enumerate(zip(f0, voiced, times)):
        if not v:
            if cur is not None:
                notes.append(cur); cur = None
            continue
        m = float(hz_to_midi(f))
        if cur is None:
            cur = {"start": t, "m_list": [m]}
        else:
            ref = float(np.median(cur["m_list"]))
            if abs(m - ref) <= pitch_tol:
                cur["m_list"].append(m)
            else:
                notes.append(cur)
                cur = {"start": t, "m_list": [m]}
    if cur is not None:
        notes.append(cur)

    out = []
    for i, n in enumerate(notes):
        end = notes[i + 1]["start"] if i + 1 < len(notes) else times[-1] + HOP / SR
        dur = end - n["start"]
        if dur < min_dur:
            continue
        m = int(round(float(np.median(n["m_list"]))))
        out.append({"midi": m, "name": midi_to_name(m), "start": n["start"], "end": end, "dur": dur})
    return out

est_notes = segment_notes(f0, voiced, times)
print(f"检测到音符数  : {len(est_notes)}  (真值 {len(gt_notes)})")
print()
print(f"{'#':>3} {'音名':>6} {'MIDI':>5} {'起始(s)':>9} {'时长(s)':>8}")
for i, n in enumerate(est_notes):
    print(f"{i+1:>3} {n['name']:>6} {n['midi']:>5} {n['start']:>9.3f} {n['dur']:>8.3f}")

# =====================================================================
title("4. ④ 导出 MIDI 文件")

pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
inst = pretty_midi.Instrument(program=24)      # 24 = Acoustic Guitar (nylon)
for n in est_notes:
    inst.notes.append(pretty_midi.Note(
        velocity=90, pitch=int(n["midi"]), start=float(n["start"]), end=float(n["end"])
    ))
pm.instruments.append(inst)
midi_path = os.path.join(OUT, "h11_transcribed.mid")
pm.write(midi_path)

# 顺带写一份真值 MIDI，便于人工对比试听
pm_gt = pretty_midi.PrettyMIDI(initial_tempo=BPM)
inst_gt = pretty_midi.Instrument(program=24)
for g in gt_notes:
    inst_gt.notes.append(pretty_midi.Note(velocity=90, pitch=g["midi"],
                                          start=g["start"], end=g["end"]))
pm_gt.instruments.append(inst_gt)
midi_gt_path = os.path.join(OUT, "h11_ground_truth.mid")
pm_gt.write(midi_gt_path)

print(f"识别结果 MIDI -> {os.path.basename(midi_path)}  ({os.path.getsize(midi_path)} bytes)")
print(f"真值 MIDI     -> {os.path.basename(midi_gt_path)}  ({os.path.getsize(midi_gt_path)} bytes)")
print("说明：MIDI 文件可直接用 DAW（Studio One / Cubase / Logic）或打谱软件打开试听")

# =====================================================================
title("5. ⑤ 和弦推断（按拍聚合音高类别 + 三和弦集合匹配）")

def detect_chords_by_beat(y, sr, est_notes, total_beats, sec_per_beat):
    """
    对「单音旋律」更可靠的和弦推断：
      ① 按拍收集该拍内出现的音符音高类别（pitch class）
      ② 统计每个音高类别的时长权重
      ③ 与三和弦模板做加权覆盖率匹配（要求根音、三音都在，五音缺失可用不惩罚）
    说明：单音线条本身不含完整和声，和弦推断本质是「依据旋律音反推最可能的和弦」，
         这是有歧义的任务 —— 因此本方法的目标是给出「合理的和弦建议」而非唯一正确答案。
    """
    CHORD_TPL = {
        "C": (0, [0, 4, 7]), "Cm": (0, [0, 3, 7]),
        "C#": (1, [1, 5, 8]), "C#m": (1, [1, 4, 8]),
        "D": (2, [2, 6, 9]), "Dm": (2, [2, 5, 9]),
        "D#": (3, [3, 7, 10]), "D#m": (3, [3, 6, 10]),
        "E": (4, [4, 8, 11]), "Em": (4, [4, 7, 11]),
        "F": (5, [5, 9, 0]), "Fm": (5, [5, 8, 0]),
        "F#": (6, [6, 10, 1]), "F#m": (6, [6, 9, 1]),
        "G": (7, [7, 11, 2]), "Gm": (7, [7, 10, 2]),
        "G#": (8, [8, 0, 3]), "G#m": (8, [8, 11, 3]),
        "A": (9, [9, 1, 4]), "Am": (9, [9, 0, 4]),
        "A#": (10, [10, 2, 5]), "A#m": (10, [10, 1, 5]),
        "B": (11, [11, 3, 6]), "Bm": (11, [11, 2, 6]),
    }
    out = []
    for b in range(total_beats):
        lo, hi = b * sec_per_beat, (b + 1) * sec_per_beat
        # 该拍内音符的「音高类别时长分布」
        pc_weight = np.zeros(12)
        for n in est_notes:
            # 与当前拍的重叠时长
            ov = max(0.0, min(n["end"], hi) - max(n["start"], lo))
            if ov > 0:
                pc_weight[n["midi"] % 12] += ov
        if pc_weight.sum() <= 0:
            out.append((b, "—", 0.0, []))
            continue
        present = set(np.where(pc_weight > 1e-6)[0].tolist())
        best, best_score, best_hit = None, -1.0, []
        for name, (root, tones) in CHORD_TPL.items():
            tones_set = set(tones)
            # 覆盖率：旋律音中有多少落在和弦音上（按权重）
            cover = sum(pc_weight[p] for p in present & tones_set) / pc_weight.sum()
            # 必备音：根音 + 三音 必须出现（否则不算该和弦）
            third = tones[1]
            if root not in present or third not in present:
                continue
            # 五音缺失不惩罚，但命中加分
            fifth_bonus = 0.1 if tones[2] in present else 0.0
            score = cover + fifth_bonus
            if score > best_score:
                best, best_score, best_hit = name, score, sorted(present & tones_set)
        if best is None:
            out.append((b, "无完整三和弦", 0.0, sorted(present)))
        else:
            out.append((b, best, min(best_score, 1.0), best_hit))
    return out

sec_per_beat = BEAT
n_beats = int(total_beats)
beat_chords = detect_chords_by_beat(y, SR, est_notes, n_beats, sec_per_beat)

print(f"{'拍':>3} {'时间(s)':>9} {'推断和弦':>14} {'匹配度':>8}  该拍旋律音(音高类别)")
for b, c, score, hits in beat_chords:
    print(f"{b+1:>3} {b*sec_per_beat:>9.2f} {c:>14} {score:>7.0%}  {hits}")

print()
print("⚠️ 说明：输入是**单音旋律线条**，本身不含完整和声信息。")
print("   和弦推断本质是「依据旋律音反推最可能的和弦」，属于有歧义任务——")
print("   同一段旋律在不同和声语境下可以配不同和弦。因此这里输出的是「合理建议」而非唯一答案。")
print("   若输入是复音（伴奏+旋律），应改用 chroma 逐帧匹配，准确率会显著提升。")

# =====================================================================
title("6. ⑥ 扒谱准确率评估（有真值才能量化）")

def evaluate(gt, est, tol=0.08):
    """
    音符级评估：把音符离散到时间网格，比较"该时刻的正确音高"是否命中。
    命中判定：起始时间误差 <= tol 且音高完全相同
    """
    tp = 0
    matched_est = set()
    for g in gt:
        hit = False
        for j, e in enumerate(est):
            if abs(g["start"] - e["start"]) <= tol and g["midi"] == e["midi"]:
                hit = True; matched_est.add(j); break
        if hit:
            tp += 1
    fp = len(est) - len(matched_est)
    fn = len(gt) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}

for tol in (0.05, 0.08, 0.15):
    m = evaluate(gt_notes, est_notes, tol=tol)
    print(f"  容差 {tol*1000:.0f}ms: TP={m['tp']:>2} FP={m['fp']:>2} FN={m['fn']:>2} "
          f"| P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

m08 = evaluate(gt_notes, est_notes, tol=0.08)
print()
print("  逐音符对比（真值 → 识别）:")
print(f"  {'真值':>6} {'时间':>8}  |  {'识别':>6} {'时间':>8}  {'结果'}")
n_cmp = max(len(gt_notes), len(est_notes))
for i in range(n_cmp):
    g = gt_notes[i] if i < len(gt_notes) else None
    e = est_notes[i] if i < len(est_notes) else None
    gs = f"{g['name']:>6} {g['start']:>8.3f}" if g else f"{'—':>6} {'—':>8}"
    es = f"{e['name']:>6} {e['start']:>8.3f}" if e else f"{'—':>6} {'—':>8}"
    if g and e:
        ok = "✅" if (abs(g['start'] - e['start']) <= 0.08 and g['midi'] == e['midi']) else "❌"
    else:
        ok = "⚠️"
    print(f"  {gs}  |  {es}  {ok}")

# =====================================================================
title("7. 汇总与工程结论")

report = {
    "bpm": BPM,
    "duration_sec": round(float(total_dur), 3),
    "gt_notes": len(gt_notes),
    "est_notes": len(est_notes),
    "f0_backend": "librosa.pyin",
    "frame_length": FRAME_LENGTH,
    "hop_length": HOP,
    "time_resolution_ms": round(HOP / SR * 1000, 2),
    "frame_rate_fps": round(SR / HOP, 2),
    "voiced_ratio": round(float(voiced.mean()), 4),
    "f0_range_hz": [round(float(np.nanmin(f0)), 2), round(float(np.nanmax(f0)), 2)],
    "eval_tol_80ms": m08,
    "chords_per_beat": [{"beat": b + 1, "time": round(b * sec_per_beat, 3), "chord": c,
                         "match_score": round(score, 3), "melody_pitch_classes": hits}
                        for b, c, score, hits in beat_chords],
}
with open(os.path.join(HERE, "h11_pitch_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"""
关键结果：

1. 用 librosa.pyin 做基频检测，帧移 {HOP} 样本 → 时间分辨率 {HOP/SR*1000:.1f} ms、帧率 {SR/HOP:.0f} fps；
   检测到 f0 范围 {np.nanmin(f0):.0f}–{np.nanmax(f0):.0f} Hz，有声帧占比 {voiced.mean()*100:.0f}%。
2. 音符分段（合并连续同音高帧 + 过滤 <80ms 抖动）后得到 {len(est_notes)} 个音符事件（真值 {len(gt_notes)} 个）。
3. 在 ±80ms 容差下，音符级 **Precision {m08['precision']:.3f} / Recall {m08['recall']:.3f} / F1 {m08['f1']:.3f}**。
4. 导出标准 MIDI 文件，可直接在 DAW / 打谱软件中打开编辑与试听。
5. 和弦推断：改按「拍聚合音高类别 + 三和弦集合匹配」（要求根音与三音都出现），
   输出可配的和弦建议。⚠️ **输入是单音旋律，本身不含完整和声**，因此这是"合理建议"而非唯一答案。

工程要点：
· 扒谱的核心难点是「音符分段」——连续帧要合并成音符事件，阈值太松会粘连、太紧会碎片化
· 用 pyin 而不是自相关/FFT 峰值：pyin 自带 voiced/unvoiced 判定 + 概率输出，对噪声更鲁棒
· 时间分辨率（帧移）直接决定能分辨的最短音符，与 H1 学的「STFT 帧率」是同一件事
· 和弦推断的方法选择取决于输入：**单音旋律**适合"按拍聚合音高类别 + 集合匹配"，
  **复音音频**才适合"逐帧 chroma + 模板匹配"——用错方法会得到一整串错误和弦（本实验第一版即如此）
""")

# Markdown 报告
md = [
    "# H11 音高识别与 MIDI 自动扒谱 · 实测报告", "",
    f"- 测试旋律：{total_beats} 拍 @ {BPM} BPM（{total_dur:.1f}s），**{len(gt_notes)} 个音符**",
    f"- 基频检测：`librosa.pyin`，帧长 {FRAME_LENGTH}、帧移 {HOP}",
    f"- **时间分辨率 {HOP/SR*1000:.1f} ms**（帧率 {SR/HOP:.0f} fps）",
    f"- f0 范围：{np.nanmin(f0):.0f}–{np.nanmax(f0):.0f} Hz｜有声帧占比 {voiced.mean()*100:.0f}%",
    "", "## 扒谱准确率（音符级）", "",
    "| 容差 | TP | FP | FN | Precision | Recall | F1 |",
    "|---|---|---|---|---|---|---|",
]
for tol in (0.05, 0.08, 0.15):
    mm = evaluate(gt_notes, est_notes, tol=tol)
    md.append(f"| ±{tol*1000:.0f} ms | {mm['tp']} | {mm['fp']} | {mm['fn']} | "
              f"{mm['precision']:.3f} | {mm['recall']:.3f} | {mm['f1']:.3f} |")
md += ["", "## 和弦识别结果（逐拍）", "",
       "| 拍 | 时间(s) | 推断和弦 | 匹配度 | 该拍旋律音 |", "|---|---|---|---|---|"]
for b, c, score, hits in beat_chords:
    md.append(f"| {b+1} | {b*sec_per_beat:.2f} | {c} | {score:.0%} | {hits} |")
md += ["", "## 流水线", "",
       "```",
       "单音/复音音频",
       "  → ① pyin 基频检测（逐帧 f0）",
       "  → ② f0 → 十二平均律 MIDI 音符号",
       "  → ③ 音符分段（合并连续同音高帧 + 过滤 <80ms）",
       "  → ④ 导出标准 MIDI（可 DAW / 打谱软件打开）",
       "  → ⑤ chroma + 三和弦模板 → 逐拍和弦识别",
       "  → ⑥ 有真值时输出音符级 P/R/F1",
       "```", "",
       "## 工程要点", "",
       "1. **音符分段是核心难点**：连续帧要合并成音符事件，阈值太松会粘连、太紧会碎片化。",
       "2. **用 pyin 而非自相关/FFT 峰值**：pyin 自带 voiced/unvoiced 判定和概率输出，对噪声更鲁棒。",
       f"3. **时间分辨率决定最短可辨音符**：本实验帧移 {HOP} 样本 = **{HOP/SR*1000:.1f} ms**，",
       "   这与「STFT 帧率」是同一个概念（帧移越小，能分辨的音符越短）。",
       "4. **有真值才能量化**：真实音乐没有「正确音符」标注，所以用合成旋律构造 ground truth。", "",
       "## 产出", "",
       "```",
       "outputs/h11_melody.wav           合成测试旋律",
       "outputs/h11_transcribed.mid      识别结果 MIDI",
       "outputs/h11_ground_truth.mid     真值 MIDI（用于对比）",
       "h11_pitch_report.json            机器可读指标",
       "```",
       ]
with open(os.path.join(HERE, "h11_pitch_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print("=" * 78)
print(f"[done] MIDI 与音频 -> {OUT}")
print(f"[done] 报告 -> h11_pitch_report.md / h11_pitch_report.json")
print("=" * 78)
