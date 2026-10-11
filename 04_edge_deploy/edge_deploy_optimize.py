# -*- coding: utf-8 -*-
"""
H10 · 模型轻量化与端侧部署优化

目标：把训练好的音频模型，从 PyTorch 一路优化到可端侧部署的形态，并量化每一步的收益。

对比 4 个版本：
    ① PyTorch fp32（基线）
    ② ONNX fp32（图优化 + 跨框架）
    ③ ONNX fp16（半精度）
    ④ ONNX int8 动态量化（端侧最常用）

度量：
    · 模型体积（MB）
    · 推理延迟（ms / 每次）
    · 加速比
    · 输出一致性（与 PyTorch 基线的最大差异 / 余弦相似度）
    · 精度保持（在真实音频上的分类一致性）

用法：
    python edge_deploy_optimize.py
"""
import os
import sys
import io
import time
import json
import glob
import contextlib
import tempfile

# 抑制第三方库的冗余告警，保持输出整洁（不影响功能）
#   · torch.onnx 导出器关于 dynamic_axes 的 UserWarning
#   · copyreg 的 FutureWarning（torch 内部使用）
#   · onnxruntime 量化前的 root WARNING
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx")
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    import logging as _logging
    _logging.getLogger("root").setLevel(_logging.ERROR)
except Exception:
    pass
warnings.filterwarnings("ignore", message=".*dynamic_axes.*")
warnings.filterwarnings("ignore", message=".*Please consider to run pre-processing.*")

# 抑制 torch 首次导入时的 triton 探测提示（不影响功能）
try:
    import logging as _lg
    _lg.getLogger("torch.utils.flop_counter").setLevel(_lg.ERROR)
    _lg.getLogger("torch").setLevel(_lg.ERROR)
except Exception:
    pass

# ⚠️ 关键坑：onnxruntime 的量化工具内部使用系统临时目录（%TEMP%）。
#    本机用户名含中文（伍浩然），%TEMP% 路径中的非 ASCII 字符会导致
#    中间文件路径被错误编码 → FileNotFoundError。
#    解决：在导入 onnxruntime 之前，把临时目录重定向到纯 ASCII 路径。
_ASCII_TMP = r"D:\ort_tmp"
os.makedirs(_ASCII_TMP, exist_ok=True)
os.environ["TMP"] = _ASCII_TMP
os.environ["TEMP"] = _ASCII_TMP
tempfile.tempdir = _ASCII_TMP

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import librosa
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cpu"          # 端侧部署重点看 CPU 推理
torch.manual_seed(42)
np.random.seed(42)

SR = 22050
N_MELS, N_FRAMES, N_FFT, HOP = 64, 128, 1024, 256
N_CLASSES = 4


def title(s):
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


# =====================================================================
title("1. 模型定义（音频分类网络：Conv2d 处理 log-mel）")

class AudioSpecNet(nn.Module):
    """把 log-mel 当 2D 图处理，输出分类 logits。用于演示端侧部署优化链路。"""
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 48, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(48, n_classes)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.act(self.bn1(self.conv1(x))))   # -> (B,16,32,64)
        x = self.pool(self.act(self.bn2(self.conv2(x))))   # -> (B,32,16,32)
        x = self.pool(self.act(self.bn3(self.conv3(x))))   # -> (B,48,8,16)
        x = self.gap(x).flatten(1)                         # -> (B,48)
        return self.fc(self.drop(x))


model = AudioSpecNet().eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {n_params:,}")

# =====================================================================
title("2. 准备音频样本（复用仓库里的音频）")

REPO = r"C:\Users\伍浩然\OneDrive\Desktop\ForInterview\SkillsetLab\github_repo\audio-aigc-lab"
pats = [
    os.path.join(REPO, "03_music_gen_edit", "outputs", "*.wav"),
    os.path.join(REPO, "02_data_pipeline", "data", "clean", "*.wav"),
    os.path.join(REPO, "02_data_pipeline", "data", "raw", "*.wav"),
]
files = []
for p in pats:
    files += glob.glob(p)
files = sorted(set(files))
if not files:
    files = [librosa.ex("choice"), librosa.ex("trumpet"), librosa.ex("sweetwaltz")]
print(f"找到 {len(files)} 个音频")

def wav_to_feat(y):
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    m = librosa.power_to_db(m, ref=np.max)
    if m.shape[1] < N_FRAMES:
        m = np.pad(m, ((0, 0), (0, N_FRAMES - m.shape[1])), constant_values=m.min())
    else:
        m = m[:, :N_FRAMES]
    return m.astype(np.float32)

feats = []
for f in files[:16]:
    y, _ = librosa.load(f, sr=SR, mono=True, duration=20.0)
    feats.append(wav_to_feat(y))
X = np.stack(feats)[:, None, :, :]      # (N, 1, n_mels, frames)
print(f"输入张量: {X.shape} = (样本, 通道, mel, 帧)")

dummy = torch.from_numpy(X[:1])
print(f"单样本输入: {tuple(dummy.shape)}")

# =====================================================================
title("3. ① PyTorch fp32 基线")

pt_path = os.path.join(OUT, "model_pytorch.pt")
torch.save(model.state_dict(), pt_path)
pt_size = os.path.getsize(pt_path) / 1024 / 1024

xb = torch.from_numpy(X)

def bench(fn, x, n=30, warmup=5):
    for _ in range(warmup):
        fn(x)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(x)
    return (time.perf_counter() - t0) / n * 1000      # ms per call

with torch.no_grad():
    pt_logits = model(xb).numpy()
pt_ms = bench(lambda x: model(x), xb)
print(f"  模型文件      : {pt_path.split(os.sep)[-1]}  {pt_size:.3f} MB")
print(f"  推理延迟      : {pt_ms:.3f} ms / 批次（{len(X)} 条）")
print(f"  单样本延迟    : {pt_ms/len(X):.3f} ms")
print(f"  输出形状      : {pt_logits.shape}")

# =====================================================================
title("4. ② 导出 ONNX fp32")

import onnx
import onnxruntime as ort

onnx_fp32 = os.path.join(OUT, "model_fp32.onnx")
with contextlib.redirect_stdout(io.StringIO()):      # 静音 torch.onnx 导出过程的进度输出
    torch.onnx.export(
        model, dummy,
        onnx_fp32,
        input_names=["mel"],
        output_names=["logits"],
        dynamic_axes={"mel": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=21,               # 用高版本避免 ONNX 版本转换（低版本会触发 axes 属性转换 bug）
        do_constant_folding=True,       # 常量折叠：图优化
    )
sz32 = os.path.getsize(onnx_fp32) / 1024 / 1024

sess32 = ort.InferenceSession(onnx_fp32, providers=["CPUExecutionProvider"])
onnx32_logits = sess32.run(None, {"mel": X})[0]
ms32 = bench(lambda x: sess32.run(None, {"mel": x}), X)

print(f"  文件大小      : {sz32:.3f} MB   （对比 PyTorch {pt_size:.3f} MB）")
print(f"  推理延迟      : {ms32:.3f} ms / 批次")
print(f"  相对 PyTorch  : {pt_ms/ms32:.2f}x")

def compare(a, b, name_a="A", name_b="B"):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    maxdiff = float(np.max(np.abs(a - b)))
    fa, fb = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    cos = float(np.mean([
        np.dot(fa[i], fb[i]) / (np.linalg.norm(fa[i]) * np.linalg.norm(fb[i]) + 1e-12)
        for i in range(len(fa))
    ]))
    return maxdiff, cos

d32, c32 = compare(pt_logits, onnx32_logits)
print(f"  输出一致性    : 最大差异 {d32:.2e} | 余弦相似度 {c32:.6f}")

# =====================================================================
title("5. ③ ONNX fp16（半精度）")

onnx_fp16 = os.path.join(OUT, "model_fp16.onnx")
try:
    from onnxconverter_common import float16
    m = onnx.load(onnx_fp32)
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, onnx_fp16)
    ok16 = True
except ImportError:
    ok16 = False
    print("  [跳过] 需要 onnxconverter-common: pip install onnxconverter-common")

if ok16:
    sz16 = os.path.getsize(onnx_fp16) / 1024 / 1024
    sess16 = ort.InferenceSession(onnx_fp16, providers=["CPUExecutionProvider"])
    onnx16_logits = sess16.run(None, {"mel": X})[0]
    ms16 = bench(lambda x: sess16.run(None, {"mel": x}), X)
    d16, c16 = compare(onnx32_logits, onnx16_logits)
    print(f"  文件大小      : {sz16:.3f} MB   （相对 fp32 {sz16/sz32*100:.1f}%）")
    print(f"  推理延迟      : {ms16:.3f} ms / 批次")
    print(f"  输出一致性    : 最大差异 {d16:.2e} | 余弦相似度 {c16:.6f}")

# =====================================================================
title("6. ④ ONNX int8 动态量化（端侧最常用）")

from onnxruntime.quantization import quantize_dynamic, QuantType
import shutil
import tempfile

onnx_int8 = os.path.join(OUT, "model_int8.onnx")

# ⚠️ 坑：onnxruntime 的量化工具会在输入文件同目录生成 "-inferred.onnx" 中间文件，
#    若路径含非 ASCII 字符（本机用户名为中文），onnx.load 会因编码问题失败。
#    解决：把模型复制到纯 ASCII 路径（D 盘根目录）做量化，完成后再复制回。
tmpdir = r"D:\onnx_quant_tmp"
os.makedirs(tmpdir, exist_ok=True)
tmp_in = os.path.join(tmpdir, "model_fp32.onnx")
tmp_out = os.path.join(tmpdir, "model_int8.onnx")

# 坑 1：torch 导出的 ONNX 可能把权重存成「外部数据文件」，量化工具读不了
#      → 重新导出为单文件（external_data=False）
with contextlib.redirect_stdout(io.StringIO()):      # 静音 torch.onnx 导出过程的进度输出
    torch.onnx.export(
        model, dummy, tmp_in,
        input_names=["mel"], output_names=["logits"],
        dynamic_axes={"mel": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=21, do_constant_folding=True, external_data=False,
    )
# 坑 2：形状推断可能与已有 value_info 冲突 → 去掉中间层 value_info 再量化
_m = onnx.load(tmp_in)
del _m.graph.value_info[:]
onnx.save(_m, tmp_in)

try:
    quantize_dynamic(tmp_in, tmp_out, weight_type=QuantType.QInt8)
    shutil.copyfile(tmp_out, onnx_int8)
    print(f"  [提示] 因路径含非 ASCII 字符与形状推断限制，量化在纯 ASCII 目录、去 value_info 后完成")
except Exception as ex:
    print(f"  [量化失败] {type(ex).__name__}: {ex}")
    raise

sz8 = os.path.getsize(onnx_int8) / 1024 / 1024

sess8 = ort.InferenceSession(onnx_int8, providers=["CPUExecutionProvider"])
onnx8_logits = sess8.run(None, {"mel": X})[0]
ms8 = bench(lambda x: sess8.run(None, {"mel": x}), X)
d8, c8 = compare(onnx32_logits, onnx8_logits)

print(f"  文件大小      : {sz8:.3f} MB   （相对 fp32 {sz8/sz32*100:.1f}%）")
print(f"  推理延迟      : {ms8:.3f} ms / 批次")
print(f"  相对 PyTorch  : {pt_ms/ms8:.2f}x")
print(f"  输出一致性    : 最大差异 {d8:.2e} | 余弦相似度 {c8:.6f}")

# =====================================================================
title("7. 精度保持验证（分类结果一致性）")

pt_pred = pt_logits.argmax(1)
o32_pred = onnx32_logits.argmax(1)
o8_pred = onnx8_logits.argmax(1)
agree32 = float((pt_pred == o32_pred).mean())
agree8 = float((pt_pred == o8_pred).mean())
print(f"  PyTorch vs ONNX fp32  预测一致率: {agree32*100:.1f}%")
print(f"  PyTorch vs ONNX int8  预测一致率: {agree8*100:.1f}%")
print("\n说明：本模型未训练（随机权重），此处验证的是「部署转换前后的一致性」，")
print("      即量化是否引入了破坏性误差——这是部署环节的核心验证项。")

# =====================================================================
title("8. 汇总对比表")

rows = [
    ("PyTorch fp32", pt_size, pt_ms, 1.0, "—"),
    ("ONNX fp32", sz32, ms32, pt_ms / ms32, f"{d32:.2e}"),
]
if ok16:
    rows.append(("ONNX fp16", sz16, ms16, pt_ms / ms16, f"{d16:.2e}"))
rows.append(("ONNX int8", sz8, ms8, pt_ms / ms8, f"{d8:.2e}"))

print(f"{'版本':14s} {'体积(MB)':>10s} {'延迟(ms)':>10s} {'相对加速':>9s} {'输出最大差异':>14s}")
print("-" * 66)
for name, size, ms, sp, diff in rows:
    print(f"{name:14s} {size:>10.3f} {ms:>10.3f} {sp:>9.2f}x {diff:>14s}")

best_small = min(rows, key=lambda r: r[1])
best_fast = min(rows, key=lambda r: r[2])
print()
print(f"  体积最小: {best_small[0]}  ({best_small[1]:.3f} MB)")
print(f"  速度最快: {best_fast[0]}  ({best_fast[2]:.3f} ms)")

# =====================================================================
# 上表即为最终结果；以下写入 JSON / Markdown 报告

report = {
    "model_params": n_params,
    "input_shape": list(X.shape[1:]),
    "results": [
        {"name": n, "size_mb": round(s, 4), "latency_ms": round(m, 4),
         "speedup": round(sp, 3), "max_diff": d}
        for n, s, m, sp, d in rows
    ],
    "consistency": {"pytorch_vs_onnx_fp32": round(agree32, 4),
                    "pytorch_vs_onnx_int8": round(agree8, 4)},
    "size_reduction_int8_pct": round(sz8 / sz32 * 100, 2),
}

with open(os.path.join(HERE, "h10_deploy_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

md = ["# H10 模型轻量化与端侧部署优化 · 实测报告", "",
      f"- 模型参数量：**{n_params:,}**",
      f"- 输入形状：`{list(X.shape[1:])}`（单通道 log-mel，{N_MELS}×{N_FRAMES}）",
      f"- 测试样本数：{len(X)}",
      f"- 推理设备：CPU（端侧场景）", "",
      "## 对比结果", "",
      "| 版本 | 体积 (MB) | 延迟 (ms/批次) | 相对 PyTorch 加速 | 输出最大差异 |",
      "|---|---|---|---|---|"]
for n, s, m, sp, d in rows:
    md.append(f"| {n} | {s:.3f} | {m:.3f} | {sp:.2f}x | {d} |")
md += ["",
       "## 精度保持", "",
       f"- PyTorch vs ONNX fp32 预测一致率：**{agree32*100:.1f}%**",
       f"- PyTorch vs ONNX int8 预测一致率：**{agree8*100:.1f}%**", "",
       ]
with open(os.path.join(HERE, "h10_deploy_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print("=" * 78)
print(f"[done] 模型文件 -> {OUT}")
print(f"[done] 报告 -> h10_deploy_report.md / h10_deploy_report.json")
print("=" * 78)
