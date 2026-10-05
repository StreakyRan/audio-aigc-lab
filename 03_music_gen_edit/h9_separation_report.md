# H9 音乐源分离实测报告 · Demucs v4 (htdemucs)

- 输入：admiralbob77_-_Choice_-_Drum-bass.ogg（25.03s @ 44100 Hz）
- 模型：`htdemucs`，参数量 **42.0M**
- 设备：NVIDIA GeForce RTX 3050 Laptop GPU（可用显存 3.23 GB / 共 4.00 GB）
- 加载耗时：16.9s
- **分离耗时：2.58s（音频时长 25.03s，约 9.70× 实时）**
- **显存峰值：0.59 GB**

## 各 stem 统计

| stem | RMS | Peak | 能量占比 |
|---|---|---|---|
| drums | 0.06726 | 0.947 | 15.8% |
| bass | 0.15121 | 0.949 | 80.0% |
| other | 0.03332 | 0.877 | 4.0% |
| vocals | 0.00785 | 0.942 | 0.2% |

## 重建保真度

把 4 条 stem 相加与原始混音比较：**MAE = 0.01455，SNR = 12.69 dB**。

> 源分离是**近似分解**：stems 之和 ≈ 原始混音但不严格相等（存在分离残差），
> 这正是后续做 stem 级编辑时会产生**相位/抵消问题**的根源。

## 输出文件

```
outputs/stems/choice_drumbass/
├── mixture.wav   （原始混音，归一化）
├── vocals.wav
├── drums.wav
├── bass.wav
└── other.wav
```