# H8 实测记录表 · MusicGen-small

- 设备：NVIDIA GeForce RTX 3050 Laptop GPU（显存 4 GB，实际可用约 2.8 GB）
- dtype：torch.float16
- 模型参数量：586.9M
- 模型加载耗时：139.3s
- 输出采样率：32000 Hz
- 生成配置：max_new_tokens=200（约 4 秒），guidance_scale=3.0

| # | Prompt | 时长(s) | 生成耗时(s) | RTF | 显存峰值(GB) | 峰值 | RMS | 文件 |
|---|---|---|---|---|---|---|---|---|
| 1 | lo-fi hip hop with warm piano, soft vinyl crackle, mellow drums | 3.94 | 6.2 | 1.57 | 1.25 | 0.446 | 0.0478 | h8_prompt1.wav |
| 2 | cinematic orchestral strings with slow build up and deep bass | 3.94 | 4.6 | 1.18 | 1.25 | 0.174 | 0.0456 | h8_prompt2.wav |
| 3 | gentle acoustic guitar fingerpicking, calm and intimate | 3.94 | 4.8 | 1.21 | 1.25 | 0.539 | 0.0955 | h8_prompt3.wav |

> RTF（real-time factor）= 生成耗时 / 音频时长。RTF < 1 表示比实时快。
> 生成配置属于**保守设置**：受 4 GB 显存限制，仅生成约 4 秒片段；
> MusicGen-small 论文与官方 demo 通常在更长时长（10–30 秒）下评估。