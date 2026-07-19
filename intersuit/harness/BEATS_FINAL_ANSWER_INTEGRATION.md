# BEATs 最终回答链路接入记录

日期：2026-07-19（Asia/Shanghai）

## 范围

第一版只实现：

```text
固定 1 秒音频窗口
-> 冻结 BEATs
-> 可训练 audio projector
-> 固定顺序线性映射到视频帧
-> 可训练 zero-init residual fusion
-> M4
```

本版不调用 audio event detector、learned temporal aligner、learned confidence
gate、动态窗口或 MUG。旧 `aligned_gated` 路径仍是默认值；只有显式设置
`AS_M4_FUSION_MODE=beats_simple_residual` 才启用新链路。

## 本地依赖锁

- BEATs checkpoint：
  `intersuit/checkpoints/BEATs_iter3_plus_AS2M.pt`
- BEATs checkpoint SHA256：
  `d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34`
- 本地源码：
  `third_party/OmniMMI/baselines/videollama2/model/beats/BEATs.py`
- 自动下载：禁止。

BEATs 外部模型不注册进 M4 module tree，不会在每个 M4 checkpoint 中重复保存
345 MB 权重。注册、保存和训练的 scene-audio 参数只有：

```text
scene_audio_encoder.audio_projector.weight
scene_audio_encoder.audio_projector.bias
```

BEATs 本体始终保持 `requires_grad=False` 和 `eval()`。训练入口只允许：

```text
MM_TUNABLE_PARTS=scene_audio_projector,streaming_av_fusion
```

对 BEATs 使用 `MM_TUNABLE_PARTS=scene_audio_encoder` 会硬失败，防止意外解冻。

## 数据路径

训练数据集现在支持：

- `video_path`：作为 `video` 的路径别名；
- `scene_audio_path`：直接从音频文件或视频音轨解码；
- `scene_audio_sample_rate`；
- `scene_audio_window_sec`；
- `scene_audio_hop_sec`。

实测首条冻结样本被解码为 `111 x 16000` 音频窗口和 `111 x 2` 时间戳，
所有值有限。

冻结 300 条只用于评测，禁止用于训练。`run_as_m4_beats_stage.sh` 要求显式提供
`DATA_PATH`，并从仓库根和 `intersuit/` 两种相对路径解析方式检查是否误指向
`frozen_eval.json`；命中时退出码为 2。

当前本地只有少量 AS-M4 smoke manifest 含场景音频，没有可用于正式训练且与
冻结 300 条明确隔离的训练 manifest。因此本轮没有启动 projector/fusion 训练，
也没有用评测答案训练模型。

## 验证

CPU 与代码检查：

- Python `py_compile` 通过；
- 三个 Shell 启动脚本 `bash -n` 通过；
- scene encoder、fusion integration、prediction runner、dataset collator、
  model builder 和 audio stream 相关回归共 `68 passed`；
- 真实 BEATs CPU 两窗口前向输出有限，projector 反向梯度有限；
- 真实 BEATs bf16 GPU 两窗口前向/反向通过；
- bf16 GPU 注册参数只有 projector 两个参数，BEATs 保持冻结。

真实最终回答链路 canary 使用原始 12k M4 和冻结样本 `avut_0663`：

| 条件 | 输出 | BEATs/audio norm | gate | residual/video |
|---|---|---:|---:|---:|
| 强制 gate=0 | `A` | 54.4454 | 0 | 0 |
| 固定 gate=1、zero-init fusion | `A` | 54.4454 | 1 | 0 |

两条件输出完全一致，诊断均为有限值，并明确记录：

```text
fusion_mode=beats_simple_residual
dynamic_alignment_enabled=false
learned_gate_enabled=false
```

运行后五张 GPU 均回到约 11 MiB。

## 后续合法顺序

1. 准备与冻结 300 条样本 ID 和视频 ID 均无重叠的 scene-audio 训练 manifest。
2. 先运行 2-step canary，确认只有 projector/fusion 获得有限非零梯度。
3. 再运行 20-step canary，确认 loss、grad norm、显存和保存/加载正常。
4. 训练完成后固定 20～30 条 smoke，比较关闭音频、正确音频、静音和错配音频。
5. 只有正确音频新增答对多于新增答错，才在同一冻结 300 条上复测。

启动形式：

```bash
DATA_PATH=/absolute/path/to/non_overlapping_scene_audio_train.json \
bash scripts/run_as_m4_beats_stage.sh 12k-smoke
```
