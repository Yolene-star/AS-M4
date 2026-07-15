---
title: AS-M4 Codex Implementation Playbook
language: zh-CN
audience: Codex / 开发者 / 研究复现人员
status: implementation-ready
version: 1.1
date: 2026-07-16
project_name: AS-M4 (Audio-Sensitive M4)
source_repositories:
  m4:
    url: https://github.com/OmniMMI/M4
    pinned_commit: 574d2244576190cd150cd28aaaa8dbd2ea7502e6
  omnimmi:
    url: https://github.com/OmniMMI/OmniMMI
    pinned_commit: bd1f8460a39b821e1c5c830b710c049cd324ff2d
---

# AS-M4：面向 Codex 的分阶段实现与实验手册

## 0. 本文件如何使用

本文件不是论文摘要，也不是只供人阅读的概念说明。它是交给 Codex 执行的工程协议，规定：

- 为什么要修改 M4；
- 每个阶段允许修改哪些文件；
- 每个模块的输入、输出和职责；
- 每个 Harness（实验支架）怎样构造、怎样运行；
- 每个阶段必须满足什么条件才能进入下一阶段；
- 哪些数据可直接使用现有标注，哪些数据需要自动增强，哪些结论需要少量人工复核；
- 失败时退回到哪里，以及不得用什么方式掩盖失败。

Codex 执行本项目时，必须遵循以下状态机：

```text
P0 -> P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P8 -> P9 -> P10 -> P11
```

除非本文件明确允许，否则不得跳过阶段。每个阶段结束时必须：

1. 运行该阶段的 Harness；
2. 保存配置、日志、指标和失败样本；
3. 对照 Exit Gate（出口条件）；
4. 只有全部硬性条件通过，才能进入下一阶段；
5. 若失败，先修复当前阶段，不得通过修改测试集、删除失败样本或降低指标定义来“通过”。

每一次代码或脚本修改都必须绑定一个最小验证闭环，不能连续修改多个模块后再统一测试。每次修改后至少完成：

```text
1. 静态检查：py_compile、import 或 rg 检查；
2. Smoke Test：最小样本/最小入口能跑通；
3. Harness：验证该模块核心行为。
```

Codex 在执行每个阶段前必须先说明修改目的、实现原理、涉及文件、是否影响 checkpoint、计划 smoke/harness 和失败回退方式；得到确认后再执行。

---

# 1. 项目目标、边界和核心判断

## 1.1 项目目标

将原 M4 从以“流式视频帧 + 用户问题文字/语音”为中心的交互框架，扩展为能够：

1. 持续接收视频中的场景音频 `scene_audio`；
2. 将场景音频与视频帧放在统一时间轴上；
3. 根据声音事件强度调整帧选择和关注时间段；
4. 估计音频相对画面的时间偏移 `offset_sec`；
5. 根据音频质量 `q` 和问题相关性 `r` 计算门值 `g=q*r`；
6. 使用残差方式把可信音频融合进视频特征；
7. 使用 MUG 启发的伪标签训练事件、对齐和门控辅助头；
8. 在坏音频、缺失音频或错配音频下自动退化为原视觉能力；
9. 通过 E0-E7 反事实实验说明提升是否确实来自场景音频。

## 1.1.1 当前复现基线与功率前提

AS-M4 后续开发必须以当前本地复现事实为前提：

- 12k no-freeze full 已完成，输出目录为 `intersuit/checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze`，`MODEL_MAX_LENGTH=12288`，`M4_FREEZE_LM_HEAD=0`，`train_loss=0.06275078892921938`。
- 32k no-freeze full 已完成，输出目录为 `intersuit/checkpoints/M4-LongVA-7B-Qwen2-repro-32k-lowmem-nofreeze-v1`，`MODEL_MAX_LENGTH=32000`，`M4_FREEZE_LM_HEAD=0`，`train_loss=0.09159564487920577`。
- 12k 用于快速开发、模块 harness 和多数消融；32k 用于最终长上下文主实验或论文级对照。
- `300W` 是保守稳定默认功率策略；`450W` 已验证可完成 32k no-freeze full 训练，因此后续不再把非 300W 功率限制视为硬性 blocker。每次训练仍必须记录实际功率限制。
- 稳定四卡默认仍为 `CUDA_VISIBLE_DEVICES=0,2,3,4` 和 `NUM_GPUS=4`。GPU 1 在单独多卡验证前不进入生产多卡训练。

## 1.2 第一版非目标

第一版明确不做：

- 不重写 Qwen2、SigLIP、Whisper 等大骨干；
- 不把原 `speech` / `question_audio` 字段改名为场景音频；
- 不在第一版新增 `<scene_audio>` 特殊 token；
- 不修改 tokenizer 词表；
- 不把 MUG 的 AV-Mamba 网络原样移植；
- 不训练端到端原始波形大模型；
- 不把说话人分离、声源空间定位和多通道音频作为第一版硬依赖；
- 不使用测试集选择阈值、学习率或最佳 checkpoint；
- 不把 Hugging Face 个人镜像的 license 标签直接当作原视频授权证明。

## 1.3 三条不可破坏的兼容约束

1. `query_speech` 与 `scene_audio` 必须是两个独立输入职责。
2. 残差融合后的视频 token 数量必须与融合前完全一致。
3. 所有新增模块必须可以通过配置关闭；关闭后模型必须接近原 M4 行为。

## 1.4 回退与冻结策略

AS-M4 必须同时支持行为回退和权重回退，二者不能混淆。

行为回退用于验证新增路径没有破坏原 M4：

- `enable_scene_audio=false` 时，不读取或不使用 `scene_audio`；
- `use_audio_event_detector=false`、`use_temporal_aligner=false`、`use_audio_confidence_gate=false`、`use_gated_av_fusion=false` 时，对应模块必须旁路；
- 强制 `gate=0` 时，残差融合输出必须接近原视频特征；
- E1 归因实验必须覆盖 AS-M4 静音/关闭音频路径，并与 E0 原 M4 对齐比较。

权重回退用于恢复已经训练改变的模型参数：

- 冻结模块只表示后续训练不再更新这些参数，不能还原已经写入 checkpoint 的权重；
- 真正恢复原始能力必须加载保留的原始 M4、12k no-freeze 或 32k no-freeze checkpoint；
- AS-M4 checkpoint 必须写入新的 versioned run 目录，不得覆盖任何 M4 基线目录；
- 新增模块权重应尽量单独命名保存，方便只加载/卸载 AS-M4 增量；
- 每个集成阶段的 harness 必须验证“关闭 AS-M4 或 `gate=0` 后接近原 M4 行为”。

脚本层配置约定：

```bash
# 关闭 scene_audio，走原 M4 行为路径
AS_M4_ENABLE_SCENE_AUDIO=0

# 强制音频门控为 0，验证残差融合退化为原 video feature
AS_M4_FORCE_AUDIO_GATE=0

# 不训练 AS-M4 新增模块，只训练原 M4 可训练部分
MM_TUNABLE_PARTS=mm_vision_tower,mm_mlp_adapter,mm_language_model

# 训练 AS-M4 新增模块
MM_TUNABLE_PARTS=mm_vision_tower,mm_mlp_adapter,mm_language_model,streaming_av_module
```

验收要求：每次使用 `AS_M4_FORCE_AUDIO_GATE=0` 或 `AS_M4_ENABLE_SCENE_AUDIO=0` 时，dry-run 生成的 `.env` 必须记录这些值；E1/E7 归因实验必须保存对应 `.env` 和预测文件。

---

# 2. 已核验的原始代码锚点

以下路径和函数已针对固定 commit 核验。开始编码前仍需使用 `rg` 再确认目标工作树中的实际签名。

## 2.1 M4 仓库

固定仓库：

```text
https://github.com/OmniMMI/M4
commit: 574d2244576190cd150cd28aaaa8dbd2ea7502e6
```

关键位置：

| 文件 | 已核验对象 | 当前作用 / 缺口 |
|---|---|---|
| `intersuit/local_demo/proactive_cli.py` | `ffmpeg_once()` | 命令包含 `-an`，缓存视频时删除原音轨 |
| `intersuit/local_demo/proactive_cli.py` | `main()` | 按 `frame_fps` 调用 `LiveInfer.input_video_stream()` |
| `intersuit/local_demo/inference.py` | `LiveInfer` | 本地流式状态、视频帧和查询输入入口 |
| `intersuit/intersuit/model/llava_arch.py` | `LlavaMetaModel.__init__()` | 创建视觉塔和原问题语音编码器 |
| `intersuit/intersuit/model/llava_arch.py` | `encode_speech()` | 编码用户问题语音；不得复用为场景音频职责 |
| `intersuit/intersuit/model/llava_arch.py` | `prepare_inputs_labels_for_multimodal_av()` | 把视觉与 speech 特征插入语言模型输入；没有时间对齐和门控 |
| `intersuit/intersuit/model/language_model/llava_qwen.py` | `forward()` | 训练与推理总入口之一 |
| 同上 | `generate()` / `generate_parallel()` | 流式/并行生成相关入口；新增输入必须完整透传 |
| `intersuit/intersuit/train/train.py` | `DataArguments` | 训练数据参数定义 |
| 同上 | `LazySupervisedDataset._get_item()` | 单样本读取和预处理 |
| 同上 | `DataCollatorForSupervisedDataset.__call__()` | batch padding、mask 与张量拼装 |
| `intersuit/intersuit/model/builder.py` | `load_pretrained_model()` | checkpoint 和各模块权重加载 |

## 2.2 OmniMMI 评测仓库

固定仓库：

```text
https://github.com/OmniMMI/OmniMMI
commit: bd1f8460a39b821e1c5c830b710c049cd324ff2d
```

关键位置：

| 文件 | 对象 | 计划用途 |
|---|---|---|
| `evaluations/online_inference_utils.py` | `online_inference()` | OmniMMI 流式评测主循环 |
| 同上 | `ffmpeg_once()` | 必须保留画面缓存，同时增加独立场景音频提取 |
| `model_testing_zoo.py` | 模型适配入口 | 注册 AS-M4 评测适配器 |

## 2.3 论文依据

OmniMMI 论文中的 M4 主要解决：

- Streaming KV Cache；
- Highlight Spot；
- Proactive Interruption；
- Parallel Decoding；
- M4-IT 交互格式训练。

论文第 4-5 节和图 4-5 显示 Highlight Spot 主要使用文本 query 对视频 frame 计算注意力；论文同时指出音视觉模型未优于纯视觉模型，说明音视觉对齐仍然不足。本项目针对的是这一缺口，而不是替换 M4 的交互控制框架。

---

# 3. 目标目录结构

在 M4 的 `intersuit/intersuit/` 下新增：

```text
intersuit/intersuit/
├── streaming/
│   ├── __init__.py
│   ├── audio_stream.py
│   ├── av_buffer.py
│   └── frame_scheduler.py
├── model/
│   ├── scene_audio_encoder/
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   └── scene_audio_encoder.py
│   └── streaming_av/
│       ├── __init__.py
│       ├── builder.py
│       ├── event_detector.py
│       ├── temporal_aligner.py
│       ├── confidence_gate.py
│       ├── fusion.py
│       ├── losses.py
│       └── outputs.py
├── preprocess/
│   ├── datasets/
│   │   ├── common.py
│   │   ├── convert_avqa.py
│   │   ├── convert_music_avqa.py
│   │   ├── convert_ave.py
│   │   ├── convert_llp.py
│   │   ├── convert_longvale.py
│   │   └── convert_unav100.py
│   ├── extract_scene_audio.py
│   ├── generate_pseudo_labels.py
│   ├── build_shifted_pairs.py
│   ├── build_audio_corruptions.py
│   ├── build_counterfactual_eval.py
│   └── validate_streaming_av_json.py
├── harness/
│   ├── README.md
│   ├── run.py
│   ├── configs/
│   ├── fixtures/
│   ├── runners/
│   └── metrics/
└── tests/
    ├── test_audio_stream.py
    ├── test_av_buffer.py
    ├── test_scene_audio_encoder.py
    ├── test_event_detector.py
    ├── test_temporal_aligner.py
    ├── test_confidence_gate.py
    ├── test_fusion.py
    ├── test_dataset_adapters.py
    └── test_checkpoint_roundtrip.py
```

---

# 4. Harness 总契约

## 4.1 Harness 的含义

Harness 是可重复运行的实验支架。它不是单个单元测试，而是固定：

- 输入样本；
- 数据增强参数；
- 模型配置；
- 随机种子；
- 输出目录；
- 指标算法；
- 通过阈值。

## 4.2 统一运行接口

计划实现以下命令：

```bash
python -m intersuit.harness.run \
  --phase p5_alignment \
  --config intersuit/harness/configs/p5_alignment.yaml \
  --output artifacts/harness/p5_alignment/run_001
```

`run.py` 必须：

1. 解析阶段和配置；
2. 固定随机种子；
3. 记录 git commit 和工作树是否有未提交更改；
4. 记录 Python、PyTorch、CUDA、GPU 与关键依赖版本；
5. 调用对应 runner；
6. 保存指标和逐样本结果；
7. 根据阶段 Exit Gate 输出 `PASS` 或 `FAIL`；
8. 失败时返回非 0 退出码。

## 4.3 每次运行必须保存

```text
artifacts/harness/<phase>/<run_id>/
├── config.yaml
├── environment.json
├── git_state.json
├── metrics.json
├── predictions.jsonl
├── failures.jsonl
├── runtime.log
├── exit_gate.json
└── plots/
```

`exit_gate.json` 示例：

```json
{
  "phase": "p5_alignment",
  "status": "PASS",
  "checks": [
    {"name": "offset_mae_le_hop", "passed": true, "value": 0.18},
    {"name": "causal_mask_no_future", "passed": true},
    {"name": "no_nan", "passed": true}
  ]
}
```

## 4.4 测试数据分层

Harness 必须使用三层数据：

1. `synthetic`：程序生成，可精确知道真值；
2. `mini-real`：每个公开数据集固定 20-100 条，供开发快速运行；
3. `official-eval`：官方验证/测试集，仅在阶段完成后运行。

不得在 `official-eval` 上反复选择阈值。

---

# 5. P0：固定代码、环境与原始 M4 基线

## 5.1 操作目的

在任何功能修改前建立可复现基线。P0 的输出是后续所有性能差值的零点。

## 5.2 前置条件

- M4 和 OmniMMI 仓库已取得；
- 模型权重和官方依赖可用；
- GPU 能运行原 M4；
- 不改动任何核心模型逻辑。

## 5.3 实践步骤

1. 检查实际仓库：

   ```bash
   git status --short
   git rev-parse HEAD
   rg -n "def ffmpeg_once|class LiveInfer|prepare_inputs_labels_for_multimodal_av|def generate_parallel" .
   ```

2. 固定 M4 commit 和 OmniMMI commit。
3. 保存环境：

   ```bash
   python --version
   python -c "import torch; print(torch.__version__, torch.version.cuda)"
   nvidia-smi
   pip freeze > artifacts/baseline/pip-freeze.txt
   ```

4. 运行官方本地 proactive demo。
5. 从 OmniMMI 固定抽取 20-50 条样本，至少覆盖 PA、PT、SG、SI。
6. 使用相同 seed 连续运行三次。
7. 保存原始预测、触发时间、处理 FPS、显存峰值。

## 5.4 新建 Harness

```text
harness/runners/reproduce_baseline.py
harness/configs/p0_baseline.yaml
```

建议函数：

```python
def load_fixed_manifest(path): ...
def run_fixed_subset(model, manifest, seed): ...
def compare_repeated_runs(run_paths): ...
def measure_trigger_time(history): ...
```

## 5.5 Exit Gate

- [ ] 官方 demo 完整运行；
- [ ] 固定样本无崩溃；
- [ ] 三次准确率差异不超过 1 个百分点；
- [ ] 没有无法解释的随机触发漂移；
- [ ] baseline 的配置、预测、日志和资源指标齐全；
- [ ] 后续 E0 使用的就是本阶段保存的基线定义。

## 5.6 失败处理

- 环境失败：修环境，不改模型；
- 权重不兼容：记录权重来源和 commit，不用 `strict=False` 静默吞掉全部错误；
- 官方代码本身崩溃：建立最小复现并记录为 baseline blocker；
- P0 未通过：禁止进入 P1。

---

# 6. P1：独立场景音频 I/O 与时间戳

## 6.1 操作目的

保留原问题语音支路，新增视频场景音频支路，并为所有波形块提供准确时间戳。

## 6.2 关键设计

```text
query_speech -> 原 Whisper / speech_projector -> 用户问题语义
scene_audio -> 新场景音频编码器 -> 环境事件语义
```

不得把整个视频音轨写入 `speech` 字段。

## 6.3 修改文件

### 修改

- `intersuit/local_demo/proactive_cli.py::ffmpeg_once()`
- `intersuit/local_demo/proactive_cli.py::main()`
- `evaluations/online_inference_utils.py::ffmpeg_once()`
- `evaluations/online_inference_utils.py::online_inference()`

### 新建

- `streaming/audio_stream.py`
- `preprocess/extract_scene_audio.py`
- `tests/test_audio_stream.py`

## 6.4 实践方法

1. 保留原只含画面的缓存视频，以避免改变视觉解码行为。
2. 额外使用 ffmpeg 提取单声道 16 kHz WAV：

   ```bash
   ffmpeg -y -i input.mp4 -vn -ac 1 -ar 16000 scene_audio.wav
   ```

3. 新增 `extract_scene_audio()`；不得把命令散落复制到多个文件。
4. 为每个音频块记录：

   ```python
   @dataclass
   class AudioChunk:
       samples: torch.Tensor
       start_sec: float
       end_sec: float
       sequence_id: int
       sample_rate: int
   ```

5. 支持音频缺失、损坏和全静音；这些情况返回诊断状态，不直接抛出难以定位的深层模型错误。
6. 所有时间使用视频起点为 0 的秒数，禁止混用毫秒整数、帧编号和 wall-clock 时间而不转换。

## 6.5 Harness

新建：

```text
harness/fixtures/build_beep_flash_fixture.py
harness/runners/test_audio_io.py
```

生成 60 秒视频：

- 10、20、40 秒画面闪烁；
- 同时播放短蜂鸣；
- 另生成音频提前/延迟 0.5 秒版本；
- 另生成无音轨版本。

检查：

- WAV 是否可解码；
- 时长误差；
- chunk 是否覆盖完整时间线；
- 时间戳是否单调；
- 缺失音频是否安全降级；
- 10 分钟回放是否产生累计漂移。

## 6.6 Exit Gate

- [ ] 音频与视频分别缓存；
- [ ] `query_speech` 与 `scene_audio` 字段独立；
- [ ] 时间戳单调且窗口无重叠错误；
- [ ] 无音轨视频不崩溃；
- [ ] 60 秒合成视频中蜂鸣位置误差不超过一个音频 hop；
- [ ] 10 分钟回放无明显累计漂移。

---

# 7. P2：冻结的场景音频编码器

## 7.1 操作目的

把波形转换为窗口级音频特征，同时避免第一版把问题语音和场景声音混在同一语义职责中。

## 7.2 第一版选择原则

优先级：

1. 冻结的 BEATs 或等价事件编码器；
2. PANNs 作为更轻量替代；
3. CLAP 用于文本-音频相关性教师或第二编码支路，不要求第一版主编码器必须用 CLAP。

最终选择前记录：

- 权重许可；
- 商用/非商用限制；
- 单窗口显存和延迟；
- 在目标数据域上的事件表现；
- 是否能离线缓存特征。

## 7.3 新建文件

```text
model/scene_audio_encoder/builder.py
model/scene_audio_encoder/scene_audio_encoder.py
tests/test_scene_audio_encoder.py
```

接口：

```python
class SceneAudioEncoder(nn.Module):
    def forward(
        self,
        waveform: Tensor,
        sample_mask: Tensor,
        window_timestamps: Tensor,
    ) -> SceneAudioEncoderOutput:
        ...
```

输出至少包含：

```python
audio_features       # [B, Ta, Da]
audio_valid_mask     # [B, Ta]
window_timestamps    # [B, Ta, 2]
```

## 7.4 实践方法

1. 只在 `builder.py` 解析编码器类型；
2. 统一采样率和归一化规则；
3. 编码器参数 `requires_grad=False`；
4. 只训练新增 projector；
5. 支持预计算 `.npy` / `.pt` 特征，训练时避免重复跑冻结编码器；
6. 任何 padding 位置必须由 mask 排除；
7. 静音窗口必须产生有限数值，禁止 NaN/Inf。

## 7.5 Harness

```text
harness/runners/test_audio_encoder.py
```

测试：静音、正常讲话、蜂鸣、音乐、噪声、不同长度、混合 batch。

记录：

- 输出 shape；
- 相同输入重复误差；
- 单窗口延迟；
- RTF；
- 显存峰值；
- 冻结参数是否出现梯度。

## 7.6 Exit Gate

- [ ] 输出 shape 和 mask 正确；
- [ ] 静音不产生 NaN/Inf；
- [ ] 编码器确实冻结；
- [ ] 相同输入输出稳定；
- [ ] 训练可直接读取缓存特征；
- [ ] 资源指标已记录并满足当前硬件预算。

## 7.7 参考仓库

- BEATs: https://github.com/microsoft/unilm/tree/master/beats
- PANNs: https://github.com/qiuqiangkong/audioset_tagging_cnn
- CLAP: https://github.com/LAION-AI/CLAP

---

# 8. P3：音视频双流缓冲与统一时间轴

## 8.1 操作目的

处理音频和视频异步到达、乱序、丢包、不同采样频率和可选 lookahead。

## 8.2 新建文件

```text
streaming/av_buffer.py
tests/test_av_buffer.py
harness/runners/replay_stream.py
```

接口：

```python
class AVBuffer:
    def push_audio(self, chunk: AudioChunk) -> None: ...
    def push_frame(self, frame: Tensor, timestamp: float) -> None: ...
    def pop_window(
        self,
        current_time: float,
        window_sec: float,
        hop_sec: float,
        lookahead_sec: float,
    ) -> AVWindow | None: ...
    def prune(self, current_time: float) -> None: ...
```

## 8.3 配置

```yaml
sample_rate: 16000
window_sec: 2.0
hop_sec: 0.5
history_sec: 10.0
max_offset_sec: 1.5
lookahead_sec: 0.0
```

必须支持：

- `lookahead_sec=0.0`：严格因果；
- `lookahead_sec=0.5`：低延迟高稳定模式。

## 8.4 实践方法

1. AudioChunk 和 VideoFrame 都使用秒时间戳；
2. buffer 只保留有限历史；
3. 乱序数据在容忍范围内排序，超过范围记录并丢弃；
4. 不完整窗口使用 padding + valid mask；
5. 严格因果模式断言任何读取时间不超过 `current_time`；
6. 0.5 秒模式断言读取时间不超过 `current_time+0.5`；
7. reset 必须清空音频、视频、窗口序号和统计量。

## 8.5 Harness

模拟：

- 音频晚到 100/300/500 ms；
- 视频帧乱序；
- 每 20 个窗口丢 1 个音频块；
- 每 30 帧丢 1 帧；
- 输入速度抖动；
- 30 分钟长流；
- reset 后重新开始。

## 8.6 Exit Gate

- [ ] 输出时间戳单调；
- [ ] 0 lookahead 无未来泄漏；
- [ ] 0.5 秒模式不超过允许前瞻；
- [ ] 内存随时间有界；
- [ ] reset 无残留状态；
- [ ] 在线回放与等价离线切窗输出一致到允许容差。

---

# 9. P4：音频事件检测与事件感知帧调度

## 9.1 操作目的

让明显声音事件提高相关时间段的帧预算，补足原 M4 只用文本 query 对 frame 计算 Highlight Spot 的不足。

## 9.2 新建文件

```text
model/streaming_av/event_detector.py
streaming/frame_scheduler.py
tests/test_event_detector.py
harness/runners/test_event_scheduler.py
```

接口：

```python
class AudioEventDetector(nn.Module):
    def forward(self, audio_features, valid_mask) -> AudioEventOutput:
        ...

class EventAwareFrameScheduler:
    def update(self, event_intensity, timestamp): ...
    def select(self, candidate_frames, frame_budget): ...
```

输出：

```python
event_logits
event_probability
event_intensity
event_embedding
```

## 9.3 初版强度定义

```text
intensity =
    lambda_prob  * event_probability
  + lambda_delta * positive_probability_change
  + lambda_energy * normalized_audio_energy
```

所有 lambda 只能在训练/验证集选择。

## 9.4 实践方法

1. 使用 AVE/LLP 已有标签训练事件头；
2. 安静段使用基础帧率；
3. 高事件强度段临时提高采样；
4. 设置最小持续时间，防止单个异常尖峰反复触发；
5. 设置冷却时间，防止相同事件无限抢占帧预算；
6. 固定总帧预算，公平比较固定 1 FPS 与动态调度。

## 9.5 Harness

比较：

```text
A: fixed_1fps
B: event_aware_same_budget
```

指标：

- event-frame recall；
- segment F1；
- frame budget；
- trigger delay；
- 静音误触发率；
- QA 指标（仅作辅助）。

## 9.6 Exit Gate

- [ ] 相同帧预算下事件相关帧召回提高；
- [ ] 安静段不会持续高采样；
- [ ] 纯音量变化不会无条件等同语义事件；
- [ ] 调度器可关闭并恢复固定帧率；
- [ ] 没有因为调度器导致缓存无限增长。

## 9.7 参考仓库

- AVE: https://github.com/YapengTian/AVE-ECCV18
- LLP/AVVP: https://github.com/YapengTian/AVVP-ECCV20

---

# 10. P5：因果音画时间对齐

## 10.1 操作目的

预测场景音频相对视频画面的时间偏移，并把音频特征重新对齐到相应视频窗口。

## 10.2 新建文件

```text
model/streaming_av/temporal_aligner.py
preprocess/build_shifted_pairs.py
tests/test_temporal_aligner.py
harness/runners/test_alignment.py
```

## 10.3 接口

```python
class CausalTemporalAligner(nn.Module):
    def forward(
        self,
        audio_features,
        video_features,
        audio_timestamps,
        video_timestamps,
        audio_mask,
        video_mask,
        lookahead_sec,
    ) -> AlignmentOutput:
        ...
```

`AlignmentOutput`：

```python
alignment_matrix
offset_sec
offset_confidence
aligned_audio_features
```

## 10.4 训练数据

自动平移同步音频：

```text
-1.50, -1.00, -0.50, -0.25,
 0.00,
+0.25, +0.50, +1.00, +1.50 seconds
```

必须明确偏移符号定义。例如：

```text
offset_sec = audio_time - video_time
```

所有代码、配置、图表使用同一符号约定。

## 10.5 实践方法

1. 将音频和视频特征投影到同一对齐维度；
2. 计算候选偏移内的跨模态相似度；
3. 使用 causal mask 屏蔽不允许的未来位置；
4. 从相似度分布预测 offset 和 confidence；
5. confidence 低于阈值时保持原窗口，不做激进修正；
6. 将对齐后的音频按视频时间位置聚合。

## 10.6 Harness

```bash
python -m intersuit.harness.run \
  --phase p5_alignment \
  --config intersuit/harness/configs/p5_alignment_0lookahead.yaml

python -m intersuit.harness.run \
  --phase p5_alignment \
  --config intersuit/harness/configs/p5_alignment_05lookahead.yaml
```

指标：

- offset MAE；
- within-one-hop accuracy；
- 0-offset false correction；
- alignment confidence calibration；
- E5/E6 问答恢复量。

## 10.7 初始 Exit Gate

阈值是第一版工程门槛，不是最终论文结论；只能在 validation 调整。

- [ ] 合成偏移 MAE 不超过一个 hop；
- [ ] 0 偏移样本不会频繁预测为大偏移；
- [ ] causal mask 单元测试证明无未来泄漏；
- [ ] 低 confidence 时执行安全回退；
- [ ] 0.5 秒 lookahead 相对 0 秒模式的精度和延迟均有记录。

---

# 11. P6：音频质量、相关性与置信度门控

## 11.1 操作目的

区分“音频能不能信”和“音频该不该用于当前问题”。

```text
q = quality score
r = relevance score
g = q * r
```

## 11.2 新建文件

```text
model/streaming_av/confidence_gate.py
preprocess/build_audio_corruptions.py
tests/test_confidence_gate.py
harness/runners/test_gate.py
```

## 11.3 接口

```python
class AudioQualityEstimator(nn.Module):
    def forward(self, waveform_stats, audio_features, valid_mask): ...

class AudioRelevanceEstimator(nn.Module):
    def forward(
        self,
        audio_features,
        video_features,
        question_features,
        offset_confidence,
        valid_mask,
    ): ...

class AudioConfidenceGate(nn.Module):
    def forward(self, quality, relevance):
        return quality * relevance
```

## 11.4 q 输入信号

- SNR 或可学习噪声表示；
- 音频能量；
- 静音比例；
- clipping 比例；
- 缺失比例；
- VAD；
- 多说话人重叠提示；
- 编码器异常和 valid mask。

## 11.5 r 输入信号

- audio-video similarity；
- audio-question similarity；
- offset confidence；
- scene event 与问题是否相关；
- 历史一致性。

## 11.6 自动标签构造

| 样本 | q target | r target | g target |
|---|---:|---:|---:|
| 清晰同步同源音频 | 高 | 高 | 高 |
| 清晰但跨视频换音轨 | 高 | 低 | 低 |
| 正确但强噪声 | 低 | 中/高 | 低 |
| 静音/缺失 | 0 | masked | 0 |
| 明显错位 | 中/高 | 低 | 低 |
| 与画面相关但与问题无关 | 高 | 低 | 低 |

## 11.7 损失

```text
L_gate = BCE(q, q_target)
       + BCE(r, r_target)
       + SmoothL1(g, g_target)
```

伪标签必须乘以 confidence；缺失标签使用 mask，不得当作负类。

## 11.8 Harness

同一视频自动生成：

```text
clean / mute / noise / swap / shift / clipping / partial_missing
```

记录：

- q/r/g 分布；
- gate AUROC；
- ECE/calibration；
- robustness drop；
- 门控饱和比例；
- 按增强类型的逐组表现。

## 11.9 Exit Gate

- [ ] clean 的平均 g 高于 swap/noise/mute；
- [ ] mute 的 g 接近 0；
- [ ] SNR 下降时 q 总体下降；
- [ ] 坏音频下门控融合优于直接融合；
- [ ] g 不全部接近 0 或 1；
- [ ] 所有硬回退规则有测试。

## 11.10 辅助数据

- MUSAN: https://www.openslr.org/17/
- DNS Challenge: https://github.com/microsoft/DNS-Challenge
- LibriMix: https://github.com/JorisCos/LibriMix

---

# 12. P7：残差融合并接入 M4

## 12.1 操作目的

将已对齐、带门控的音频作为视频特征增量，同时维持原视频 token 形状和 checkpoint 兼容性。

## 12.2 新建文件

```text
model/streaming_av/fusion.py
model/streaming_av/outputs.py
model/streaming_av/builder.py
model/streaming_av/losses.py
tests/test_fusion.py
tests/test_checkpoint_roundtrip.py
```

## 12.3 融合公式

```text
delta_audio = audio_projector(aligned_audio_features)
fused_video = video_features + gate * delta_audio
```

必须断言：

```python
assert fused_video.shape == video_features.shape
```

## 12.4 修改 `llava_arch.py`

修改/新增：

```text
LlavaMetaModel.__init__()
get_scene_audio_encoder()
get_streaming_av_modules()
initialize_scene_audio_modules()
encode_scene_audio()
prepare_inputs_labels_for_streaming_av()
```

要求：

- 保留 `encode_speech()`；
- 保留旧 multimodal 路径；
- 新路径用配置开关选择；
- FSDP 包装方式与现有 vision/speech 模块一致；
- 所有辅助输出集中放入 dataclass，不用无说明 tuple。

## 12.5 修改 `llava_qwen.py`

在实际签名核验后修改：

```text
forward()
generate()
generate_parallel()
generate_streaming()（若目标 commit 存在有效实现）
```

新增参数：

```python
scene_audio
scene_audio_timestamps
frame_timestamps
audio_valid_mask
video_valid_mask
pseudo_labels
aux_loss_mask
```

禁止：

- 只改 `forward()` 而忘记 `generate()`；
- 用固定 token 公式计算 prefix length；
- 在生成路径静默丢弃 scene_audio；
- 在旧 checkpoint 加载时隐藏所有 missing key。

## 12.6 修改 `model/builder.py`

`load_pretrained_model()`：

- 创建新模块；
- 加载 scene audio/projector/event/align/gate/fusion 权重；
- 输出 missing/unexpected keys 摘要；
- 只允许明确白名单中的新增 key 缺失；
- 旧 checkpoint 加载后新模块使用安全初始化，默认 gate 上限较低。

## 12.7 Harness

```text
harness/runners/test_integration.py
```

矩阵：

1. 无 scene_audio；
2. 静音；
3. 正确音频；
4. 错配音频；
5. 旧 checkpoint；
6. 新 checkpoint；
7. batch size 1；
8. batch size >1；
9. 所有开关关闭；
10. 仅对齐/仅门控/完整模块。

## 12.8 Exit Gate

- [ ] 无音频可运行；
- [ ] 开关关闭时接近原 M4；
- [ ] token 数量不变；
- [ ] 旧 checkpoint 可加载；
- [ ] 新 checkpoint 保存/恢复一致；
- [ ] 所有输入路径无 NaN、shape 或 device 错误；
- [ ] 辅助日志能看到 q/r/g/offset，而不泄露大张量导致日志爆炸。

---

# 13. P8：数据适配、MUG 式伪标签和人工抽检

## 13.1 操作目的

复用公开标注，自动生成偏移、噪声、错配和门控监督，只把人工工作用于验证“音频必要性”和“伪标签质量”。

## 13.2 统一样本结构

`preprocess/datasets/common.py`：

```python
@dataclass
class AVSample:
    id: str
    dataset: str
    original_split: str
    video: str
    scene_audio: str | None
    query_speech: list[str]
    conversations: list[dict]
    frame_timestamps: list[float]
    audio_window_timestamps: list[list[float]]
    pseudo_labels: dict | None
```

必须保留 `dataset` 和 `original_split`，训练入口断言不存在 test 样本。

## 13.3 数据适配文件

| 文件 | 函数 | 作用 |
|---|---|---|
| `convert_avqa.py` | `load_official_qa()` / `resolve_video()` / `convert_split()` | AVQA 问答与原视频适配 |
| `convert_music_avqa.py` | `map_question_type()` / `convert_split()` | 保留 audio/visual/audio-visual 题型 |
| `convert_ave.py` | `load_boundaries()` / `build_event_windows()` | 生成事件和时间边界监督 |
| `convert_llp.py` | `load_weak_labels()` / `load_dense_eval()` | 弱标签训练与秒级评测 |
| `convert_longvale.py` | `load_event_captions()` / `build_stream_windows()` | 长视频窗口和关系描述 |
| `convert_unav100.py` | `load_events()` / `build_untrimmed_windows()` | 多事件和未剪辑压力数据 |

## 13.4 伪标签文件

```text
generate_pseudo_labels.py
build_shifted_pairs.py
build_audio_corruptions.py
build_counterfactual_eval.py
validate_streaming_av_json.py
```

四类目标：

```json
{
  "audio_events": [],
  "offset_sec": 0.5,
  "offset_confidence": 0.91,
  "av_match": 1,
  "gate_target": 0.82,
  "teacher_name": "teacher_x",
  "teacher_version": "revision_y",
  "confidence": 0.89
}
```

## 13.5 MUG 启发的实践规则

1. 使用音频教师生成分段声音事件；
2. 使用视觉教师生成视觉事件；
3. 使用文本语义或跨模态教师判断一致性；
4. 低置信度标签不进入对应辅助损失；
5. 跨视频换音轨用于训练 `r`；
6. 如果换音轨后问题答案不再成立，该样本只训练辅助头，不训练 QA 主损失；
7. 保存教师版本和生成参数，确保可追溯。

## 13.6 最低人工复核

不需要从零标注大数据集。建议：

| 人工任务 | 数量 | 用途 |
|---|---:|---|
| MUSIC-AVQA 音频必要性 | 300 题 | 验证题目是否真的必须听声音 |
| AVQA 音频必要性 | 200 题 | 补充真实生活场景证据 |
| 伪标签抽检 | 300 条窗口 | 估计教师标签精度 |
| 加噪/错配自动样本抽检 | 100 条 | 检查自动管线没有生成错误 |
| 真实设备延迟 | 0-100 段 | 仅在声称真实在线对齐时需要 |

人工评测集默认只用于评测，不用于训练。

## 13.7 Harness

```text
harness/runners/test_dataset_adapters.py
harness/runners/audit_pseudo_labels.py
tests/test_dataset_adapters.py
```

检查：

- 文件存在且可解码；
- 音轨未意外丢失；
- split 隔离；
- 时间戳单调；
- 题型映射完整；
- 伪标签 coverage；
- 类别分布；
- 教师置信度分布；
- 人工抽检准确率。

## 13.8 Exit Gate

- [ ] 训练 manifest 中 test 数量为 0；
- [ ] 所有音频路径可读或有明确 missing 标记；
- [ ] 高置信度伪标签人工正确率达到预设门槛（初始建议 80%）；
- [ ] 低置信度标签已 mask；
- [ ] 跨视频样本未盲目沿用音频依赖 QA 答案；
- [ ] 数据许可证和来源记录齐全。

---

# 14. P9：分阶段训练

## 14.1 操作目的

先让新增小模块学会事件、对齐和门控，再逐步允许语言模型适应音频增量，避免一次性破坏原 M4。

AS-M4 训练采用双基线策略：

- 12k 路线：默认开发路线，用于新增模块、单元测试、harness、E0-E7 小集归因和快速训练。
- 32k 路线：最终长上下文路线，只有 12k smoke/canary/full 和小集归因通过后才能启动。32k 必须重新执行 `20-step -> 150-step -> full` 阶梯，不能因为原 M4 32k 已完成就直接启动 AS-M4 full。

默认训练环境继承 M4 成功路线：

```text
CUDA_VISIBLE_DEVICES=0,2,3,4
NUM_GPUS=4
DEEPSPEED_CONFIG=scripts/zero3_lowmem.json
M4_CHUNKED_LM_LOSS=1
M4_CHUNKED_LM_LOSS_TOKENS=512
M4_CHUNKED_LM_LOSS_CHECKPOINT=0
M4_FREEZE_LM_HEAD=0
```

AS-M4 新增模块不得通过增加大量 LLM token 来接入场景音频。首版必须优先使用冻结或预计算的 scene audio features，并以残差方式融合进视频特征，保持视频 token 数量不变。

## 14.2 Stage A：只训练新增辅助头

冻结：

- Qwen2；
- vision tower；
- scene audio encoder；
- 原 speech encoder。

训练：

- scene audio projector；
- event detector；
- temporal aligner；
- quality/relevance gate；
- residual fusion。

## 14.3 Stage B：加入 Qwen2 LoRA

辅助指标稳定后再开启 LoRA。不同参数组使用独立学习率。

## 14.4 Stage C：可选小学习率联合微调

只有 Stage B 仍存在明确瓶颈时才考虑；默认仍冻结大型编码器。

## 14.5 总损失

```text
L_total = L_lm
        + lambda_event       * L_event
        + lambda_align       * L_align
        + lambda_gate        * L_gate
        + lambda_consistency * L_consistency
```

所有 lambda 在 validation 选择，并写入 config。

## 14.6 必做训练 Harness

### 32 样本过拟合

目的：证明数据、loss 和梯度链路真的工作。

通过条件：训练损失明显下降，关键辅助目标能被学会。

### 100 step smoke test

检查：

- 无 NaN/Inf；
- 显存无持续增长；
- checkpoint 保存/恢复；
- 所有 loss 存在；
- q/r/g/offset 有合理分布；
- dataloader 可跨 epoch。

### 正式训练

记录：

- 各项 loss；
- QA validation；
- event F1；
- offset MAE；
- gate AUROC/ECE；
- q/r/g 直方图；
- 显存、吞吐、checkpoint 大小。

### 32k OOM 保精度降级顺序

如果 AS-M4 新增模块导致 32k OOM，必须优先牺牲时间来保精度。每次只改一个降级项，并重新跑 `2-step -> 20-step -> 150-step`。

允许的降级顺序：

1. 重新运行 preflight，确认不是残留进程、坏样本、缓存、磁盘或功率记录问题。
2. 离线预计算冻结场景音频特征，训练时不再跑 scene audio encoder forward。
3. 对 event detector、temporal aligner、confidence gate、fusion 和 projector 启用 activation checkpointing。
4. 将 `M4_CHUNKED_LM_LOSS_TOKENS` 从 `512` 降到 `256`，仍 OOM 再降到 `128`；这不改变 loss 目标，只增加重算/循环时间。
5. 将 micro batch 降到 1，并通过 `gradient_accumulation_steps` 保持有效 batch。
6. 对音画 similarity matrix 使用分块计算，不缩短原定 offset 搜索范围。
7. 减少训练中保留的诊断张量，只记录抽样统计，不保存全量 alignment matrix。
8. 受控尝试 ZeRO-3 optimizer CPU offload；必须先确认系统内存并做 canary，避免再次触发系统 OOM killer。
9. 以上仍失败时，才小幅缩短 `audio_history_sec` 或 `max_offset_sec`。
10. 最后才考虑降低 LoRA rank 或冻结更多新增模块。

不优先采用：

- 不优先冻结 `lm_head`；
- 不优先降低 `MODEL_MAX_LENGTH`；
- 不优先减少视频 token、帧预算或有效场景音频信息；
- 不优先关闭音频模块来让训练“通过”。

## 14.7 Exit Gate

- [ ] 32 样本可过拟合；
- [ ] 100 step smoke test 全通过；
- [ ] checkpoint roundtrip 一致；
- [ ] 没有数据泄漏；
- [ ] validation 指标支持进入完整评测；
- [ ] 原视觉能力没有不可接受的坍塌。

---

# 15. P10：E0-E7 场景音频归因评测

## 15.1 操作目的

区分音频真实收益、参数增加、训练数据增加和坏音频干扰。

## 15.2 实验矩阵

| ID | 模型与输入 | 研究问题 |
|---|---|---|
| E0 | 原 M4 | 原始基线 |
| E1 | AS-M4 + 静音 scene_audio | 参数/训练结构本身是否带来收益 |
| E2 | AS-M4 + 正确同步音频 | 正确场景音频净增益 |
| E3 | AS-M4 + 跨视频错配音频 | 是否只利用任意声音或数据先验 |
| E4 | AS-M4 + 多级 SNR 噪声 | q 是否有效 |
| E5 | AS-M4 + 已知错位，关闭对齐 | 错位伤害 |
| E6 | 同 E5，开启对齐 | 对齐恢复量 |
| E7 | 分别关闭 q、r、全部 gate | 门控消融 |

## 15.3 主数据集

- AVQA：真实生活问答；
- MUSIC-AVQA：强音频依赖和题型分组；
- MUSIC-AVQA v2.0：可选偏差稳健性。

## 15.4 辅助数据集

- AVE：事件和对齐；
- LLP：事件解析与弱监督；
- LongVALE：长视频全模态事件；
- UnAV-100：未剪辑多事件压力。

## 15.5 Harness 与新建文件

```text
evaluations/evaluate_audio_attribution.py
harness/runners/run_ablation_matrix.py
harness/metrics/qa_metrics.py
harness/metrics/alignment_metrics.py
harness/metrics/gate_metrics.py
```

建议函数：

```python
def run_matrix(config): ...
def report_by_question_type(predictions): ...
def paired_bootstrap(baseline, treatment, n_resamples=10000): ...
def summarize_gate_behavior(records): ...
def summarize_alignment_recovery(e5, e6): ...
```

## 15.6 必报指标

- QA Accuracy / Exact Match；
- audio-necessary accuracy；
- visual-only accuracy；
- audio-visual question accuracy；
- offset MAE；
- within-one-hop accuracy；
- gate AUROC；
- ECE/calibration；
- robustness drop；
- latency / RTF；
- 95% paired confidence interval。

## 15.7 允许的归因结论

只有同时满足以下证据，才可写“提升主要来自场景音频”：

1. E2 显著高于 E1；
2. E3/E4/E5 出现可解释下降；
3. E6 相对 E5 恢复；
4. 完整门控在坏音频下优于 E7 无门控；
5. 音频必要题增益大于纯视觉题；
6. 95% 置信区间支持差值；
7. 结果不是只来自某一个随机种子。

若只有 E2>E0，只能描述相关性，不得写成因果证明。

## 15.8 Exit Gate

- [ ] E0-E7 同一题目、同一评分器、同一 split；
- [ ] 只改变实验定义中的音频或模块开关；
- [ ] 所有组保存逐样本预测；
- [ ] 按题型和音频必要性分组；
- [ ] 报告置信区间和失败案例；
- [ ] 没有只挑最佳 seed。

---

# 16. P11：实时部署、lookahead 和压力测试

## 16.1 操作目的

确认模型是真流式系统，而不是把完整视频离线处理后伪装成实时。

## 16.2 Harness 与新建文件

```text
harness/runners/profile_streaming.py
harness/metrics/latency_metrics.py
```

## 16.3 两套必须报告的模式

```yaml
# strict causal
lookahead_sec: 0.0
```

```yaml
# low-latency stabilized
lookahead_sec: 0.5
```

## 16.4 压力场景

- 视频丢帧；
- 音频丢块；
- 音频延迟 0.1-1.5 秒；
- 延迟随时间变化；
- 输入乱序；
- 新问题打断旧回答；
- 连续噪声；
- 音频突然缺失；
- 30 分钟与 1 小时长流；
- reset 和重新连接。

## 16.5 指标

- 平均延迟；
- P50/P95/P99 延迟；
- RTF；
- 平均/峰值显存；
- CPU 内存增长；
- audio/video buffer 长度；
- 触发延迟；
- 丢包恢复；
- 长流准确率漂移。

## 16.6 安全回退

以下情况硬性令 `g=0`：

- scene_audio 缺失；
- 全静音；
- 时间戳无效；
- 编码器输出 NaN/Inf；
- offset confidence 低于硬阈值；
- audio mask 全 0。

## 16.7 Exit Gate

- [ ] 目标硬件 RTF<1；
- [ ] 30 分钟以上无崩溃和无界内存增长；
- [ ] 缺失音频继续走视觉路径；
- [ ] 0/0.5 秒模式均有精度-延迟报告；
- [ ] 所有流式状态可 reset；
- [ ] 失败输入有诊断日志而不是静默错误。

---

# 17. 数据来源、作用和使用边界

| 数据集 | 主要角色 | 是否需要新增人工大标注 | 官方来源 / 优先来源 |
|---|---|---:|---|
| AVQA | 真实生活音视频问答 | 否；建议小规模音频必要性复核 | https://github.com/AlyssaYoung/AVQA |
| MUSIC-AVQA | 强音频依赖 QA 与题型分组 | 否；建议 300 题复核 | https://github.com/GeWu-Lab/MUSIC-AVQA |
| MUSIC-AVQA v2.0 | 数据偏差稳健性 | 否 | https://huggingface.co/datasets/DraculaDragon/MUSIC-AVQA-v2.0 |
| AVE | 事件边界、对齐 | 否 | https://github.com/YapengTian/AVE-ECCV18 |
| LLP/AVVP | 弱监督、伪标签、分段解析 | 否；只做伪标签抽检 | https://github.com/YapengTian/AVVP-ECCV20 |
| LongVALE | 长视频事件、时间戳、关系描述 | 否 | https://github.com/ttgeng233/LongVALE |
| UnAV-100 | 未剪辑多事件和对齐压力 | 否 | https://github.com/ttgeng233/UnAV |
| MUSAN | 噪声/音乐/语音混合 | 否 | https://www.openslr.org/17/ |
| DNS Challenge | 真实噪声与混响 | 否 | https://github.com/microsoft/DNS-Challenge |
| LibriMix | 多人重叠语音 | 否 | https://github.com/JorisCos/LibriMix |
| AVA Active Speaker | 活跃说话人，第二阶段可选 | 否 | https://research.google.com/ava/ |

许可证要求：

1. 代码许可证和原视频版权分开记录；
2. LongVALE/UnAV-100 等非商业许可不得忽略；
3. 社区镜像只用于字段检查时，也要记录来源；
4. 原作者仓库未明确数据许可时，在正式发布权重或数据前必须再次确认；
5. 不重新分发无权分发的原视频。

---

# 18. Codex 每阶段工作模板

Codex 接到任一阶段任务时，按以下模板执行并汇报：

```markdown
## Phase
P5: Causal Temporal Alignment

## Grounded repository state
- repository:
- commit:
- dirty files before work:
- files/functions verified with rg:

## Intended changes
- existing files modified:
- new files:
- interfaces added:

## Implementation
- design decision:
- tensor shapes:
- masks/timestamps:
- fallback behavior:

## Harness
- fixture:
- command:
- metrics:
- expected gate:

## Verification results
- unit tests:
- harness status:
- metrics:
- failures:

## Exit Gate
- [ ] ...

## Decision
- PASS: proceed to next phase
- FAIL: remain in current phase
```

Codex 不得用“代码已经写完”代替 Harness 结果。

---

# 19. 需要人工决策的暂停点

以下情况 Codex 必须暂停并请求项目负责人决定：

1. 原 M4 基线无法复现，且差异超过 P0 容差；
2. 场景音频编码器的许可证、显存或数据域选择会改变项目范围；
3. 是否允许线上 0.5 秒 lookahead 尚未确定；
4. 官方数据许可不清楚且计划发布衍生数据/权重；
5. 高置信度伪标签人工准确率低于预设门槛；
6. E2 不高于 E1，需要决定继续调模型还是终止该路线；
7. 完整门控在好音频上长期把 g 压到接近 0；
8. 新增模块导致原视觉任务不可接受地退化；
9. 完整训练需要超出已授权的算力或外部服务费用。

---

# 20. 最小可行版本与完整版

## 20.1 最小可行版本（MVP）

```text
P0 -> P1 -> P2 -> P3 -> P5 -> P6 -> P7 -> P10
```

暂缓 P4 大规模事件调度和 P8 大规模伪标签。目标是先证明：

1. 正确同步音频优于静音；
2. 错配/噪声音频会被门控降权；
3. 对齐能恢复人为错位造成的损失；
4. 关闭新增模块可回到原 M4。

## 20.2 完整研究版本

完成 P0-P11，并加入：

- AVE/LLP 事件监督；
- LongVALE 长视频；
- MUG 式伪标签；
- 音频必要性人工子集；
- 真实设备延迟小数据；
- E0-E7 全矩阵与统计置信区间。

---

# 21. Definition of Done

只有全部满足，项目才算完成：

- [ ] 原 M4 基线可复现；
- [ ] scene_audio 与 query_speech 完全分离；
- [ ] 0/0.5 秒模式均无未来信息违规；
- [ ] offset、q、r、g 可记录、可解释；
- [ ] 音频缺失或损坏时自动回退视觉；
- [ ] 残差融合不改变视频 token 数量；
- [ ] 旧 checkpoint 可加载，新 checkpoint 可 roundtrip；
- [ ] 数据 split 无泄漏；
- [ ] 伪标签经过人工抽检；
- [ ] E0-E7 使用同题、同评分器和固定 split；
- [ ] 音频必要题收益高于纯视觉题；
- [ ] 对齐与门控消融支持各自作用；
- [ ] 结果包含置信区间而不是单次最好值；
- [ ] 实时模式 RTF、延迟、显存和长流稳定性有报告；
- [ ] 所有关键模块都可通过配置关闭；
- [ ] 论文表述不超过实验能够支持的因果结论。

---

# 22. 主要参考资料

## 论文与核心代码

- OmniMMI: A Comprehensive Multi-modal Interaction Benchmark in Streaming Video Contexts
- M4: https://github.com/OmniMMI/M4
- OmniMMI evaluation: https://github.com/OmniMMI/OmniMMI
- MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing
- MUG repository: https://github.com/WangLY136/MUG

## 问答数据

- AVQA: https://github.com/AlyssaYoung/AVQA
- MUSIC-AVQA: https://github.com/GeWu-Lab/MUSIC-AVQA
- MUSIC-AVQA v2.0: https://huggingface.co/datasets/DraculaDragon/MUSIC-AVQA-v2.0

## 事件、对齐和长视频

- AVE: https://github.com/YapengTian/AVE-ECCV18
- LLP/AVVP: https://github.com/YapengTian/AVVP-ECCV20
- LongVALE: https://github.com/ttgeng233/LongVALE
- LongVALE data: https://huggingface.co/datasets/ttgeng233/LongVALE
- UnAV-100: https://github.com/ttgeng233/UnAV
- UnAV-100 data: https://huggingface.co/datasets/ttgeng233/UnAV-100

## 音频编码与噪声

- BEATs: https://github.com/microsoft/unilm/tree/master/beats
- PANNs: https://github.com/qiuqiangkong/audioset_tagging_cnn
- CLAP: https://github.com/LAION-AI/CLAP
- MUSAN: https://www.openslr.org/17/
- DNS Challenge: https://github.com/microsoft/DNS-Challenge
- LibriMix: https://github.com/JorisCos/LibriMix

---

# 23. 最终执行提醒

本项目最容易失败的方式不是“模型不够大”，而是：

- 音频与视频时间戳不可靠；
- 把问题语音和场景音频混为一谈；
- 直接融合坏音频；
- 使用测试集调阈值；
- 只报告 E2>E0；
- 伪标签没有人工抽检；
- 所有模块一次性接入，出现问题后无法定位。

因此，Codex 的首要任务始终是：保持阶段隔离、保留回退路径、运行 Harness、保存证据，然后再扩大模型和数据规模。
