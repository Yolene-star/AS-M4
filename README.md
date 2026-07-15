# AS-M4：Audio-Sensitive M4

AS-M4 是在 M4 / LongVA / LLaVA-NeXT / Qwen2 多模态交互框架上的本地扩展工程。原始 M4 主要处理视频/图像、文本问题和用户语音问题；AS-M4 在此基础上新增独立的场景音频 `scene_audio` 路径，用声音事件、音视频时间对齐、置信门控和残差融合提升流式视频理解能力。

原始 M4 README 已保留为 [`README_M4_ORIGINAL.md`](README_M4_ORIGINAL.md)。更完整的中文架构说明见 [`docs/M4_architecture_zh.md`](docs/M4_architecture_zh.md)，本地复现状态见 [`REPRODUCTION_STATUS.md`](REPRODUCTION_STATUS.md)，AS-M4 分阶段实现协议见 [`AS_M4_CODEX_IMPLEMENTATION_PLAYBOOK.md`](AS_M4_CODEX_IMPLEMENTATION_PLAYBOOK.md)。

## 当前工程目标

AS-M4 的目标不是重写 Qwen2、CLIP、Whisper 等大骨干，而是在原 M4 的视频 token 流上增加一条可关闭、可回退、可渐进训练的场景音频增强链路：

```text
视频帧 / 预计算视频特征
  -> CLIP / video_feature
  -> video tokens [B, T, N, H]

场景音频 scene_audio
  -> SceneAudioEncoder
  -> audio features [B, A, H]
  -> AudioEventDetector
  -> CausalTemporalAligner
  -> AudioConfidenceGate
  -> GatedAVFusion
  -> fused video tokens [B, T, N, H]

fused video tokens + text tokens
  -> Qwen2 Causal LM
```

第一版 AS-M4 遵守三条兼容约束：

1. `speech` / 用户问题语音与 `scene_audio` / 视频场景音频是两条独立输入职责。
2. 残差融合前后视频 token 数量保持一致，不新增 `<scene_audio>` 特殊 token，不改 tokenizer 词表。
3. 新增模块都必须支持配置关闭；关闭 `AS_M4_ENABLE_SCENE_AUDIO` 或强制 `AS_M4_FORCE_AUDIO_GATE=0` 时，应回退到原 M4 行为路径。

## 主要目录

```text
M4-main/
├── AGENTS.md                                  # 本机 GPU、训练和 Git 执行规则
├── AS_M4_CODEX_IMPLEMENTATION_PLAYBOOK.md     # AS-M4 分阶段实现手册
├── REPRODUCTION_STATUS.md                     # 本地复现状态记录
├── docs/
│   └── M4_architecture_zh.md                  # M4/AS-M4 中文架构说明
├── preprocess/
│   └── tts/                                   # 音频训练数据构建脚本
├── scripts/
│   └── download_omnimmi_assets.py             # OmniMMI 资源下载辅助脚本
└── intersuit/
    ├── intersuit/
    │   ├── model/
    │   │   ├── llava_arch.py                  # 多模态打包与 AS-M4 融合主逻辑
    │   │   ├── language_model/llava_qwen.py   # Qwen2 forward / chunked LM loss
    │   │   ├── scene_audio_encoder/           # 场景音频编码器
    │   │   └── streaming_av/                  # 事件、对齐、门控、融合模块
    │   ├── streaming/                         # 流式音视频 buffer / 切窗 / 调度工具
    │   └── train/                             # 数据集、collator、训练入口
    ├── scripts/                               # 训练、复现、评测脚本
    ├── tests/                                 # AS-M4 单元与集成测试
    └── local_demo/                            # M4 demo 入口
```

## 新增模块概览

`scene_audio_encoder` 提供场景音频编码接口：

- `DummySceneAudioEncoder`：用窗口波形统计量生成特征，用于 smoke test。
- `PrecomputedSceneAudioEncoder`：读取预计算音频特征，并对齐到模型 hidden size。

`streaming_av` 提供音视频融合组件：

- `AudioEventDetector`：预测音频窗口的事件类别、事件强度和边界。
- `CausalTemporalAligner`：在统一时间轴上做音频窗口与视频帧软对齐。
- `AudioConfidenceGate`：估计音频质量 `quality` 和相关性 `relevance`，门控为 `gate = quality * relevance`。
- `GatedAVFusion`：用残差方式把可信音频注入视频 token。

`streaming` 提供工程工具：

- `audio_stream.py`：从音频/视频文件加载场景音频，并切成带时间戳窗口。
- `av_buffer.py`：维护统一时间轴上的音频窗口和视频帧。
- `frame_scheduler.py`：根据音频事件强度做帧选择优先级调度。

## 环境约束

当前工程在本地服务器 `star-SYS-420GP-TNR` 上复现，关键环境如下：

- Ubuntu 20.04
- Python 3.10
- PyTorch `2.5.1+cu121`
- NCCL `2.21.5+cuda12.4`
- NVIDIA Driver `535.230.02`
- 5 张 NVIDIA GeForce RTX 4090，每张按 24 GB 显存估算

稳定四卡训练默认使用：

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4
```

通信友好的三卡组合：

```bash
CUDA_VISIBLE_DEVICES=2,3,4
```

不要默认使用 5 张卡。GPU 1 可用于隔离单卡任务，但在单独完成多卡稳定性验证前，不进入生产多卡训练。

## 安装

建议使用已有本地 conda 环境：

```bash
conda activate /home/yjm/miniconda3/envs/M4
cd /home/yjm/M4-main/intersuit
pip install -e .
```

如果需要从头安装，可参考原始 M4 README 和 `requirements.txt`。由于本项目依赖 CUDA、FlashAttention、DeepSpeed、Whisper、torchaudio 等组件，版本应优先与本机复现记录保持一致。

## 数据与权重

Git 仓库只跟踪代码、脚本、配置、文档和小型 manifest。以下运行资产不会进入 Git：

- `intersuit/checkpoints/`
- `intersuit/asset/`
- `VoiceAssistant-400K/`
- `m4-it/`
- `third_party/`
- 训练日志、缓存、音视频文件、模型权重文件

本地已验证的数据状态：

- `m4-it-qwen.json`：`9963` 条样本。
- 图像引用：`4052` 个，缺失数 `0`。
- 生成版音频 JSON：`intersuit/inputs/texts/m4-it-qwen-audio.generated.json`。
- 生成版音频 JSON 引用唯一 wav：`30188` 个，缺失数 `0`。

审计数据：

```bash
cd /home/yjm/M4-main/intersuit
python scripts/audit_m4_data.py
```

从已有 wav 构建带 `speech` 字段的音频训练 JSON：

```bash
python scripts/build_m4_audio_json.py
```

## 训练前检查

启动长训练前必须先检查 GPU 空闲和功率策略：

```bash
cd /home/yjm/M4-main/intersuit
python scripts/check_m4_repro_preflight.py
```

稳定四卡训练推荐环境：

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4 \
NCCL_DEBUG=WARN \
NCCL_P2P_DISABLE=1 \
NCCL_SHM_DISABLE=1 \
NCCL_IB_DISABLE=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

## M4 基线复现

当前默认保精度训练策略：

```bash
cd /home/yjm/M4-main/intersuit
CUDA_VISIBLE_DEVICES=0,2,3,4 \
NUM_GPUS=4 \
MODEL_MAX_LENGTH=12288 \
M4_CHUNKED_LM_LOSS=1 \
M4_CHUNKED_LM_LOSS_TOKENS=512 \
M4_CHUNKED_LM_LOSS_CHECKPOINT=0 \
M4_FREEZE_LM_HEAD=0 \
DEEPSPEED_CONFIG=scripts/zero3_lowmem.json \
bash scripts/finetune_m4.sh
```

已验证完成的 no-freeze 基线：

| 基线 | 输出目录 | 上下文 | 训练步数 | train_loss |
| --- | --- | ---: | ---: | ---: |
| 12k no-freeze | `intersuit/checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze` | 12288 | 622/622 | 0.06275078892921938 |
| 32k no-freeze | `intersuit/checkpoints/M4-LongVA-7B-Qwen2-repro-32k-lowmem-nofreeze-v1` | 32000 | 622/622 | 0.09159564487920577 |

32k 阶段复现入口：

```bash
bash scripts/run_m4_32k_repro_stage.sh 16k-canary150
bash scripts/run_m4_32k_repro_stage.sh 32k-canary20
bash scripts/run_m4_32k_repro_stage.sh 32k-canary150
bash scripts/run_m4_32k_repro_stage.sh 32k-full
```

## AS-M4 分阶段训练

12k 快速开发与 canary：

```bash
cd /home/yjm/M4-main/intersuit
bash scripts/run_as_m4_stage.sh 12k-smoke
bash scripts/run_as_m4_stage.sh 12k-canary20
bash scripts/run_as_m4_stage.sh 12k-canary150
bash scripts/run_as_m4_stage.sh 12k-full
```

32k 长上下文阶段：

```bash
bash scripts/run_as_m4_32k_stage.sh 32k-canary20
bash scripts/run_as_m4_32k_stage.sh 32k-canary150
bash scripts/run_as_m4_32k_stage.sh 32k-full
```

AS-M4 默认关键环境变量：

```bash
AS_M4_ENABLE_SCENE_AUDIO=1
AS_M4_SCENE_AUDIO_ENCODER_TYPE=dummy
AS_M4_STREAMING_AV_LR=1e-4
AS_M4_SCENE_AUDIO_PROJECTOR_LR=1e-4
MM_TUNABLE_PARTS=mm_vision_tower,mm_mlp_adapter,mm_language_model,streaming_av_module
```

## 行为回退验证

完全关闭场景音频路径：

```bash
AS_M4_ENABLE_SCENE_AUDIO=0 \
MM_TUNABLE_PARTS=mm_vision_tower,mm_mlp_adapter,mm_language_model \
bash scripts/run_as_m4_stage.sh 12k-smoke
```

保留编码/对齐链路，但强制残差门控为 0：

```bash
AS_M4_ENABLE_SCENE_AUDIO=1 \
AS_M4_FORCE_AUDIO_GATE=0 \
bash scripts/run_as_m4_stage.sh 12k-smoke
```

权重回退不要依赖“冻结模块”。如果要恢复原模型能力，必须显式加载保留的原始 M4、12k no-freeze 或 32k no-freeze checkpoint。

## 测试

AS-M4 新增模块对应测试位于 `intersuit/tests/`：

```bash
cd /home/yjm/M4-main/intersuit
pytest tests/test_audio_stream.py
pytest tests/test_av_buffer.py
pytest tests/test_event_detector.py
pytest tests/test_temporal_aligner.py
pytest tests/test_confidence_gate.py
pytest tests/test_fusion.py
pytest tests/test_streaming_av_integration.py
pytest tests/test_dataset_collator.py
```

如果环境中没有 pytest，可先用 Python 编译检查关键模块：

```bash
python -m py_compile \
  intersuit/model/scene_audio_encoder/scene_audio_encoder.py \
  intersuit/model/streaming_av/event_detector.py \
  intersuit/model/streaming_av/temporal_aligner.py \
  intersuit/model/streaming_av/confidence_gate.py \
  intersuit/model/streaming_av/fusion.py
```

## Demo

原 M4 demo 仍保留：

```bash
cd /home/yjm/M4-main/intersuit
./scripts/run_demos_stable.sh turntaking-valid
./scripts/run_demos_stable.sh turntaking-interrupt
./scripts/run_demos_stable.sh turntaking-noise
./scripts/run_demos_stable.sh proactive
./scripts/run_audio_demos_stable.sh baseline-audio-file
./scripts/run_audio_demos_stable.sh turntaking-audio-file
```

## Git 提交规则

每次完成重大设计变更、模块新增、模块删除或核心模块改动后，都要及时提交一次 Git commit，并推送到对应 GitHub 仓库。提交前必须确认没有把 checkpoint、权重、数据集、缓存、训练日志等运行资产加入 Git。

当前远端仓库：

```bash
git@github.com:Yolene-star/AS-M4.git
```

## 参考

- 原始 M4 README：[`README_M4_ORIGINAL.md`](README_M4_ORIGINAL.md)
- M4 架构中文说明：[`docs/M4_architecture_zh.md`](docs/M4_architecture_zh.md)
- 本地复现状态：[`REPRODUCTION_STATUS.md`](REPRODUCTION_STATUS.md)
- AS-M4 实现手册：[`AS_M4_CODEX_IMPLEMENTATION_PLAYBOOK.md`](AS_M4_CODEX_IMPLEMENTATION_PLAYBOOK.md)

