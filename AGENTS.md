# 服务器计算环境说明

本项目运行在 `star-SYS-420GP-TNR`，服务器型号为 Supermicro `SYS-420GP-TNR`。当前系统环境为 Ubuntu 20.04、Linux 5.15、NVIDIA 驱动 `535.230.02`、PyTorch `2.5.1+cu121`、NCCL `2.21.5+cuda12.4`，机器上共有 5 张 NVIDIA GeForce RTX 4090。

每张 RTX 4090 的可用显存按 24 GB 估算。多张显卡的显存不是统一显存，不能简单相加后当作单卡显存使用。标准 DDP 仍然受单个进程约 24 GB 显存限制；如果模型状态必须切分到多张卡上，需要使用 ZeRO、FSDP、张量并行或流水线并行。

## 文档语言要求

凡是为了帮助用户理解而新增或更新的说明类文本文件，默认使用中文撰写。包括但不限于复现报告、环境说明、操作步骤、排错说明和面向用户的脚本提示。

代码标识符、命令、路径、环境变量、模型名、文件名、参数名和必须与程序匹配的模式名可以保留英文原文，避免破坏可执行性。

## GPU 映射

| 物理 GPU | PCI Bus ID | NUMA 节点 | 状态 |
|---|---|---:|---|
| GPU 0 | `0000:4F:00.0` | 0 | 已验证 |
| GPU 1 | `0000:57:00.0` | 0 | 单卡已验证；多卡可疑 |
| GPU 2 | `0000:CE:00.0` | 1 | 已验证 |
| GPU 3 | `0000:D1:00.0` | 1 | 已验证 |
| GPU 4 | `0000:D5:00.0` | 1 | 已验证 |

当前机器没有 NVLink，也没有检测到 GPU P2P 支持。多 GPU 通信只走 PCIe。GPU 2、GPU 3、GPU 4 都在 NUMA 节点 1，是三卡通信密集任务的优先组合。

## 稳定默认配置

默认稳定四卡组合：

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4
```

默认通信友好的三卡组合：

```bash
CUDA_VISIBLE_DEVICES=2,3,4
```

生产训练不要默认使用 5 张卡。GPU 1 可以用于隔离的单卡任务，但在单独通过多卡稳定性验证前，不要放进生产多卡训练。

## 稳定 NCCL 环境

四卡稳定训练优先使用下面这组保守环境变量：

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4 \
NCCL_DEBUG=WARN \
NCCL_P2P_DISABLE=1 \
NCCL_SHM_DISABLE=1 \
NCCL_IB_DISABLE=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
torchrun \
  --standalone \
  --nproc_per_node=4 \
  train.py
```

这组配置优先保证稳定性，而不是追求最高通信性能。

## 当前 M4 训练默认策略

当前正式训练默认使用：

```bash
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

说明：

- `M4_CHUNKED_LM_LOSS=1` 会启用分块语言模型 loss，避免一次性构造完整 `seq_len x vocab_size` logits，已通过 20-step canary。
- `M4_CHUNKED_LM_LOSS_CHECKPOINT=0` 是当前默认值。此前设为 `1` 时会在 ZeRO-3 下触发 NCCL collective size mismatch；关闭它不改变 loss 目标，只是不再对 loss chunk 的 `lm_head` 做反向重算。
- 分块 loss 在 ZeRO-3 下必须保证所有 rank 执行相同次数的 `lm_head`，不能按各 rank 的有效 label 数各自决定循环次数。
- `scripts/zero3_cpu_offload.json` 只作为诊断备用；当前机器上 CPU offload 曾触发系统 OOM killer，不作为正式训练默认配置。
- `scripts/zero3_lowmem.json` 是当前 no-freeze 成功路线；它已在四卡 `0,2,3,4` 上完成 `MODEL_MAX_LENGTH=12288` 和 `MODEL_MAX_LENGTH=32000`、`M4_FREEZE_LM_HEAD=0` 的 1 epoch 正式训练。
- 如果使用 `scripts/zero3_lowmem.json` 从同一 lowmem run 的 checkpoint 恢复，需要显式设置 `M4_ALLOW_ZERO3_LOWMEM_RESUME=1`。不要把它用于恢复旧 `scripts/zero3.json` 产生的 optimizer state，否则可能触发 `parameter group mismatch`。
- `MODEL_MAX_LENGTH=16384` 曾通过 20-step canary，但早期完整训练在第 132 步附近触发反向传播 OOM；后续在 `zero3_lowmem`、分块 loss 和稳定四卡条件下，`MODEL_MAX_LENGTH=32000` 已完成正式训练。
- `SAVE_STEPS` 建议设为 `25` 或 `50`，避免长时间训练异常时完全丢失进度。
- `M4_FREEZE_LM_HEAD=1` 只保留为历史诊断折中；该方案虽然完成过训练，但最终视频人数回答出现偏移，不作为保精度默认策略。

当前已验证完成的 no-freeze 训练结果：

```text
输出目录：intersuit/checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze
最终 checkpoint：checkpoint-622
训练步数：622/622
train_runtime：41777.5616 秒
train_loss：0.06275078892921938
```

当前已验证完成的 32k no-freeze 训练结果：

```text
输出目录：intersuit/checkpoints/M4-LongVA-7B-Qwen2-repro-32k-lowmem-nofreeze-v1
训练步数：622/622
MODEL_MAX_LENGTH：32000
M4_FREEZE_LM_HEAD：0
train_runtime：43339.426 秒
train_loss：0.09159564487920577
```

后续 AS-M4 改进默认采用“双基线”策略：12k no-freeze 用于快速开发、模块 harness 和多数消融；32k no-freeze 用于最终长上下文主实验或论文级对照。AS-M4 新增模块导致 OOM 时，优先通过预计算冻结音频特征、activation checkpointing、减小 `M4_CHUNKED_LM_LOSS_TOKENS`、micro batch 降到 1 并增加梯度累积、分块计算对齐矩阵、受控 CPU optimizer offload 等方式牺牲时间保精度；不要优先冻结 `lm_head`、降低 `MODEL_MAX_LENGTH` 或削减视频/音频有效信息。

AS-M4 必须保留两层回退能力：

- 行为回退：新增 `scene_audio`、事件检测、时间对齐、门控和残差融合都必须有配置开关；关闭 `enable_scene_audio` 或强制 `gate=0` 时，应走回原 M4 的视频帧 + 用户问题文字/语音路径。
- 权重回退：冻结模块只能阻止参数继续更新，不能把已经训练改变的参数恢复到旧值。真正恢复原模型必须加载保留的原始 M4、12k no-freeze 或 32k no-freeze checkpoint。AS-M4 训练输出必须使用新的 run name，不得覆盖这些基线 checkpoint。

AS-M4 可执行回退配置示例：

```bash
# 行为回退：完全关闭场景音频路径，验证旧 M4 路径仍可用
AS_M4_ENABLE_SCENE_AUDIO=0 \
MM_TUNABLE_PARTS=mm_vision_tower,mm_mlp_adapter,mm_language_model \
bash scripts/run_as_m4_stage.sh 12k-smoke

# 残差融合回退：仍走 scene_audio 编码/对齐链路，但强制 gate=0，验证融合输出退化为原视频特征
AS_M4_ENABLE_SCENE_AUDIO=1 \
AS_M4_FORCE_AUDIO_GATE=0 \
bash scripts/run_as_m4_stage.sh 12k-smoke

# 权重回退：不要依赖冻结恢复旧权重，必须显式加载保留的基线 checkpoint
CKPT_PATH=checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze \
AS_M4_ENABLE_SCENE_AUDIO=0 \
bash scripts/run_as_m4_stage.sh 12k-smoke
```

## 论文/发布标准复现推进入口

当前新增了下面几个中文提示和中文报告优先的复现辅助脚本：

```bash
# 本地 M4-IT 图像/音频资源审计
python scripts/audit_m4_data.py

# 从已有 wav 构建带 speech 字段的 M4 音频训练 JSON
python scripts/build_m4_audio_json.py

# 四卡 16k/32k 阶梯 canary 与正式训练
bash scripts/run_m4_32k_repro_stage.sh 16k-canary150
bash scripts/run_m4_32k_repro_stage.sh 32k-canary20
bash scripts/run_m4_32k_repro_stage.sh 32k-canary150
bash scripts/run_m4_32k_repro_stage.sh 32k-full

# 音频三阶段复训
bash scripts/run_m4_audio_repro_stage.sh voiceassistant
bash scripts/run_m4_audio_repro_stage.sh llavanext-speech
bash scripts/run_m4_audio_repro_stage.sh m4-audio

# 训练前检查，确认稳定四卡空闲，并记录功率策略
python scripts/check_m4_repro_preflight.py
```

最新本地数据审计结果：

- `m4-it-qwen.json` 为 `9963` 条；
- 图像引用 `4052` 个，缺失数为 `0`；
- 生成版音频 JSON 为 `intersuit/inputs/texts/m4-it-qwen-audio.generated.json`；
- 生成版音频 JSON 引用唯一 wav `30188` 个，缺失数为 `0`；
- 重复 `id` 数量为 `2495`，当前只作为 warning，后续需要和 Hugging Face 官方清单对照。

注意：`intersuit/inputs/texts/m4-it-qwen-audio.json` 当前仍是视觉 JSON 的软链接，不含 `speech` 字段；正式 M4 audio 复训默认使用新生成的 `m4-it-qwen-audio.generated.json`。

启动 32k 四卡训练前必须先满足：

- GPU 0、2、3、4 空闲；
- 功率限制为已验证策略之一：`300W` 或 `450W`。`300W` 是保守稳定默认；`450W` 已验证可用于后续 32k/AS-M4 训练，但必须在日志和 `.env` 中记录；
- `python scripts/check_m4_repro_preflight.py` 通过。

如果 GPU 0 被其他用户任务占用，不要擅自杀进程；应暂停四卡训练，等待 GPU 0 释放，或由用户明确协调该任务。

## 执行规则

1. 不要默认使用所有可见 GPU。
2. 不要使用 `CUDA_VISIBLE_DEVICES=0,1,2,3,4`，除非用户明确要求做五卡诊断。
3. 稳定四卡任务默认使用 `CUDA_VISIBLE_DEVICES=0,2,3,4`。
4. 通信优化的三卡任务默认使用 `CUDA_VISIBLE_DEVICES=2,3,4`。
5. 单卡开发也要显式选择 GPU，不要依赖默认 CUDA 设备。
6. 设置 `CUDA_VISIBLE_DEVICES` 后，程序里的逻辑 CUDA 编号会重新映射。
7. 设置 `CUDA_VISIBLE_DEVICES` 后，不要在 Python 里硬编码原始物理 GPU 编号。
8. 分布式训练用 `LOCAL_RANK` 分配设备。
9. 每次稳定性测试都要保存 stdout、stderr、训练配置和开始时间。
10. 诊断测试结束后，检查内核日志里是否出现 `NVRM`、`Xid`、`AER`、`PCIe Bus Error`、`GPU has fallen`、`hung task`。
11. 如果训练进程停止前进并进入 `D` 状态，不要反复执行 `nvidia-smi`。
12. 每次完成重大设计变更、模块新增、模块删除或核心模块改动后，都要及时提交一次 Git commit。提交前必须确认没有把 checkpoint、权重、数据集、缓存、训练日志等运行资产加入 Git；提交信息要概括本次设计或模块变更。

逻辑编号映射示例：

```text
CUDA_VISIBLE_DEVICES=0,2,3,4
cuda:0 -> 物理 GPU 0
cuda:1 -> 物理 GPU 2
cuda:2 -> 物理 GPU 3
cuda:3 -> 物理 GPU 4
```

## 功率策略

已测试稳定的功率策略：

```bash
nvidia-smi -pm 1
nvidia-smi -pl 300
```

此外，`450W` 已验证可以完成 32k no-freeze full 训练。后续训练不再强制降到 `300W`，但必须记录实际功率限制。功率限制重启后可能会失效。可用下面命令检查：

```bash
nvidia-smi --query-gpu=index,pci.bus_id,power.limit --format=csv
```
