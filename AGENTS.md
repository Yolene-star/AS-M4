# 服务器计算环境说明

本项目运行在 `star-SYS-420GP-TNR`，服务器型号为 Supermicro `SYS-420GP-TNR`。当前系统环境为 Ubuntu 20.04、Linux 5.15、NVIDIA 驱动 `535.230.02`、PyTorch `2.5.1+cu121`、NCCL `2.21.5+cuda12.4`，机器上共有 5 张 NVIDIA GeForce RTX 4090。

每张 RTX 4090 当前按约 49 GB 可用显存规划。多张显卡的显存不是统一显存，不能简单相加后当作单卡显存使用。标准 DDP 仍然受单个进程约 49 GB 单卡显存限制；如果模型状态必须切分到多张卡上，需要使用 ZeRO、FSDP、张量并行或流水线并行。后续 AS-M4 训练默认至少保留 4 GB 左右瞬时余量，避免 CUDA context、NCCL buffer、临时 tensor 或 allocator cache flush 导致边界 OOM。

## 文档语言要求

凡是为了帮助用户理解而新增或更新的说明类文本文件，默认使用中文撰写。包括但不限于复现报告、环境说明、操作步骤、排错说明和面向用户的脚本提示。

代码标识符、命令、路径、环境变量、模型名、文件名、参数名和必须与程序匹配的模式名可以保留英文原文，避免破坏可执行性。

## Python/Conda 环境要求

后续所有 Python 相关操作，包括运行脚本、测试、训练、数据处理和安装依赖，都必须选择 Conda 中已经配置好的环境，不得直接使用未经确认的系统 Python、默认 shell 中的 Python 或临时新建环境。

- M4/AS-M4 项目代码默认使用已有的 `M4` 环境，例如 `conda run -n M4 python ...`、`conda run -n M4 pytest ...`，或先执行 `conda activate M4`。
- 如果某项任务确实需要其他依赖，必须先从 `conda env list` 中选择已经存在且匹配的环境；不得自行创建新环境或在不相关环境中安装依赖，除非用户明确同意。
- 每次执行 Python 命令前都要确认所选 Conda 环境；诊断环境问题时应记录环境名，并可通过 `python -c "import sys; print(sys.executable)"` 核对解释器路径。
- 自动化脚本、后台任务和长时间训练不能依赖交互式 shell 的隐式环境，必须显式激活 Conda 环境或使用 `conda run -n <环境名>`。

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
NCCL_SOCKET_IFNAME=lo \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
torchrun \
  --standalone \
  --nproc_per_node=4 \
  train.py
```

这组配置优先保证稳定性，而不是追求最高通信性能。

AS-M4 staged launcher 默认额外记录并设置：

```bash
NCCL_SOCKET_IFNAME=lo
```

原因：本机存在 `eth0=169.254.3.1` 链路本地地址；AS-M4 四卡 smoke 曾观察到 NCCL 自动选择该接口后出现 socket `message truncated` / collective size mismatch。单机 `nccl_probe.py` 已验证 `lo` 上 4-rank all-reduce 正常。若后续显式改用其他接口，必须在 `.env` 与训练日志中记录。

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

后续 AS-M4 改进默认采用“双基线”策略：12k no-freeze 用于快速开发、模块 harness 和多数消融；32k no-freeze 用于最终长上下文主实验或论文级对照。AS-M4 新增模块导致 32k OOM 时，必须优先牺牲时间来保精度。降级顺序为：

1. 重新运行 preflight，先排除残留进程、坏样本、缓存、磁盘、NCCL 接口或功率记录问题。
2. 离线预计算冻结场景音频特征，训练时不再运行 scene audio encoder forward。
3. 对 event detector、temporal aligner、confidence gate、fusion 和 projector 启用 activation checkpointing。
4. 将 `M4_CHUNKED_LM_LOSS_TOKENS` 从 `512` 降到 `256`，仍 OOM 再降到 `128`；这只增加 `lm_head` 循环次数，不改变 loss 目标。
5. 将 micro batch 降到 `1`，通过增加 gradient accumulation 保持有效 batch。
6. 对 temporal aligner 使用 chunked similarity，完整保留原定 offset 搜索范围，只分块计算相似度矩阵。
7. 减少训练时保留的诊断张量，只保存抽样统计，不保存全量 alignment matrix。
8. 受控尝试 ZeRO-3 optimizer CPU offload；必须先确认系统内存并做 canary，避免再次触发系统 OOM killer。
9. 以上仍失败时，才缩短训练时的 `audio_history_sec`，并保留评测配置、记录训练/评测差异。
10. 再失败时，才小幅减少对齐候选范围，并优先只作用于训练阶段。
11. 最后才考虑降低 LoRA rank 或冻结更多 AS-M4 新增模块。

不要优先冻结 `lm_head`、降低 `MODEL_MAX_LENGTH=32000`、削减视频 token/帧预算、删除有效场景音频信息，或关闭音频模块来让训练“通过”。

AS-M4 必须保留两层回退能力：

- 行为回退：新增 `scene_audio`、事件检测、时间对齐、门控和残差融合都必须有配置开关；关闭 `enable_scene_audio` 或强制 `gate=0` 时，应走回原 M4 的视频帧 + 用户问题文字/语音路径。
- 权重回退：冻结模块只能阻止参数继续更新，不能把已经训练改变的参数恢复到旧值。真正恢复原模型必须加载保留的原始 M4、12k no-freeze 或 32k no-freeze checkpoint。AS-M4 训练输出必须使用新的 run name，不得覆盖这些基线 checkpoint。

AS-M4 可执行回退配置示例：

```bash
# 简写：行为回退，关闭 scene_audio 并强制 gate=0
AS_M4_ROLLBACK_MODE=behavior \
bash scripts/run_as_m4_stage.sh 12k-smoke

# 简写：权重回退，加载保留的 12k 基线 checkpoint
AS_M4_ROLLBACK_MODE=weights12k \
bash scripts/run_as_m4_stage.sh 12k-smoke

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

`AS_M4_ROLLBACK_MODE` 可用值：

- `none`：默认 AS-M4 路径；
- `behavior`：关闭场景音频路径并强制 `gate=0`，用于确认新增路径不影响原 M4 行为；
- `gate0`：保留 scene audio 编码、事件、对齐等探针，但强制残差融合为 `gate=0`；
- `weights12k`：加载已保留的 12k no-freeze 基线 checkpoint，并关闭 scene audio；
- `weights32k`：加载已保留的 32k no-freeze 基线 checkpoint，并关闭 scene audio。

注意：`MM_TUNABLE_PARTS` 中移除 `streaming_av_module` 或冻结某些 AS-M4 模块，只能阻止这些模块继续更新；如果已经使用 AS-M4 checkpoint 改动过 Qwen2、视觉塔或 projector，必须使用 `weights12k/weights32k` 或显式 `CKPT_PATH` 才能恢复原先权重。

AS-M4 Stage A 只训练新增 `streaming_av_module` 时，优先使用普通 DDP smoke 路线：

```bash
DEEPSPEED_CONFIG=none \
TORCHDYNAMO_DISABLE=1 \
DDP_FIND_UNUSED_PARAMETERS=True \
MM_TUNABLE_PARTS=streaming_av_module \
bash scripts/run_as_m4_stage.sh 12k-smoke
```

原因：该阶段可训练参数约 `22 MB`，49GB 单卡足以复制冻结基座；使用 DDP-only 可以避开 ZeRO-3 对极小新增参数组的参数 broadcast collective mismatch。`TORCHDYNAMO_DISABLE=1` 用于避免 DDPOptimizer higher-order op 编译失败。`DDP_FIND_UNUSED_PARAMETERS=True` 用于允许尚未接入辅助 loss 的 event/offset/gate head 暂时无梯度。完整 12k/32k 主训练或训练语言模型/LoRA 时，仍应回到 `scripts/zero3_lowmem.json`，并重新执行 `2-step -> 20-step -> 150-step` canary。

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

## Codex 沙箱与数据下载权限

`AGENTS.md` 只能记录项目约定，不能在代码层面直接提升 Codex 沙箱权限。需要联网下载数据、写入 Git index 或访问代理端口时，必须在 Codex 运行配置、CLI 启动参数或权限审批层显式授予；如果审批器拒绝，不能通过替代命令、脚本包装或其他间接方式绕过。

AVE/LLP/AVQA 等外部数据准备任务若由 Codex 执行，至少需要允许：

- 网络访问：`https://hf-mirror.com` 和必要时 `https://huggingface.co`。
- 本机代理端口：`127.0.0.1:7890`、`127.0.0.1:7891`，前提是用户确认代理服务正在运行。
- 环境变量继承：`HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY`、`HF_ENDPOINT`、`NO_PROXY`。
- 命令前缀：`python intersuit/scripts/prepare_ave_hf_pilot.py`、`python intersuit/scripts/extract_ave_hf_window_features.py`、`python intersuit/scripts/extract_ave_hf_clip_window_features.py`、`python intersuit/scripts/run_ave_hf_temporal_offset_frozen_seeds.py`。
- Git 写入前缀：`git add`、`git commit`；提交前必须检查 staged 文件，确认未加入媒体、特征、checkpoint、缓存或训练日志。

当前 AVE 扩展下载优先使用用户配置的镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com \
python intersuit/scripts/prepare_ave_hf_pilot.py \
  --output-root intersuit/datasets/AVE_HF_EXPANDED \
  --limit 1000 \
  --threshold 900 \
  --parquet-file all_train \
  --selection diverse_label \
  --exclude-manifest intersuit/datasets/AVE_HF_300/ave_hf_pilot_valid.jsonl \
  --disable-xet
```

若出现 `Operation not permitted`，说明 Codex 沙箱未获得网络 socket 权限；若出现 `Network is unreachable` 或 DNS 解析失败，说明服务器或代理链路不可达。上述情况都应立即停止并报告原始错误，不要切换到不明来源、不要升级环境依赖、不要伪造缺失媒体。

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
12. 每次训练结束后都要及时清理无用 checkpoint：成功训练只保留明确需要复现、回退或继续训练的最终/关键 checkpoint；失败训练也要尽快删除失败中间产物、短步 smoke/canary 产物和重复保存的模型权重，避免 `intersuit/checkpoints` 长期堆积。清理前必须确认不会删除当前正式基线、原始模型、仍需 resume 的 checkpoint 或用户明确要求保留的结果。
13. 每次完成重大设计变更、模块新增、模块删除或核心模块改动后，都要及时提交一次 Git commit。提交前必须确认没有把 checkpoint、权重、数据集、缓存、训练日志等运行资产加入 Git；提交信息要概括本次设计或模块变更。

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
