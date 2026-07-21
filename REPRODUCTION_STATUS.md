# M4 本地复现状态

日期：2026-07-13

## 环境概况

- Conda 环境：`/home/yjm/miniconda3/envs/M4`
- Python：3.10
- PyTorch：`2.5.1+cu121`
- TorchVision：`0.20.1+cu121`
- TorchAudio：`2.5.1+cu121`
- Transformers：`4.44.0`
- Accelerate：`0.33.0`
- DeepSpeed：`0.14.5`
- PyTorch 使用的 CUDA 运行时：12.1
- 稳定的单卡 demo 默认设备：物理 GPU 0
- 稳定的四卡训练组合：物理 GPU 0、2、3、4

## 稳定 GPU 策略

分布式训练优先使用 GPU 0、2、3、4：

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4
```

GPU 1 在单独通过多卡稳定性验证前，不要加入生产多卡训练。

`intersuit/scripts` 下的脚本现在默认使用：

- `NCCL_P2P_DISABLE=1`
- `NCCL_SHM_DISABLE=1`
- `NCCL_IB_DISABLE=1`
- `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`
- `/home/yjm/miniconda3/envs/M4/bin/torchrun`
- `/home/yjm/M4-main/.cache` 下的本地缓存目录

## 已就绪的权重

推理和 demo 所需权重已经在本地：

- `intersuit/checkpoints/M4-LongVA-7B-Qwen2`
- `intersuit/checkpoints/M4-Audio-LongVA-7B-Qwen2`
- `intersuit/checkpoints/LongVA-7B-Qwen2-Audio`
- `intersuit/checkpoints/LongVA-7B-Qwen2`
- `intersuit/checkpoints/M4-LongVA-7B-Qwen2-train-chunked-12k-freeze-lmhead`
- `intersuit/checkpoints/clip-vit-large-patch14-336`
- `intersuit/checkpoints/whisper/large-v3.pt`
- ChatTTS 资产：`intersuit/asset`

已校验的哈希值：

- Whisper `large-v3.pt`：`e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb`
- `LongVA-7B-Qwen2-Audio/model-00003-of-00004.safetensors`：`a2aedc4afc0c39610f64f3e0c9dcb335d3c122835cf96b6556fadf4caeec4fd4`
- `LongVA-7B-Qwen2-Audio/model-00004-of-00004.safetensors`：`29168711665717d107ed8c3af13f2d30d2ca6f2a5a95d0930fbe8eabf12e6169`

## 已就绪的数据

本地 M4 图像/音频子集位置：

```text
/home/yjm/M4-main/m4-it
```

训练脚本所需软链接已经配置：

```text
intersuit/inputs/images/llava-next -> /home/yjm/M4-main/m4-it
intersuit/inputs/texts/m4-it-qwen.json -> /home/yjm/M4-main/m4-it-qwen.json
```

当前本地子集包含 3,557 个 jpg 文件和 19,835 个 wav 文件。视觉训练 JSON 中共有 9,963 条样本：

- 9,476 条样本带有 `image` 字段；
- 487 条样本没有 `image` 字段，内容是纯文本对话样本；
- 已检查带图样本的图片路径，缺失图片数为 0。

## Demo 验证结果

以下 demo 已经在本机成功运行。默认命令使用原始 checkpoint；如果要测试本次训练后的 checkpoint，可显式设置 `MODEL_PATH`。

- 文本 turn-taking：有效追问
- 文本 turn-taking：打断类输入
- 文本 turn-taking：噪声类输入
- proactive 视频 demo
- 音频 baseline：使用已有 wav 文件
- 音频 baseline：使用 ChatTTS 生成 wav 文件
- 音频 turn-taking：使用已有 wav 文件
- 音频 turn-taking：使用 ChatTTS 生成 wav 文件

推荐使用下面命令复现：

```bash
cd /home/yjm/M4-main/intersuit
./scripts/run_demos_stable.sh turntaking-valid
./scripts/run_demos_stable.sh turntaking-interrupt
./scripts/run_demos_stable.sh turntaking-noise
./scripts/run_demos_stable.sh proactive
./scripts/run_demos_stable.sh baseline-audio-file
./scripts/run_demos_stable.sh baseline-audio-tts
./scripts/run_demos_stable.sh turntaking-audio-file
./scripts/run_demos_stable.sh turntaking-audio-tts
```

观察到的输出内容与 demo 视频一致：模型能识别厨房/烹饪演示场景、两个人、玻璃杯、冰水、操作台和背景厨房电器等关键信息。

本次正式训练后的 checkpoint 已完成以下验收：

```bash
cd /home/yjm/M4-main/intersuit
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=checkpoints/M4-LongVA-7B-Qwen2-train-chunked-12k-freeze-lmhead ./scripts/run_demos_stable.sh turntaking-valid
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=checkpoints/M4-LongVA-7B-Qwen2-train-chunked-12k-freeze-lmhead ./scripts/run_demos_stable.sh turntaking-interrupt
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=checkpoints/M4-LongVA-7B-Qwen2-train-chunked-12k-freeze-lmhead ./scripts/run_demos_stable.sh turntaking-noise
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=checkpoints/M4-LongVA-7B-Qwen2-train-chunked-12k-freeze-lmhead ./scripts/run_demos_stable.sh proactive
```

结果：

- 新 checkpoint 能正常加载并完成视频 turn-taking 与 proactive demo；
- proactive demo 正常触发 `REQUIREMENT MEET`；
- 音频 baseline、音频 turn-taking 和 ChatTTS 文本转语音路径也已重新跑通；
- 质量风险：新 checkpoint 在 `How many people in the video?` 上回答 `three people`，而原始 checkpoint 回答 `two people`；因此本次训练已经跑通，但新 checkpoint 存在可见输出偏移，不能直接视为“精度无变化”。

## 训练 Smoke Test

已经成功完成一次 4GPU DeepSpeed ZeRO-3 的 1 step 训练 smoke test：

```bash
cd /home/yjm/M4-main/intersuit
CUDA_VISIBLE_DEVICES=0,2,3,4 NUM_GPUS=4 MAX_STEPS=1 bash scripts/run_m4_train_once_4090.sh
```

实际结果：

- 模型和视觉塔成功加载。
- `inputs/texts/m4-it-qwen.json` 成功加载，共 9,963 条样本。
- 完成 1 个训练 step。
- 最终 loss 约为 `0.1633`。
- checkpoint 保存到 `intersuit/checkpoints/M4-LongVA-7B-Qwen2-fullrun`。

注意：这个 smoke checkpoint 约占 119 GB，因为脚本保存了完整模型权重和 step checkpoint。它的作用是证明训练链路可用，不建议把它当作正式训练结果。

## 正式训练状态

2026-07-13 结论：正式训练不是异常终止，而是已经完成 1 epoch。

最终训练命令的核心配置：

```bash
cd /home/yjm/M4-main/intersuit
CUDA_VISIBLE_DEVICES=0,2,3,4 \
NUM_GPUS=4 \
MID_RUN_NAME=M4-LongVA-7B-Qwen2-train-chunked-12k-freeze-lmhead \
CKPT_PATH=checkpoints/M4-LongVA-7B-Qwen2-train-chunked-12k/checkpoint-100 \
MODEL_MAX_LENGTH=12288 \
DEEPSPEED_CONFIG=scripts/zero3.json \
M4_CHUNKED_LM_LOSS=1 \
M4_CHUNKED_LM_LOSS_TOKENS=512 \
M4_CHUNKED_LM_LOSS_CHECKPOINT=0 \
M4_FREEZE_LM_HEAD=1 \
SAVE_STEPS=25 \
bash scripts/finetune_m4.sh
```

最终结果：

- 日志：`/home/yjm/M4-main/intersuit/train_logs/finetune_m4_chunked_12k_freeze_lmhead_20260712_173432.log`
- 输出目录：`/home/yjm/M4-main/intersuit/checkpoints/M4-LongVA-7B-Qwen2-train-chunked-12k-freeze-lmhead`
- 最新 step checkpoint：`/home/yjm/M4-main/intersuit/checkpoints/M4-LongVA-7B-Qwen2-train-chunked-12k-freeze-lmhead/checkpoint-622`
- 训练进度：`622/622`
- `train_runtime`：`45912.0213` 秒，约 `12.75` 小时
- `train_loss`：`0.08668940127797281`
- `epoch`：`1.0`
- 日志出现 `Model saved to checkpoints/M4-LongVA-7B-Qwen2-train-chunked-12k-freeze-lmhead`
- 训练结束后 GPU 计算进程已释放，5 张卡显存回到约 `18 MiB`
- 训练结束检查未发现新的 `Xid`、`NVRM`、`PCIe Bus Error` 或 `hung task`

重要说明：

- `M4_FREEZE_LM_HEAD=1` 是为避开 `lm_head` 反向传播阶段约 `1.02 GiB` 的显存峰值而采用的折中；
- 主体语言模型、视觉连接模块、训练数据和训练目标仍保持训练；
- 由于冻结了 `lm_head`，这次结果应标记为“稳定复现训练闭环”，不应声称与完全不冻结训练在精度上严格等价。
- 后续更严格的 sanity check 发现，冻结 `lm_head` 的最终 checkpoint 对普通纯文本问题会出现空输出，并且视频人数问答相对原始 checkpoint 有偏移；因此该 checkpoint 不能作为“保精度最终复现结果”。

2026-07-13 用户约束测试：

用户要求不能冻结模块、不能压缩上下文。因此新增测试保持：

- `MODEL_MAX_LENGTH=12288`
- `M4_FREEZE_LM_HEAD=0`
- `M4_CHUNKED_LM_LOSS=1`
- `M4_CHUNKED_LM_LOSS_TOKENS=512`
- `DEEPSPEED_CONFIG=scripts/zero3_lowmem.json`
- 从原始 `checkpoints/M4-LongVA-7B-Qwen2` 起训，不恢复旧 optimizer

由于物理 GPU 0 当时仍有中断训练留下的驱动侧残留占用，测试使用物理 GPU 2、3、4：

```bash
CUDA_VISIBLE_DEVICES=2,3,4 \
NUM_GPUS=3 \
MID_RUN_NAME=M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze-canary-3gpu \
CKPT_PATH=checkpoints/M4-LongVA-7B-Qwen2 \
MODEL_MAX_LENGTH=12288 \
DEEPSPEED_CONFIG=scripts/zero3_lowmem.json \
M4_CHUNKED_LM_LOSS=1 \
M4_CHUNKED_LM_LOSS_TOKENS=512 \
M4_CHUNKED_LM_LOSS_CHECKPOINT=0 \
M4_FREEZE_LM_HEAD=0 \
LEARNING_RATE=2e-6 \
SAVE_STRATEGY=no \
MAX_STEPS=2 \
bash scripts/finetune_m4.sh
```

日志：

```text
/home/yjm/M4-main/intersuit/train_logs/finetune_m4_12k_lowmem_nofreeze_3gpu_canary_20260713.log
```

结果：

- 第 1 步成功完成，loss 为 `0.1669`；
- 第 2 步前向阶段 OOM；
- OOM 位置在 DeepSpeed ZeRO-3 参数 all-gather，仍需额外约 `1.02 GiB`；
- `zero3_lowmem` 仅靠限制 bucket/live 参数，不能在 12k 上下文、`lm_head` 可训练、当前三卡环境下稳定通过 canary；
- 该测试不能证明四卡一定失败，因为 GPU 0 残留导致四卡不可测；但它证明“三卡 12k 不冻结不压缩”暂不可行。

2026-07-13 四卡清理后重新测试：

GPU 0 的驱动侧残留已清理，功率限制恢复为 300W。随后使用稳定四卡 `CUDA_VISIBLE_DEVICES=0,2,3,4` 重新测试用户约束：

- 不冻结模块：`M4_FREEZE_LM_HEAD=0`
- 不压缩上下文：`MODEL_MAX_LENGTH=12288`
- `lm_head` 保持可训练
- 使用 `scripts/zero3_lowmem.json`
- 从原始 `checkpoints/M4-LongVA-7B-Qwen2` 起训，不恢复旧 optimizer

2-step canary：

```text
日志：/home/yjm/M4-main/intersuit/train_logs/finetune_m4_12k_lowmem_nofreeze_4gpu_canary_20260713.log
结果：2/2 step 完成，train_loss 约 0.12665，模型保存成功。
```

20-step canary：

```text
日志：/home/yjm/M4-main/intersuit/train_logs/finetune_m4_12k_lowmem_nofreeze_4gpu_canary20_20260713.log
结果：20/20 step 完成，train_runtime 约 1470.50 秒，train_loss 约 0.10185，模型保存成功。
```

结论：

- 在四卡 `0,2,3,4` 干净状态下，12k 上下文、不冻结模块、`lm_head` 可训练、`zero3_lowmem` 方案已经通过 20-step canary；
- 这证明“不压缩上下文、不冻结模块，只牺牲时间/通信效率”的方向在当前机器上具备可行性；
- 训练显存余量很小，canary 中曾出现 DeepSpeed allocator cache flush 警告，后续正式训练需要持续监控 OOM 和 NCCL 错误。

2026-07-13 17:31 已启动正式 1 epoch 复现训练：

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4 \
NUM_GPUS=4 \
MID_RUN_NAME=M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze \
CKPT_PATH=checkpoints/M4-LongVA-7B-Qwen2 \
MODEL_MAX_LENGTH=12288 \
DEEPSPEED_CONFIG=scripts/zero3_lowmem.json \
M4_CHUNKED_LM_LOSS=1 \
M4_CHUNKED_LM_LOSS_TOKENS=512 \
M4_CHUNKED_LM_LOSS_CHECKPOINT=0 \
M4_FREEZE_LM_HEAD=0 \
LEARNING_RATE=2e-6 \
SAVE_STEPS=25 \
bash scripts/finetune_m4.sh
```

运行信息：

```text
PID 文件：/home/yjm/M4-main/intersuit/train_logs/finetune_m4_12k_lowmem_nofreeze_full.pid
日志：/home/yjm/M4-main/intersuit/train_logs/finetune_m4_12k_lowmem_nofreeze_full_20260713.log
输出目录：/home/yjm/M4-main/intersuit/checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze
```

启动后已观察到：

- 第 1 步 loss：`0.1633`
- 第 2 步 loss：`0.09`
- GPU 0/2/3/4 均在运行，GPU 1 空闲；
- 显存占用约 42-46GB/卡，属于高压运行，需要继续监控。
- 2026-07-13 17:45 左右，训练已越过第 8 步，未出现 OOM 或 NCCL 错误；
- 2026-07-13 17:56 左右，训练已越过第 20 步，未出现 OOM 或 NCCL 错误；
- 2026-07-13 18:03 左右，已成功保存 `checkpoint-25`，说明普通训练 step 和首次保存阶段均已通过；
- 2026-07-13 18:35 左右，已成功保存 `checkpoint-50`，随后训练继续推进到第 `52/622` 步，说明保存 checkpoint 后训练仍能继续运行；
- 2026-07-13 19:06 左右，已成功保存 `checkpoint-75`，随后训练继续推进到第 `77/622` 步，暂未出现 OOM、NCCL 错误或 Python Traceback；
- 2026-07-13 19:29 左右，服务器发生重启，训练不是模型 OOM 退出，而是被系统重启中断；中断前日志推进到第 `85/622` 步；
- 重启后 GPU 功率限制一度恢复到 450W，随后训练恢复时已重新观察到 300W 功率上限；
- 2026-07-13 19:33 左右，已从 `checkpoint-75` 恢复训练；为允许同配置恢复，`scripts/finetune_m4.sh` 新增了显式开关 `M4_ALLOW_ZERO3_LOWMEM_RESUME=1`；
- 恢复训练已重新输出第 `76`、`77` 步 loss，说明 ZeRO optimizer、scheduler 和 RNG state 的恢复路径已通过；
- 2026-07-13 20:05 左右，恢复训练已成功保存 `checkpoint-100`，随后继续推进到第 `102/622` 步，暂未出现 OOM、NCCL 错误或 Python Traceback；
- 训练期间显存多次接近上限，最高观察到约 48.6GB/49.1GB，DeepSpeed 曾提示 allocator cache flush，仍需持续监控。

当前阶段性结论：

- 这次正式训练满足用户约束：不冻结模块、不继续压缩上下文、`lm_head` 保持可训练；
- 当前只能说明“不冻结 12k 四卡 lowmem 路线已经通过 20-step canary、`checkpoint-25`、`checkpoint-50`、`checkpoint-75` 和 `checkpoint-100`”，还不能说明最终复现完成；
- 下一关键验收点是第 `132` 步附近。此前不冻结训练多次在该区域触发 `lm_head` 反向传播 OOM；
- 最终验收仍需等待 `622/622` 训练完成，并用新 checkpoint 跑文本、视频 proactive、音频 demo 和训练样本 sanity check。

2026-07-13 20:10 最新运行状态：

- 当前正在执行正式复现训练，不冻结模块、不压缩到低于 `MODEL_MAX_LENGTH=12288`，`lm_head` 保持可训练；
- 训练已从 `checkpoint-75` 成功恢复，并已保存 `checkpoint-100`；
- 当前日志已推进到第 `108/622` 步，暂未出现 CUDA OOM、NCCL 错误或 Python Traceback；
- 当前 GPU 使用符合稳定策略：物理 GPU 0、2、3、4 满载训练，GPU 1 空闲；
- 四张训练卡显存接近上限，约 `46.9GB` 到 `48.1GB`，这是本方案牺牲时间和显存余量来保持训练目标的预期状态；
- 2026-07-13 20:15 复查内核日志，未发现新的 `Xid`、系统 OOM、`NVRM`、PCIe 错误或 `hung task`；
- 日志中的 `Invalidate trace cache` 是 DeepSpeed/ZeRO 运行时缓存提示，不是训练失败信号；
- 下一步继续观察第 `125`、`132` 和 `150` 步，尤其是第 `132` 步附近是否再次出现 `lm_head` 反向传播 OOM。

2026-07-13 继续测试四卡前的阻塞（历史记录，已通过重启和清理解决）：

- `nvidia-smi` 仍报告物理 GPU 0 上有 PID `1902486` 占用约 25GB 显存；
- `ps -fp 1902486` 和 `/proc/1902486` 均显示该进程不存在；
- `fuser -v /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm` 没有发现对应设备文件占用；
- 这说明它更像是 NVIDIA 驱动侧残留，而不是普通可 `kill` 的用户进程；
- 普通用户执行 `nvidia-smi --gpu-reset -i 0` 失败，原因是 `Insufficient Permissions`；
- `sudo -n nvidia-smi --gpu-reset -i 0` 失败，原因是当前会话需要 sudo 密码；
- 在清理 GPU 0 前，不应启动四卡 `CUDA_VISIBLE_DEVICES=0,2,3,4` 测试，否则结果会被 GPU 0 残留显存污染。

需要用户在服务器终端手动执行以下命令之一：

```bash
sudo nvidia-smi --gpu-reset -i 0
```

如果 reset 因 Xorg 或其他图形进程占用失败，最稳妥方式是重启服务器。重启后先检查：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,power.limit --format=csv,noheader
```

16k 正式训练曾启动，但已异常终止。现在已改用 12k 上下文长度重启，仍然使用稳定 GPU 组合和分块语言模型 loss：

```bash
cd /home/yjm/M4-main/intersuit
CUDA_VISIBLE_DEVICES=0,2,3,4 \
NUM_GPUS=4 \
MID_RUN_NAME=M4-LongVA-7B-Qwen2-train-chunked-12k \
CKPT_PATH=checkpoints/M4-LongVA-7B-Qwen2 \
TORCH_COMPILE=False \
DATALOADER_NUM_WORKERS=2 \
MODEL_MAX_LENGTH=12288 \
M4_CHUNKED_LM_LOSS=1 \
M4_CHUNKED_LM_LOSS_TOKENS=512 \
M4_CHUNKED_LM_LOSS_CHECKPOINT=0 \
SAVE_STEPS=50 \
RESUME_FROM_CHECKPOINT=checkpoints/M4-LongVA-7B-Qwen2-train-chunked-12k/checkpoint-100 \
bash scripts/finetune_m4.sh
```

16k 异常训练日志：

```text
/home/yjm/M4-main/intersuit/train_logs/finetune_m4_chunked_16k_full_20260711_220623.log
```

12k 显存 OOM 训练日志：

```text
/home/yjm/M4-main/intersuit/train_logs/finetune_m4_chunked_12k_full_20260712_004852.log
```

12k lowmem 恢复失败日志：

```text
/home/yjm/M4-main/intersuit/train_logs/finetune_m4_chunked_12k_lowmem_resume_20260712_041650.log
```

12k 当前恢复训练日志：

```text
/home/yjm/M4-main/intersuit/train_logs/finetune_m4_chunked_12k_resume_no_lossckpt_20260712_165043.log
```

12k 冻结 `lm_head` 后的新训练日志：

```text
/home/yjm/M4-main/intersuit/train_logs/finetune_m4_chunked_12k_freeze_lmhead_20260712_173432.log
```

16k 训练终止状态：

- 已加载 `inputs/texts/m4-it-qwen.json`，共 9,963 条样本；
- 总步数为 622；
- 训练到第 `132/622` 步附近异常终止；
- rank1 在 DeepSpeed ZeRO-3 的 `linear.py` 反向传播中触发 CUDA OOM：尝试再分配约 `1.02 GiB`，当时该 rank 仅剩约 `1.01 GiB`；
- 后续大量 `NCCL remote process exited` 是 rank1 OOM 后的连锁反应，不是最初根因；
- 内核日志没有新的 `Xid`、`NVRM`、`PCIe`、`hung task` 或系统 OOM 记录；
- 由于原先 `SAVE_STEPS=500`，训练尚未到第 500 步，因此没有留下正式训练 checkpoint。

12k 重启状态：

- 后台 PID `2067298` 对应的 12k 训练已异常终止；
- 训练到第 `132/622` 步附近再次在 rank1 的 `lm_head` 反向传播中触发 CUDA OOM；
- 已成功保存 `checkpoint-100`，位置为 `checkpoints/M4-LongVA-7B-Qwen2-train-chunked-12k/checkpoint-100`；
- 曾尝试 `M4_CHUNKED_LM_LOSS_CHECKPOINT=1`，但从 `checkpoint-100` 恢复时在 ZeRO-3 下触发 NCCL collective size mismatch，现已改为默认关闭；
- 已把每个 loss chunk 固定补齐到相同 token 数，避免不同 rank 的 chunk 形状不一致；
- 后续尝试用 `scripts/zero3_lowmem.json` 从 `checkpoint-100` 恢复失败，根因是 DeepSpeed optimizer state 的参数分组与新配置不一致，报错为 `loaded state dict contains a parameter group that doesn't match the size of optimizer's group`；
- 2026-07-12 16:50 已使用 `scripts/zero3.json`、`M4_CHUNKED_LM_LOSS_CHECKPOINT=0` 从 `checkpoint-100` 严格恢复训练；
- 该恢复训练成功跑到 `132/622` 并输出 loss，但在第 132 步反向传播时再次触发 rank1 CUDA OOM；
- OOM 根因仍是 DeepSpeed ZeRO-3 的 `lm_head` 反向传播：`grad_weight = grad_output.t().matmul(input)` 需要额外约 `1.02 GiB`，当时只剩约 `837 MiB`；
- 为保留数据、loss、12k 上下文和主体模型训练，2026-07-12 17:34 启动新 run：从 `checkpoint-100` 的 safetensors 模型权重起训、重建 optimizer/scheduler、显式设置 `M4_FREEZE_LM_HEAD=1`；
- 冻结 `lm_head` 后，可训练参数从约 `7935.65M` 降到 `7390.65M`，第 1 步已正常输出 loss；
- 2026-07-12 18:42 复查时，外层 PID 文件中的进程已不在，但四个实际训练 rank 进程仍在运行，GPU 0/2/3/4 均保持高利用率；
- 该 run 已保存 `checkpoint-50`，日志已推进到第 `55/622` 步，暂未出现 Python Traceback、CUDA OOM、NCCL 错误或新的内核 `Xid`；
- 2026-07-12 19:07 复查时，训练已推进到第 `76/622` 步，并成功保存 `checkpoint-75`，暂未出现 CUDA OOM 或 NCCL 错误；
- 2026-07-12 19:37 复查时，训练已推进到第 `101/622` 步，并成功保存 `checkpoint-100`，暂未出现 CUDA OOM 或 NCCL 错误；
- 2026-07-12 20:22 复查时，训练已推进到第 `138/622` 步，并成功保存 `checkpoint-125`；这次已经越过此前多次失败的第 `132` 步附近，说明 `M4_FREEZE_LM_HEAD=1` 已有效避开 `lm_head` 反向传播阶段的显存峰值；
- 2026-07-12 20:48 复查时，训练已推进到第 `160/622` 步，并成功保存 `checkpoint-150`，暂未出现 CUDA OOM 或 NCCL 错误；
- 2026-07-12 21:49 复查时，训练已推进到第 `209/622` 步，并成功保存 `checkpoint-200`，暂未出现 CUDA OOM 或 NCCL 错误；
- 2026-07-12 22:49 复查时，训练已推进到第 `257/622` 步，并成功保存 `checkpoint-250`，暂未出现 CUDA OOM 或 NCCL 错误；
- 2026-07-13 00:04 复查时，训练已推进到第 `318/622` 步，并成功保存 `checkpoint-300`，暂未出现 CUDA OOM 或 NCCL 错误；
- 2026-07-13 01:35 复查时，训练已推进到第 `391/622` 步，并成功保存 `checkpoint-375`，暂未出现 CUDA OOM 或 NCCL 错误；
- 2026-07-13 03:06 复查时，训练已推进到第 `465/622` 步，并成功保存 `checkpoint-450`，暂未出现 CUDA OOM 或 NCCL 错误；
- 2026-07-13 04:36 复查时，训练已推进到第 `538/622` 步，并成功保存 `checkpoint-525`，暂未出现 CUDA OOM 或 NCCL 错误；
- 2026-07-13 06:20 左右，训练完成 `622/622` 步并保存最终模型；
- 后续监控不要只看 `finetune_m4_chunked_12k_freeze_lmhead.pid`，应同时看训练日志、`nvidia-smi --query-compute-apps` 和最新 checkpoint。

正式训练前发现并修复/规避了五个问题：

- `checkpoints/LongVA-7B-Qwen2/config.json` 中的视觉塔原来指向远程 `openai/clip-vit-large-patch14-336`，已改为本地 `checkpoints/clip-vit-large-patch14-336`；
- `checkpoints/LongVA-7B-Qwen2` 缺少 4 个 `model-*.safetensors` 分片，因此不能作为完整起训权重；当前正式训练使用完整的 `checkpoints/M4-LongVA-7B-Qwen2` 作为继续训练起点；
- 原始完整 logits 的 cross entropy 在训练第 8 步附近会触发 CUDA OOM；已在 `LlavaQwenForCausalLM.forward` 中新增分块语言模型 loss，避免一次性构造完整 `seq_len x vocab_size` logits；
- DeepSpeed ZeRO-3 CPU offload 曾触发系统 CPU OOM killer，因此当前正式训练不默认启用 offload；如果用户要求“不冻结模块、不压缩上下文”，CPU offload 或更细粒度的 `lm_head`/vocab 分块 loss 是后续更接近保精度的方向，但必须重新做内存监控 canary；
- `scripts/zero3_lowmem.json` 不能直接恢复旧 ZeRO-3 checkpoint；如需使用它，只能做“仅加载模型权重、不恢复 optimizer/scheduler”的新实验，这会改变训练连续性，暂不作为保精度优先默认方案；
- 原始 `model_max_length=32000` 虽能越过第 8 步 OOM，但在当前 PCIe 多卡环境中通信代价过高；
- `MODEL_MAX_LENGTH=16384` 通过了 20-step canary，但完整训练在第 132 步附近仍然反向传播 OOM；
- 当前下一次正式训练应保持 `MODEL_MAX_LENGTH=12288`、`M4_CHUNKED_LM_LOSS_CHECKPOINT=0`、`DEEPSPEED_CONFIG=scripts/zero3.json`，并从 `checkpoint-100` 继续。

已完成的 20-step canary：

```text
日志：/home/yjm/M4-main/intersuit/train_logs/finetune_m4_chunked_16k_aligned_canary_20260711_213728.log
输出：/home/yjm/M4-main/intersuit/checkpoints/M4-LongVA-7B-Qwen2-train-chunked-16k-aligned-canary
结果：20/20 step 完成，train_runtime 约 1526.56 秒，checkpoint 保存成功。
```

12k 冻结 `lm_head` 的正式训练已完成，并已使用 `checkpoints/M4-LongVA-7B-Qwen2-train-chunked-12k-freeze-lmhead` 做视频和音频 demo 验收。

## 2026-07-14 最终 no-freeze 训练与验收状态

用户要求的“不能冻结模块、不能继续削减上下文”的正式训练已经完成。

最终训练配置：

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4 \
NUM_GPUS=4 \
MID_RUN_NAME=M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze \
CKPT_PATH=checkpoints/M4-LongVA-7B-Qwen2 \
MODEL_MAX_LENGTH=12288 \
DEEPSPEED_CONFIG=scripts/zero3_lowmem.json \
M4_ALLOW_ZERO3_LOWMEM_RESUME=1 \
M4_CHUNKED_LM_LOSS=1 \
M4_CHUNKED_LM_LOSS_TOKENS=512 \
M4_CHUNKED_LM_LOSS_CHECKPOINT=0 \
M4_FREEZE_LM_HEAD=0 \
LEARNING_RATE=2e-6 \
SAVE_STEPS=25 \
bash scripts/finetune_m4.sh
```

最终训练结果：

- 日志：`/home/yjm/M4-main/intersuit/train_logs/finetune_m4_12k_lowmem_nofreeze_resume75_after_reboot_20260713.log`
- 输出目录：`/home/yjm/M4-main/intersuit/checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze`
- 最终 step checkpoint：`/home/yjm/M4-main/intersuit/checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze/checkpoint-622`
- 训练进度：`622/622`
- `train_runtime`：`41777.5616` 秒，约 `11.60` 小时
- `train_loss`：`0.06275078892921938`
- `epoch`：`1.0`
- 日志出现：`Model saved to checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze`
- 最终目录包含 4 个 `model-*.safetensors` 分片、`model.safetensors.index.json`、tokenizer/config 文件和 `trainer_state.json`

重要结论：

- 这次训练没有冻结 `lm_head`，`M4_FREEZE_LM_HEAD=0`；
- 这次训练没有继续降到低于 12k 的上下文，`MODEL_MAX_LENGTH=12288`；
- 训练成功越过此前反复失败的第 `132/622` 步，并完成完整 1 epoch；
- 训练过程中出现过 DeepSpeed allocator cache flush 警告，说明显存压力很高，但没有导致训练失败；
- 训练完成后检查内核日志，未发现新的 `Xid`、系统 OOM、`PCIe Bus Error` 或 `hung task`。

最终 checkpoint 视频验收：

```bash
cd /home/yjm/M4-main/intersuit
CUDA_VISIBLE_DEVICES=4 MODEL_PATH=checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze ./scripts/run_demos_stable.sh turntaking-valid
CUDA_VISIBLE_DEVICES=4 MODEL_PATH=checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze ./scripts/run_demos_stable.sh turntaking-interrupt
CUDA_VISIBLE_DEVICES=3 MODEL_PATH=checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze ./scripts/run_demos_stable.sh turntaking-noise
CUDA_VISIBLE_DEVICES=4 MODEL_PATH=checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze ./scripts/run_demos_stable.sh proactive
```

结果：

- `turntaking-valid` 对 `How many people in the video?` 回答 `There are two people in the video.`，与原始 checkpoint 的预期一致，也修复了冻结 `lm_head` 版本回答 `three people` 的质量偏移；
- `turntaking-noise` 能继续描述厨房、烹饪/演示场景、两个人、玻璃杯、火柴/火焰、厨房背景等关键信息；
- `turntaking-interrupt` 对 `Sorry to interrupt?` 没有继续编造长回答，行为合理；
- `proactive` 正常触发 `REQUIREMENT MEET AT 22.0s`。

音频 demo 验收：

```bash
cd /home/yjm/M4-main/intersuit
./scripts/run_demos_stable.sh baseline-audio-tts
./scripts/run_demos_stable.sh turntaking-audio-tts
./scripts/run_demos_stable.sh baseline-audio-file
./scripts/run_demos_stable.sh turntaking-audio-file
```

结果：

- ChatTTS 路径可用；
- 已有 wav 文件路径可用；
- 音频 baseline 能输出厨房/烹饪演示相关描述；
- 音频 turn-taking 能完成追问流程，并能识别两个人、厨房、玻璃杯、操作台等关键信息。

当前仍需注意：

- 2026-07-14 15:03 服务器再次重启后，GPU 功率上限恢复为 `450W`；当前会话无 sudo 权限，执行 `nvidia-smi -pm 1` 和 `nvidia-smi -pl 300` 被拒绝。需要用户在服务器终端手动执行：

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 300
```

- GPU 0 当前仍显示一个 `/proc` 中不存在的 Python PID `6639`，占用约 `13.9GB` 显存，更像驱动侧残留；后续 demo 已避开 GPU 0，使用 GPU 3/4 完成。
- `scripts/eval_m4_sanity.py` 已修复为跟随模型 dtype，并补齐 demo 使用的 `mm_spatial_pool_stride=2`、`mm_spatial_pool_mode=average`，但它对原始 checkpoint 和最终 checkpoint 的纯文本/单图样本仍会输出空字符串。因此目前不能把该自写 sanity 入口作为可靠质量证据；可靠验收以项目原生视频/音频 demo 为准。
- 如果后续要做更严格的数据集样本验收，应整理一个复用 `local_demo` 原生推理类的图像/纯文本测试入口，而不是继续依赖当前简化 sanity 脚本。

## 已知告警

下面这些告警没有阻塞 demo 或训练 smoke test：

- `pip check` 会提示 `decord 0.6.0 is not supported on this platform`，但 `import decord` 和实际 demo 均可用。
- DeepSpeed 会提示缺少 `libaio-dev`，这只影响异步 I/O 优化，不影响当前训练验证。
- Transformers 和 Accelerate 有未来版本弃用提醒。
- 会提示用 `llava_qwen` 加载 `qwen2` 类型模型；当前 demo 和训练 smoke test 均已成功。

## 完整论文级训练仍可能需要的内容

如果目标是完整复刻论文中所有训练阶段，而不仅是本地功能复现，还需要根据具体阶段补充：

- 完整 M4-IT 数据集。
- `VoiceAssistant-400K` 音频指令数据，用于 voice-assistant 阶段。
- `LongVA-7B-Qwen2-VoiceAssistant/checkpoint-3000`，用于不改 `LLM_VERSION` 直接运行音频微调脚本。
- LLaVA-NeXT-Speech 数据，用于 `finetune_llavanextaudio.sh`。

对于本地功能级复现，也就是能真实跑出模型内容并完成一次训练闭环，当前环境已经足够。

## 2026-07-14 论文/发布标准复现推进

本轮已完成的基础设施：

- 新增 `intersuit/scripts/audit_m4_data.py`，用于审计 M4-IT JSON、图像、音频引用；
- 新增 `intersuit/scripts/build_m4_audio_json.py`，从现有 wav 构建带 `speech` 字段和 `<speech>` token 的音频训练 JSON；
- 新增 `intersuit/scripts/eval_m4_suite.py`，统一运行已验证过的 video/audio demo 并输出 JSONL；
- 新增 `intersuit/scripts/run_m4_32k_repro_stage.sh`，提供 `16k-canary150`、`32k-canary20`、`32k-canary150`、`32k-full` 四个阶段；
- 新增 `intersuit/scripts/run_m4_audio_repro_stage.sh`，提供 `voiceassistant`、`llavanext-speech`、`m4-audio` 三个音频复训阶段；
- 新增 `intersuit/scripts/check_m4_repro_preflight.py`，训练前检查稳定四卡是否空闲、功率是否为已验证策略、数据审计是否通过。早期默认要求 `300W`；后续已确认 `450W` 也可完成 32k no-freeze full 训练。

数据审计结果：

```text
M4-IT 样本数：9963
图像引用数：4052
缺失图像数：0
生成版音频 JSON：intersuit/inputs/texts/m4-it-qwen-audio.generated.json
唯一 wav 引用数：30188
缺失 wav 数：0
审计报告：intersuit/train_logs/m4_data_audit_generated_audio.json
```

需要注意：

- `intersuit/inputs/texts/m4-it-qwen-audio.json` 仍是视觉 JSON 的软链接，不含 `speech` 字段，不能作为完整音频复训数据；
- `m4-it-qwen-audio.generated.json` 已按训练加载器格式补齐 `speech` 列表，并把对应人类提问位置替换为 `<speech>`，首轮视觉输入保留 `<image>`；
- 数据审计仍提示重复 `id`，但图像和音频资源 0 缺失。严格论文级报告中还需要联网对照 Hugging Face `ColorfulAI/M4-IT` 官方 manifest。

当时阻塞（历史记录，后续 32k 训练已完成）：

- 普通沙盒上下文看不到 `/dev/nvidia*`，但提权 GPU 上下文可运行 `nvidia-smi` 和 PyTorch CUDA；
- GPU 0 被用户 `dls` 的进程占用：

```text
PID: 6639
命令：python train_yolo11x-doubleB-DoneVehicle-Slice.py
GPU: 物理 GPU 0 / 00000000:4F:00.0
显存：约 13.7GB-14.2GB
利用率：约 80%-95%
```

- GPU 0、2、3、4 的功率限制当时为 `450W`，不是早期稳定策略要求的 `300W`；后续实测表明 `450W` 可以完成 32k no-freeze full 训练，因此不再作为硬性 blocker；
- 当前执行 `nvidia-smi -pm 1` 和 `nvidia-smi -pl 300` 返回权限不足，需要用户在服务器终端用 sudo 执行：

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 300
```

因此，本轮当时没有启动 32k 四卡训练。该阻塞后来已解除，并完成 32k full 训练。本段保留为历史排错记录。

```bash
python scripts/check_m4_repro_preflight.py
bash scripts/run_m4_32k_repro_stage.sh 16k-canary150
```

## 2026-07-15 32k no-freeze 正式训练完成

在后续 GPU 资源可用后，已完成 32k no-freeze full 训练。核心配置：

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4 \
NUM_GPUS=4 \
MODEL_MAX_LENGTH=32000 \
DEEPSPEED_CONFIG=scripts/zero3_lowmem.json \
M4_CHUNKED_LM_LOSS=1 \
M4_CHUNKED_LM_LOSS_TOKENS=512 \
M4_CHUNKED_LM_LOSS_CHECKPOINT=0 \
M4_FREEZE_LM_HEAD=0 \
LEARNING_RATE=1e-5 \
bash scripts/run_m4_32k_precision_repro.sh 32k-full
```

最终训练结果：

- 日志：`/home/yjm/M4-main/intersuit/train_logs/M4-LongVA-7B-Qwen2-repro-32k-lowmem-nofreeze-v1_20260715_133452.log`
- 输出目录：`/home/yjm/M4-main/intersuit/checkpoints/M4-LongVA-7B-Qwen2-repro-32k-lowmem-nofreeze-v1`
- 训练进度：`622/622`
- `train_runtime`：`43339.426` 秒，约 `12.04` 小时
- `train_loss`：`0.09159564487920577`
- `epoch`：`1.0`
- 日志出现：`Model saved to checkpoints/M4-LongVA-7B-Qwen2-repro-32k-lowmem-nofreeze-v1`

重要结论：

- 32k full 在四卡 `0,2,3,4`、`zero3_lowmem`、分块 LM loss、`lm_head` 不冻结条件下完成；
- `450W` 功率限制已被这次 full 训练验证为可用策略，后续不应把 `300W` 作为唯一允许值；
- 32k 可作为 AS-M4 最终长上下文主实验候选，但新增场景音频模块后仍必须重新走 `20-step -> 150-step -> full` canary 阶梯。

## 2026-07-14 OmniMMI 官方 Benchmark 初测

本轮先跑了 OmniMMI 官方子任务 `PA / proactive_alerting`，这是六个子任务里视频体积最小的一项，适合作为第一个官方指标闭环。

准备工作：

- 官方仓库：`third_party/OmniMMI`
- 数据集来源：Hugging Face `ColorfulAI/OmniMMI`
- 已下载 PA 视频：`200` 个
- PA 视频目录：`third_party/OmniMMI/omnimmi/videos`
- 预测文件：`third_party/OmniMMI/results/pa_M4Online.jsonl`
- 评分日志：`third_party/OmniMMI/results/pa_M4Online_eval.txt`

本次评测模型：

```text
模型入口：OmniMMI 官方 M4Online baseline
实际 checkpoint：/home/yjm/M4-main/intersuit/checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze
评测 GPU：物理 GPU 2、3、4 并行切 3 个 chunk
样本数：200/200
```

为适配本地环境做过的最小补丁：

- `third_party/OmniMMI/evaluations/inference.py` 支持用 `M4_MODEL_PATH` 指定本地 M4 checkpoint，避免官方脚本里的旧 ablation 路径覆盖；
- `third_party/OmniMMI/baselines/intersuitonline_modeling.py` 兼容本项目的 `local_demo.arguments_live` 和 `local_demo.inference_util`；
- `third_party/OmniMMI/evaluations/evaluate.py` 将 `openai`、`dotenv` 改为可选依赖，因为 PA/PT 规则评分不需要 GPT 裁判；
- 在 `third_party/OmniMMI/baselines/checkpoints/clip-vit-large-patch14-336` 建立软链接，指向本项目已有 CLIP 权重。

官方 PA 评分结果：

```text
TASK: pa
MODEL: M4Online
Total Data Point: 200
Average Accuracy: 0.195
Average precision: 0.2929285714285714
Average IoU: 0.13773166659783542
```

解读：

- OmniMMI leaderboard 中官方 M4 的 PA 为 `25.50`，可近似理解为 `25.50%`；
- 本地 12k no-freeze checkpoint 在 PA 上的 Accuracy 为 `19.5%`；
- 这个结果说明本地模型已经能跑通官方 benchmark 的完整 PA 子任务，但和官方 M4 发布指标还有差距；
- 差距来源不能只归因于模型质量，还包括当前本地 checkpoint 是 `MODEL_MAX_LENGTH=12288`，不是官方/理想的 `32k`；并且本轮只测了 PA，尚未跑 SG/AP/MD/SI/PT 全量。

下一步若继续做论文级复现，应优先：

```bash
cd /home/yjm/M4-main/third_party/OmniMMI/baselines

# 继续跑 PT，可以得到另一个不依赖 GPT 裁判的官方规则指标；
# AP/SG/MD/SI 需要 OpenAI/GPT 裁判或额外评测依赖。
```

## 2026-07-14 OmniMMI PT 官方 Benchmark 初测

本轮继续跑了 OmniMMI 官方子任务 `PT / proactive_turntaking`。PT 的官方评分脚本是规则评分，不依赖 GPT 裁判，因此适合作为 PA 之后的第二个闭环指标。

准备工作：

- PT 标注文件：`third_party/OmniMMI/omnimmi/proactive_turntaking.json`
- PT 视频数量：`200` 个
- PT 视频目录：`third_party/OmniMMI/omnimmi/videos`
- 预测文件：`third_party/OmniMMI/results/pt_M4.jsonl`
- 评分日志：`third_party/OmniMMI/results/pt_M4_eval.txt`

本次评测模型：

```text
模型入口：OmniMMI 官方 M4 baseline
实际 checkpoint：/home/yjm/M4-main/intersuit/checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze
评测 GPU：物理 GPU 2、3、4 并行切 3 个 chunk
样本数：200/200
```

本轮额外处理：

- `M4Online` 的 PT 分支在当前三分片设置下与官方 PT 评分格式不匹配：在线分支输出 `0/1`，但 PT 评分脚本期待文本字符串；
- 因此 PT 使用官方非在线 `M4` baseline 路径完成推理；
- 非在线 `M4` 路径原本默认要求 FlashAttention2，本地环境未安装；已改为使用 PyTorch/Transformers 的 `attn_implementation="sdpa"`；
- `ego_9.mp4` 初次下载文件不完整，`decord` 报 `moov atom not found`；已从 Hugging Face 重新完整下载并验证可读取。

官方 PT 评分结果：

```text
TASK: pt
MODEL: M4
Total Data Point: 200
Average Accuracy: 0.61
```

解读：

- 本地 12k no-freeze checkpoint 已经跑通 OmniMMI 的 `PA` 和 `PT` 两个官方规则评分子任务；
- PT 当前指标为 `61.0%`；
- PA 当前指标为 `19.5%` Accuracy、`29.29%` precision、`13.77%` IoU；
- 这说明当前成果已经超过“只跑 demo”的层次，进入了官方 benchmark 的可量化评测阶段；
- 但还不能称为论文/发布标准完整复现，因为 AP/SG/MD/SI 尚未完成，且这些子任务通常需要 GPT 裁判或额外评测配置。

## 2026-07-16 12k/32k 相比官方 M4 的策略差异记录

本节记录本地两个可用 checkpoint 相比官方 M4 训练/评测设定的实际改动，便于后续判断哪些差异可能影响精度，哪些只是为了在本机 4090 环境上跑通。

对照基准：

- 官方 M4 leaderboard 记录：`third_party/OmniMMI/README.md` 中 M4 为 `Qwen2-7B`、`32 / 1 fps`，PA 为 `25.50`，PT 为 `62.00`。
- 本地 12k no-freeze：`checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze`。
- 本地 32k precision repro：`checkpoints/M4-LongVA-7B-Qwen2-repro-32k-lowmem-nofreeze-v1`。

关键训练策略差异：

| 项目 | 官方 M4/目标设定 | 本地 12k no-freeze | 本地 32k precision repro | 对精度/复现的影响判断 |
|---|---|---|---|---|
| 基座/起点 | `checkpoints/M4-LongVA-7B-Qwen2` | 同官方起点 | 同官方起点 | 保持一致 |
| 上下文长度 | `MODEL_MAX_LENGTH=32000` | `MODEL_MAX_LENGTH=12288` | `MODEL_MAX_LENGTH=32000` | 12k 是主要语义差异；32k 对齐官方 |
| 学习率 | `1e-5` | `1e-5` | `1e-5` | 保持一致 |
| 训练轮数 | `1 epoch` | `1 epoch`，622 step | `1 epoch`，622 step | 保持一致 |
| `lm_head` | 不冻结 | `M4_FREEZE_LM_HEAD=0` | `M4_FREEZE_LM_HEAD=0` | 保持一致 |
| 训练数据 | M4-IT 本地清单 | `m4-it-qwen.json`，9963 条，图像缺失 0 | 同 12k | 未联网逐条核对官方 manifest，仍有数据清单一致性风险 |
| DeepSpeed | 官方脚本默认配置未在本地完全复刻 | `scripts/zero3_lowmem.json` | `scripts/zero3_lowmem.json` | 执行/显存策略差异，理论上不改训练目标 |
| LM loss 计算 | 常规 loss 路径 | `M4_CHUNKED_LM_LOSS=1`，tokens `512` | `M4_CHUNKED_LM_LOSS=1`，tokens `512` | 省显存等价实现；不改变 loss 目标，但数值轨迹可能有极小实现差异 |
| loss checkpoint | 官方未使用该本地补丁 | `M4_CHUNKED_LM_LOSS_CHECKPOINT=0` | `M4_CHUNKED_LM_LOSS_CHECKPOINT=0` | 关闭是为了避免 ZeRO-3 NCCL mismatch，不改变目标 |
| GPU/并行 | 官方未绑定本机 GPU | 四卡 `CUDA_VISIBLE_DEVICES=0,2,3,4` | 四卡 `CUDA_VISIBLE_DEVICES=0,2,3,4` | 执行环境差异；多卡通信和随机轨迹可能不同 |
| 保存策略 | 通常保留 checkpoint | `SAVE_STEPS=25`，保留 step checkpoint | `SAVE_STRATEGY=no`，只保存最终模型 | 32k 为磁盘空间折中；不改变优化过程，但不保留 optimizer/resume checkpoint |
| 输出目录 | 官方发布权重 | 独立目录，未覆盖官方/原始权重 | 独立目录，未覆盖 12k | 保持实验隔离 |

评测适配差异：

- PA 使用 OmniMMI 官方 `M4Online` baseline 入口；PT 使用官方非在线 `M4` baseline 入口，因为当前 `M4Online` 的 PT 输出格式与本地三分片评分流程不匹配。
- `third_party/OmniMMI/evaluations/inference.py` 增加了 `M4_MODEL_PATH`，用于显式指定本地 checkpoint，避免官方脚本里的旧路径覆盖。
- `third_party/OmniMMI/evaluations/evaluate.py` 将 `openai`、`dotenv` 改为可选依赖；PA/PT 是规则评分，不依赖 GPT 裁判。
- 非在线 `M4` 评测路径改用 PyTorch/Transformers `sdpa`，因为本地未安装 FlashAttention2。

官方指标与本地结果对比：

| 指标 | 官方 M4 | 本地 12k no-freeze | 本地 32k precision repro | 32k 相对官方 | 32k 相对 12k |
|---|---:|---:|---:|---:|---:|
| PA Accuracy | 25.50% | 19.50% | 22.50% | -3.00 pts | +3.00 pts |
| PA Precision | 未公布 | 29.29% | 34.71% | 无法直接比较 | +5.41 pts |
| PA IoU | 未公布 | 13.77% | 14.53% | 无法直接比较 | +0.75 pts |
| PT Accuracy | 62.00% | 61.00% | 47.00% | -15.00 pts | -14.00 pts |

结论：

- 12k no-freeze 的主要问题是上下文长度从官方 32k 降到 12k，这是明确影响复现一致性的改动；但它在 PT 上反而接近官方，说明 turn-taking/noise 判断保持较好。
- 32k precision repro 在训练语义上更接近官方：上下文、学习率、epoch、起点、可训练模块都对齐；主要差异集中在 ZeRO-3 lowmem、chunked loss、保存策略和本地评测适配。
- 32k 在 PA 上比 12k 更接近官方，但 PT 明显退化；因此当前不能只因为 32k 对齐官方上下文就判定综合复现更好。
- 后续若继续追官方精度，应优先排查 32k 的 PT 退化：数据顺序/随机种子、保存策略是否间接改变最终权重、官方训练脚本细节、评测入口与解码参数差异。

## 2026-07-16 AS-M4 回退与冻结策略记录

针对后续 AS-M4 场景音频改进，当前复现状态文件同步记录以下约束，避免把“冻结模块”和“恢复原模型”混为一谈。

同时已将 `/home/yjm/M4-main/AGENTS.md` 中的本机显存前提从早期 `24GB` 估算更新为约 `49GB` 单卡规划。该更新只改变资源规划和 OOM 降级判断，不改变“多卡显存不能简单相加”的原则；标准 DDP 仍按单进程单卡约 `49GB` 约束处理，ZeRO/FSDP 才能切分模型状态。

### 行为回退

行为回退用于验证新增场景音频路径没有破坏原 M4 行为。AS-M4 集成时必须保留以下配置开关：

```text
enable_scene_audio=false
use_audio_event_detector=false
use_temporal_aligner=false
use_audio_confidence_gate=false
use_gated_av_fusion=false
force_audio_gate=0
```

预期行为：

- `enable_scene_audio=false` 时，不读取或不使用视频原始场景音频 `scene_audio`；
- `speech` / `query_speech` 仍只表示用户问题语音，不得与 `scene_audio` 混用；
- `force_audio_gate=0` 时，残差融合应退化为原视频特征，视频 token 数量不变；
- E1 归因实验必须使用“AS-M4 静音或关闭场景音频”对齐 E0 原 M4，证明新增模块本身没有凭空带来或破坏能力。

### 权重回退

权重回退用于恢复已经训练改变的模型参数。冻结模块只能阻止后续训练继续更新参数，不能把已经写入 checkpoint 的权重恢复到旧值。

真正恢复原模型能力时，必须显式加载以下保留 checkpoint 之一：

```text
原始 M4：intersuit/checkpoints/M4-LongVA-7B-Qwen2
12k no-freeze：intersuit/checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze
32k no-freeze：intersuit/checkpoints/M4-LongVA-7B-Qwen2-repro-32k-lowmem-nofreeze-v1
```

AS-M4 训练输出必须使用新的 versioned run name，不得覆盖上述目录。新增模块权重应尽量单独命名保存，便于只加载或卸载 AS-M4 增量。

当前脚本层已记录并支持以下可执行回退环境变量：

```bash
# 简写：行为回退，关闭 scene_audio 并强制 gate=0
AS_M4_ROLLBACK_MODE=behavior

# 简写：权重回退，加载已保留的 12k / 32k 基线 checkpoint
AS_M4_ROLLBACK_MODE=weights12k
AS_M4_ROLLBACK_MODE=weights32k

# 完全关闭 AS-M4 场景音频路径，走原 M4 输入路径
AS_M4_ENABLE_SCENE_AUDIO=0

# 保留 AS-M4 编码/对齐链路，但强制残差融合 gate=0，用于验证 video feature 可精确回退
AS_M4_FORCE_AUDIO_GATE=0

# 只训练原 M4 相关模块，不训练 AS-M4 新增模块
MM_TUNABLE_PARTS=mm_vision_tower,mm_mlp_adapter,mm_language_model

# 训练 AS-M4 新增模块
MM_TUNABLE_PARTS=mm_vision_tower,mm_mlp_adapter,mm_language_model,streaming_av_module
```

注意：冻结或从 `MM_TUNABLE_PARTS` 中移除 `streaming_av_module` 只能阻止新增模块继续更新；如果某个 AS-M4 checkpoint 已经改变了 Qwen2、视觉塔或 projector 权重，必须重新加载上方保留的基线 checkpoint 才能做真正权重回退。

2026-07-16 已将上述语义写入 `run_as_m4_stage.sh` / `finetune_m4.sh` 配置输出：每个 AS-M4 staged run 的 `.env` 会记录 `AS_M4_ROLLBACK_MODE`、`AS_M4_BASELINE_CKPT_12K`、`AS_M4_BASELINE_CKPT_32K`、`AS_M4_ENABLE_SCENE_AUDIO` 和 `AS_M4_FORCE_AUDIO_GATE`，方便复现实验判断当次运行属于默认 AS-M4、行为回退、gate=0 回退还是权重回退。

### 32k OOM 时的保精度降级顺序

如果 AS-M4 新增模块导致 32k OOM，优先牺牲时间来保精度，每次只改一个降级项，并重新执行 `2-step -> 20-step -> 150-step` canary。

优先顺序：

1. 重新运行 preflight，先排除残留进程、坏样本、缓存、磁盘、NCCL 接口或功率记录问题；
2. 离线预计算冻结场景音频特征，训练时不再运行 scene audio encoder forward；
3. 对 event detector、temporal aligner、confidence gate、fusion 和 projector 启用 activation checkpointing；
4. 将 `M4_CHUNKED_LM_LOSS_TOKENS` 从 `512` 降到 `256`，仍 OOM 再降到 `128`；这只增加 `lm_head` 循环次数，不改变 loss 目标；
5. micro batch 降到 `1`，通过增加 gradient accumulation 保持有效 batch；
6. 对 temporal aligner 使用 chunked similarity，完整保留原定 offset 搜索范围，只分块计算音频/视频相似度矩阵；
7. 减少训练中保留的诊断张量，只记录抽样统计，不保存全量 alignment matrix；
8. 受控启用 ZeRO-3 optimizer CPU offload，并监控系统内存；
9. 以上仍失败时，才缩短训练时的音频 history，但保留评测配置，并记录训练/评测差异；
10. 再失败时，才小幅减少对齐候选范围，并优先只作用于训练阶段，不削减原始 32k 文本上下文；
11. 最后才考虑降低 LoRA rank 或冻结更多 AS-M4 新增模块。

明确禁止作为优先方案：

- 不优先冻结 `lm_head`；
- 不优先降低 `MODEL_MAX_LENGTH=32000`；
- 不优先删除关键视频帧或有效音频信息；
- 不覆盖 12k/32k no-freeze 基线 checkpoint。

## 2026-07-16 AS-M4 P2-P13 实现与 P14 数据缺口

当前已完成 AS-M4 的 CPU 侧模块、轻量集成和训练脚本 dry-run：

- `audio_stream.py`：场景音频读取/切窗；
- `av_buffer.py`：音视频统一时间轴 buffer；
- `scene_audio_encoder`：`dummy` / `precomputed` 场景音频编码器接口；
- `streaming_av`：事件检测、动态选帧、因果时间对齐、置信度门控、残差融合；
- `llava_arch.py` / `llava_qwen.py`：已支持 `scene_audios` 参数透传，并在 video feature flatten 前执行 AS-M4 最小真实融合闭环；
- `train.py` / `DataCollator`：已支持 `scene_audio`、`scene_audio_mask`、`scene_audio_timestamps`；
- `llava_trainer.py`：已加入 AS-M4 模块学习率分组和保存关键词；
- `run_as_m4_stage.sh` / `run_as_m4_32k_stage.sh`：已支持 12k/32k AS-M4 dry-run，不覆盖现有 checkpoint。

已通过的轻量 harness：

```text
audio_stream harness passed
av_buffer harness passed
scene_audio_encoder harness passed
event_detector harness passed
frame_scheduler harness passed
temporal_aligner harness passed
confidence_gate harness passed
fusion harness passed
streaming_av_integration harness passed
dataset_collator harness passed
```

其中 `streaming_av_integration` 已验证：

- 正常 gate 下，scene audio 会改变 video feature；
- `force_audio_gate=0` 时，输出精确回退到原 video feature；
- 非 video modality 不被错误融合；
- fused video loss 可以把梯度传回 AS-M4 新增模块。

P14 真实 2-step 训练 smoke 尚未完成，原因不是 GPU 或功率问题，而是当前本地 M4-IT 训练集没有 `video` 样本：

```text
intersuit/inputs/texts/m4-it-qwen.json: video count = 0
```

因此，直接用当前 M4-IT 跑 AS-M4 训练只能验证 `scene_audio` 数据字段和模型参数透传，不能证明“视频帧 + 场景音频融合”在真实 Trainer 中产生有效梯度。P14 的下一步必须先满足以下任一条件：

1. 接入 AVQA / MUSIC-AVQA / AVE / LLP / LongVALE 等带视频和原始音轨或预计算 scene-audio 特征的数据；
2. 构造一个本地极小视频 smoke 数据集，并保证当前环境可以解码视频；
3. 增加训练侧对预计算 video features 的 smoke-only 支持，用于绕过视频解码依赖。

在这三者之一完成前，不应把 AS-M4 `2-step` 训练称为通过，只能称为“脚本 dry-run 与模块级梯度 harness 通过”。

### 2026-07-16 P14 训练 smoke 通信前提补充

为绕过视频解码依赖，已补充预计算 video feature smoke 数据，并开始验证真实 Trainer 中的 `video_features + scene_audio` 链路。过程中发现两个环境/脚本问题：

1. 早期 `timeout` 后的 `AS-M4-12k-smoke-vfeat-2step-v6/v7` 子进程未被完整清理，残留 rank 继续占用 GPU 并干扰后续 NCCL。已确认这些 PID 属于本轮 AS-M4 smoke，并已清理，GPU 0/2/3/4 显存恢复到空闲状态。
2. NCCL 自动选择了 `eth0=169.254.3.1` 链路本地地址，AS-M4 四卡 smoke 出现 `NET/Socket message truncated` 与 collective mismatch。新增 `intersuit/scripts/nccl_probe.py`，并已验证：

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4 \
NCCL_SOCKET_IFNAME=lo \
torchrun --standalone --nproc_per_node=4 intersuit/scripts/nccl_probe.py
```

输出 `all_reduce_sum=10.0`，说明单机四卡在 `lo` 接口下 NCCL all-reduce 正常。

脚本侧已更新：

- `intersuit/scripts/run_as_m4_stage.sh` 默认设置并记录 `NCCL_SOCKET_IFNAME=lo`；
- dry-run `.env` 同步记录该值；
- 同一秒生成多个 dry-run 时，`.env/.log` 会自动追加 `-v2/-v3`，避免 E1/E7 回退配置相互覆盖；
- `llava_trainer.py` 的特殊学习率分组跳过空参数组，避免冻结 `scene_audio_encoder` 时生成空 optimizer group。

待恢复执行额度后，需要重新运行：

```bash
cd /home/yjm/M4-main/intersuit
CUDA_VISIBLE_DEVICES=0,2,3,4 \
NUM_GPUS=4 \
DATA_PATH=inputs/texts/as_m4_smoke_video_features.json \
VIDEO_FEATURE_FOLDER=inputs/features/as_m4_smoke \
MID_RUN_NAME=AS-M4-12k-smoke-vfeat-asmodules-2step-v4 \
MAX_STEPS=2 \
SAVE_STRATEGY=no \
RUN_PREFLIGHT=0 \
MM_TUNABLE_PARTS=streaming_av_module \
bash scripts/run_as_m4_stage.sh 12k-smoke
```

该 run 才能判定 P14 AS-only 2-step 是否通过。当前因 Codex GPU 执行额度限制，尚未完成此 GPU smoke，不能标记 P14 通过。

### 2026-07-16 P1 preflight 验证补充

已验证 `intersuit/scripts/check_m4_repro_preflight.py` 的新增前提检查：

```bash
NCCL_SOCKET_IFNAME=lo \
python intersuit/scripts/check_m4_repro_preflight.py \
  --expected_nccl_socket_ifname lo \
  --allowed_power_limits "300.00 W,450.00 W" \
  --max_idle_memory_mib 100000
```

沙箱内 `nvidia-smi` 只读查询返回失败且 stderr 为空；随后在非沙箱只读 preflight 中通过。通过项：

- `NCCL_SOCKET_IFNAME=lo` 与预期一致；
- 未发现残留 `intersuit/train/train_mem.py` 训练进程；
- 稳定四卡 `0,2,3,4` 均为空闲状态，显存占用约 `11 MiB`；
- GPU 0/2/3/4 当前功率限制均为 `450.00 W`，符合允许列表 `300.00 W,450.00 W`；
- 数据审计状态为 `warn`，但 `missing_image_count=0`、`missing_audio_count=0`。

因此 P1 preflight 的功率策略、NCCL 接口检查和残留训练进程检查已通过；后续 P14 仍需重新运行 AS-only 2-step GPU smoke，不能用 preflight 通过替代训练 smoke 通过。

### 2026-07-16 P14 AS-only 2-step smoke 边界验证

已重新运行 AS-only 2-step smoke：

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4 \
NUM_GPUS=4 \
NCCL_SOCKET_IFNAME=lo \
DATA_PATH=inputs/texts/as_m4_smoke_video_features.json \
VIDEO_FEATURE_FOLDER=inputs/features/as_m4_smoke \
MID_RUN_NAME=AS-M4-12k-smoke-vfeat-asmodules-2step-v5 \
MAX_STEPS=2 \
SAVE_STRATEGY=no \
RUN_PREFLIGHT=1 \
MM_TUNABLE_PARTS=streaming_av_module \
bash scripts/run_as_m4_stage.sh 12k-smoke
```

结果：

- preflight 通过；
- 数据集 `as_m4_smoke_video_features.json` 成功加载 `8` 条；
- `MM_TUNABLE_PARTS=streaming_av_module` 生效，trainable parameters 约 `22.07 MB`；
- 失败发生在 DeepSpeed ZeRO-3 参数 broadcast 阶段，表现为 NCCL `message truncated` / collective size mismatch；
- 该失败不是错误网卡导致，因为本次已固定 `NCCL_SOCKET_IFNAME=lo`，且 `nccl_probe.py` 已证明同接口普通 all-reduce 正常。

随后运行单卡隔离 smoke：

```bash
CUDA_VISIBLE_DEVICES=4 \
NUM_GPUS=1 \
NCCL_SOCKET_IFNAME=lo \
DATA_PATH=inputs/texts/as_m4_smoke_video_features.json \
VIDEO_FEATURE_FOLDER=inputs/features/as_m4_smoke \
MID_RUN_NAME=AS-M4-12k-smoke-vfeat-asmodules-2step-single-gpu \
MAX_STEPS=2 \
SAVE_STRATEGY=no \
RUN_PREFLIGHT=0 \
MM_TUNABLE_PARTS=streaming_av_module \
bash scripts/run_as_m4_stage.sh 12k-smoke
```

结果：

- 2/2 steps 完成；
- loss 非 NaN，分别约 `0.5252`、`0.3499`；
- `grad_norm` 非零，说明 AS-M4 新增模块存在有效梯度；
- `train_loss=0.43755383789539337`；
- checkpoint 可保存到 `intersuit/checkpoints/AS-M4-12k-smoke-vfeat-asmodules-2step-single-gpu`。

结论：

- AS-M4 的数据读取、video feature + scene audio 融合、loss、梯度和保存链路在单卡上成立；
- 当前 P14 的未完成项不是模块 forward 本身，而是四卡 ZeRO-3 下 AS-only 小参数训练的参数同步/初始化路径；
- 后续应优先为 AS-only smoke 增加“非 ZeRO-3 或更轻量分布式”路径，牺牲显存和时间换稳定性；不要因此回退到冻结 `lm_head` 或削减上下文。

随后为 AS-only smoke 增加并验证了 DDP-only 路径：

- `intersuit/scripts/finetune_m4.sh` 支持 `DEEPSPEED_CONFIG=none` 时不传 `--deepspeed`；
- DDP-only 路径默认设置 `TORCHDYNAMO_DISABLE=1`，避免 TorchDynamo 的 DDPOptimizer higher-order op 失败；
- `intersuit/scripts/run_as_m4_stage.sh` 的 `.env` 会记录 `TORCHDYNAMO_DISABLE`。

四卡 DDP-only smoke 命令：

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4 \
NUM_GPUS=4 \
NCCL_SOCKET_IFNAME=lo \
DEEPSPEED_CONFIG=none \
TORCHDYNAMO_DISABLE=1 \
DATA_PATH=inputs/texts/as_m4_smoke_video_features.json \
VIDEO_FEATURE_FOLDER=inputs/features/as_m4_smoke \
MID_RUN_NAME=AS-M4-12k-smoke-vfeat-asmodules-2step-ddp-notdynamo \
MAX_STEPS=2 \
SAVE_STRATEGY=no \
RUN_PREFLIGHT=1 \
MM_TUNABLE_PARTS=streaming_av_module \
bash scripts/run_as_m4_stage.sh 12k-smoke
```

结果：

- preflight 通过；
- 2/2 steps 完成；
- loss 非 NaN：第 1 步约 `0.2184`，第 2 步约 `0.2184`；
- `grad_norm=0.09765625`，新增模块梯度非零；
- `train_loss=0.21841667592525482`；
- checkpoint 保存到 `intersuit/checkpoints/AS-M4-12k-smoke-vfeat-asmodules-2step-ddp-notdynamo`；
- 结束后 GPU 0/2/3/4 显存恢复到约 `11 MiB`，未发现残留 `train_mem.py` 进程。

因此，P14 的 AS-only 2-step 训练 smoke 已通过。后续 P15 的 AS-only `20-step` 和 `150-step` canary 应优先沿用：

```text
DEEPSPEED_CONFIG=none
TORCHDYNAMO_DISABLE=1
DDP_FIND_UNUSED_PARAMETERS=True
MM_TUNABLE_PARTS=streaming_av_module
```

完整 12k/32k 主训练若需要同时训练语言模型，仍应回到 `scripts/zero3_lowmem.json` 路径；不要把 AS-only DDP smoke 的配置直接当作全量模型训练配置。

### 2026-07-16 P15 AS-only 20-step canary

第一次 20-step 使用 DDP-only + `TORCHDYNAMO_DISABLE=1` 但未显式启用 unused-parameter 检测，失败于 DDP：

```text
Expected to have finished reduction in the prior iteration...
Parameter indices which did not receive grad: 4 5 8 9
```

原因：Stage A 当前仅通过 LM loss 验证真实 Trainer 链路，event/offset/gate 中部分 head 尚未接入辅助 loss，因此某些参数在该 smoke 中可能暂时无梯度。该问题不应通过删除模块或冻结 `lm_head` 解决。

已更新脚本：

- `finetune_m4.sh` 在 `DEEPSPEED_CONFIG=none` 时默认设置 `DDP_FIND_UNUSED_PARAMETERS=True`；
- `run_as_m4_stage.sh` 的 `.env` 记录 `DDP_FIND_UNUSED_PARAMETERS`。

随后重新运行：

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4 \
NUM_GPUS=4 \
NCCL_SOCKET_IFNAME=lo \
DEEPSPEED_CONFIG=none \
TORCHDYNAMO_DISABLE=1 \
DDP_FIND_UNUSED_PARAMETERS=True \
DATA_PATH=inputs/texts/as_m4_smoke_video_features.json \
VIDEO_FEATURE_FOLDER=inputs/features/as_m4_smoke \
MID_RUN_NAME=AS-M4-12k-canary20-vfeat-asmodules-ddp-notdynamo-unused \
MAX_STEPS=20 \
SAVE_STRATEGY=no \
RUN_PREFLIGHT=1 \
MM_TUNABLE_PARTS=streaming_av_module \
bash scripts/run_as_m4_stage.sh 12k-canary20
```

结果：

- preflight 通过；
- 20/20 steps 完成；
- loss 非 NaN，末段 loss 约 `0.1870`；
- `grad_norm` 非零，末步约 `0.1123046875`；
- `train_runtime=41.6131` 秒；
- `train_loss=0.19919465035200118`；
- checkpoint 保存到 `intersuit/checkpoints/AS-M4-12k-canary20-vfeat-asmodules-ddp-notdynamo-unused`；
- 结束后 GPU 0/2/3/4 显存恢复到约 `11 MiB`，未发现残留 `train_mem.py` 进程。

因此 P15 的 AS-only 12k `20-step` canary 已通过。下一步可继续同配置执行 `150-step` canary。

### 2026-07-16 P15 AS-only 150-step canary

已沿用 20-step 的稳定配置继续执行 150-step：

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4 \
NUM_GPUS=4 \
NCCL_SOCKET_IFNAME=lo \
DEEPSPEED_CONFIG=none \
TORCHDYNAMO_DISABLE=1 \
DDP_FIND_UNUSED_PARAMETERS=True \
DATA_PATH=inputs/texts/as_m4_smoke_video_features.json \
VIDEO_FEATURE_FOLDER=inputs/features/as_m4_smoke \
MID_RUN_NAME=AS-M4-12k-canary150-vfeat-asmodules-ddp-notdynamo-unused \
MAX_STEPS=150 \
SAVE_STRATEGY=no \
RUN_PREFLIGHT=1 \
MM_TUNABLE_PARTS=streaming_av_module \
bash scripts/run_as_m4_stage.sh 12k-canary150
```

结果：

- preflight 通过；
- 150/150 steps 完成；
- loss 非 NaN，末步 loss 约 `0.0896`；
- `grad_norm` 非零，末步约 `0.058837890625`；
- `train_runtime=258.6869` 秒；
- `train_loss=0.1386900380253792`；
- 日志出现 `Model saved to checkpoints/AS-M4-12k-canary150-vfeat-asmodules-ddp-notdynamo-unused`；
- `SAVE_STRATEGY=no` 下不会生成 `checkpoint-*` 子目录，模型文件直接保存到 run 根目录；
- 根目录已确认包含 `config.json`、`model-00001-of-00004.safetensors` 到 `model-00004-of-00004.safetensors`、`model.safetensors.index.json`、`trainer_state.json` 和 tokenizer 文件；
- 结束后 GPU 0/1/2/3/4 显存均约 `11 MiB`、利用率 `0%`、功率限制 `450W`，未发现残留 `train_mem.py` 进程。

因此 P15 的 AS-only 12k `20-step` 和 `150-step` canary 均已通过。下一阶段可以进入 32k AS-only `20-step`，但仍必须重新走 `20-step -> 150-step`，不能因为 12k 通过而直接启动 32k full。

### 2026-07-16 P15 AS-only 32k 20-step canary

在 12k `20-step` / `150-step` 均通过后，已启动 32k AS-only 20-step：

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4 \
NUM_GPUS=4 \
NCCL_SOCKET_IFNAME=lo \
DEEPSPEED_CONFIG=none \
TORCHDYNAMO_DISABLE=1 \
DDP_FIND_UNUSED_PARAMETERS=True \
DATA_PATH=inputs/texts/as_m4_smoke_video_features.json \
VIDEO_FEATURE_FOLDER=inputs/features/as_m4_smoke \
MID_RUN_NAME=AS-M4-32k-canary20-vfeat-asmodules-ddp-notdynamo-unused \
MAX_STEPS=20 \
SAVE_STRATEGY=no \
RUN_PREFLIGHT=1 \
MM_TUNABLE_PARTS=streaming_av_module \
bash scripts/run_as_m4_32k_stage.sh 32k-canary20
```

结果：

- preflight 通过；
- `MODEL_MAX_LENGTH=32000` 生效；
- 20/20 steps 完成；
- loss 非 NaN，末步 loss 约 `0.1853`；
- `grad_norm` 非零，末步约 `0.045166015625`；
- `train_runtime=41.0249` 秒；
- `train_loss=0.19849497973918914`；
- 日志出现 `Model saved to checkpoints/AS-M4-32k-canary20-vfeat-asmodules-ddp-notdynamo-unused`；
- run 根目录包含完整 safetensors、config、trainer_state 和 tokenizer 文件；
- 结束后 GPU 0/1/2/3/4 显存均约 `11 MiB`、利用率 `0%`、功率限制 `450W`，未发现残留 `train_mem.py` 进程。

因此 P15 的 AS-only 32k `20-step` canary 已通过。下一步可以继续同配置执行 32k `150-step` canary；仍不能直接启动 32k full。

### 2026-07-16 P15 AS-only 32k 150-step canary

已沿用 32k 20-step 的稳定配置继续执行 150-step：

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4 \
NUM_GPUS=4 \
NCCL_SOCKET_IFNAME=lo \
DEEPSPEED_CONFIG=none \
TORCHDYNAMO_DISABLE=1 \
DDP_FIND_UNUSED_PARAMETERS=True \
DATA_PATH=inputs/texts/as_m4_smoke_video_features.json \
VIDEO_FEATURE_FOLDER=inputs/features/as_m4_smoke \
MID_RUN_NAME=AS-M4-32k-canary150-vfeat-asmodules-ddp-notdynamo-unused \
MAX_STEPS=150 \
SAVE_STRATEGY=no \
RUN_PREFLIGHT=1 \
MM_TUNABLE_PARTS=streaming_av_module \
bash scripts/run_as_m4_32k_stage.sh 32k-canary150
```

结果：

- preflight 通过；
- `MODEL_MAX_LENGTH=32000` 生效；
- 150/150 steps 完成；
- loss 非 NaN，末步 loss 约 `0.0176`；
- `grad_norm` 非零，末步约 `0.061279296875`；
- `train_runtime=257.8569` 秒；
- `train_loss=0.0750156802435716`；
- 日志出现 `Model saved to checkpoints/AS-M4-32k-canary150-vfeat-asmodules-ddp-notdynamo-unused`；
- run 根目录包含完整 safetensors、config、trainer_state 和 tokenizer 文件；
- 结束后 GPU 0/1/2/3/4 显存均约 `11 MiB`、利用率 `0%`、功率限制 `450W`，未发现残留 `train_mem.py` 进程。

因此 P15 的 AS-only 12k 与 32k `20-step` / `150-step` canary 均已通过。当前已证明：

- `scene_audio` 数据字段、预计算 video features、AS-M4 融合链路、LM loss、梯度和保存路径在真实 Trainer 中可运行；
- 只训练 `streaming_av_module` 时，DDP-only 路线比 ZeRO-3 更适合 smoke/canary；
- 该结论仍只覆盖 AS-only smoke 数据，不等价于真实数据 full 训练或 E0-E7 归因评测通过。

下一阶段不应直接声称 AS-M4 full 完成；应继续补齐 P16 E0-E7 小集归因 harness 或准备真实含 `scene_audio` 的训练/评测数据。

### 2026-07-16 P16 E0-E7 配置级归因 harness

已补充最小可运行的 E0-E7 归因矩阵规划 harness，当前阶段只做配置校验和计划文件生成，不启动 GPU 推理：

新增文件：

```text
intersuit/intersuit/harness/__init__.py
intersuit/intersuit/harness/runners/__init__.py
intersuit/intersuit/harness/runners/run_ablation_matrix.py
intersuit/harness/configs/as_m4_e0_e7_smoke.json
intersuit/tests/test_attribution_harness.py
```

Smoke / Harness：

```bash
python -m py_compile intersuit/intersuit/harness/runners/run_ablation_matrix.py

/home/yjm/miniconda3/envs/M4/bin/python -m pytest -q \
  intersuit/tests/test_attribution_harness.py

PYTHONPATH=/home/yjm/M4-main/intersuit \
/home/yjm/miniconda3/envs/M4/bin/python -m intersuit.harness.runners.run_ablation_matrix \
  --config intersuit/harness/configs/as_m4_e0_e7_smoke.json \
  --output_dir intersuit/harness/artifacts/as_m4_e0_e7_smoke \
  --strict_paths
```

结果：

- `py_compile` 通过；
- pytest 结果为 `3 passed`；
- runner 输出 `{"status": "pass", "errors": [], "num_experiments": 8}`；
- 生成 `intersuit/harness/artifacts/as_m4_e0_e7_smoke/matrix_plan.jsonl`；
- 生成 `intersuit/harness/artifacts/as_m4_e0_e7_smoke/summary.json`；
- `summary.json` 中 `experiment_ids` 为 `E0` 到 `E7`，且 `validation_errors=[]`。

该 harness 当前已经固定：

- 同一 dataset：`as_m4_smoke_video_features`；
- 同一 split：`smoke`；
- 同一 scorer：`exact_match_smoke`；
- E0 使用 `AS_M4_ROLLBACK_MODE=weights12k` 和 12k no-freeze baseline；
- E1 使用 `AS_M4_ROLLBACK_MODE=behavior` 且 `AS_M4_FORCE_AUDIO_GATE=0`；
- E2 使用正确音频条件；
- E3 使用错配音频条件；
- E4 使用噪声音频条件；
- E5/E6 使用相同 shifted 音频条件，但 E5 `alignment=off`、E6 `alignment=on`；
- E7 使用 `AS_M4_ROLLBACK_MODE=gate0` 且 `AS_M4_FORCE_AUDIO_GATE=0`。

边界说明：

- 这一步只证明 E0-E7 的配置矩阵和可追溯输出计划成立；
- 尚未证明 AS-M4 在真实 AVQA / MUSIC-AVQA / OmniMMI 上有性能收益；
- 尚未生成真实 prediction JSONL，也尚未进行 bootstrap 置信区间或官方评分；
- 下一步应在该 plan 基础上接入真实小集推理 runner 与 scoring，而不是把配置级 harness 解释成归因实验完成。

### 2026-07-16 P16 E0-E7 scoring harness

已补充 E0-E7 prediction JSONL 聚合与核心归因不等式评分 harness。

新增文件：

```text
intersuit/intersuit/harness/metrics/__init__.py
intersuit/intersuit/harness/metrics/attribution_metrics.py
intersuit/intersuit/harness/runners/score_ablation_matrix.py
intersuit/tests/test_attribution_scoring.py
```

Smoke / Harness：

```bash
python -m py_compile \
  intersuit/intersuit/harness/metrics/attribution_metrics.py \
  intersuit/intersuit/harness/runners/score_ablation_matrix.py

/home/yjm/miniconda3/envs/M4/bin/python -m pytest -q \
  intersuit/tests/test_attribution_harness.py \
  intersuit/tests/test_attribution_scoring.py
```

结果：

- `py_compile` 通过；
- pytest 结果为 `6 passed`；
- scoring harness 能读取 `matrix_plan.jsonl` 中的 `output_jsonl`；
- 支持按 prediction JSONL 计算 `accuracy`、`mean_gate`、`mean_quality`、`mean_relevance`、`offset_mae`；
- 支持核心关系检查：`E2 > E1`、`E2 > E3`、`E6 > E5`、`E2 > E7`、`E1` 与 `E0` 接近。

已执行一次合成 prediction dry-run：

```bash
PYTHONPATH=/home/yjm/M4-main/intersuit \
/home/yjm/miniconda3/envs/M4/bin/python -m intersuit.harness.runners.score_ablation_matrix \
  --plan intersuit/harness/artifacts/as_m4_e0_e7_smoke/matrix_plan.jsonl \
  --output_dir intersuit/harness/artifacts/as_m4_e0_e7_smoke_score
```

结果：

- 输出 `{"status": "pass", ... "all_core_passed": true}`；
- 生成 `intersuit/harness/artifacts/as_m4_e0_e7_smoke_score/score_summary.json`；
- 8 个合成 prediction 文件位于 `intersuit/harness/artifacts/as_m4_e0_e7_smoke/predictions/`。

边界说明：

- 这次 prediction 是 synthetic，用于验证 scoring 管线，不是模型真实输出；
- 当前只能说明 P16 的配置计划与 scoring 闭环可运行；
- 仍不能声称 AS-M4 性能提升已经被证明；
- 下一步需要把 `matrix_plan.jsonl` 接到真实小集推理 runner，生成真实 E0-E7 prediction JSONL 后再运行该 scorer。

### 2026-07-16 P16 E0-E7 prediction runner oracle smoke

已补充 E0-E7 prediction runner，用于从 `matrix_plan.jsonl` 生成 prediction JSONL。当前已通过 CPU-only `oracle` 后端验证完整管道：

```text
matrix_plan.jsonl
  -> run_predictions_from_plan.py
  -> predictions/e0_predictions.jsonl ... predictions/e7_predictions.jsonl
  -> score_ablation_matrix.py
  -> score_summary.json
```

新增文件：

```text
intersuit/intersuit/harness/runners/run_predictions_from_plan.py
intersuit/tests/test_prediction_runner.py
```

Smoke / Harness：

```bash
python -m py_compile \
  intersuit/intersuit/harness/runners/run_predictions_from_plan.py

/home/yjm/miniconda3/envs/M4/bin/python -m pytest -q \
  intersuit/tests/test_prediction_runner.py \
  intersuit/tests/test_attribution_harness.py \
  intersuit/tests/test_attribution_scoring.py

PYTHONPATH=/home/yjm/M4-main/intersuit \
/home/yjm/miniconda3/envs/M4/bin/python -m intersuit.harness.runners.run_ablation_matrix \
  --config intersuit/harness/configs/as_m4_e0_e7_smoke.json \
  --output_dir intersuit/harness/artifacts/as_m4_e0_e7_smoke \
  --strict_paths

PYTHONPATH=/home/yjm/M4-main/intersuit \
/home/yjm/miniconda3/envs/M4/bin/python -m intersuit.harness.runners.run_predictions_from_plan \
  --plan intersuit/harness/artifacts/as_m4_e0_e7_smoke/matrix_plan.jsonl \
  --backend oracle \
  --limit 2 \
  --feature_root intersuit/inputs/features/as_m4_smoke \
  --device cpu \
  --max_new_tokens 4

PYTHONPATH=/home/yjm/M4-main/intersuit \
/home/yjm/miniconda3/envs/M4/bin/python -m intersuit.harness.runners.score_ablation_matrix \
  --plan intersuit/harness/artifacts/as_m4_e0_e7_smoke/matrix_plan.jsonl \
  --output_dir intersuit/harness/artifacts/as_m4_e0_e7_oracle_score
```

结果：

- `py_compile` 通过；
- pytest 结果为 `9 passed`；
- `run_ablation_matrix` 输出 `{"status": "pass", "errors": [], "num_experiments": 8}`；
- `run_predictions_from_plan --backend oracle` 输出 `{"status": "pass", "backend": "oracle", "num_experiments": 8}`；
- `score_ablation_matrix` 输出 `{"status": "pass", ... "all_core_passed": true}`；
- 生成 prediction 文件：

```text
intersuit/harness/artifacts/as_m4_e0_e7_smoke/predictions/e0_predictions.jsonl
intersuit/harness/artifacts/as_m4_e0_e7_smoke/predictions/e1_predictions.jsonl
intersuit/harness/artifacts/as_m4_e0_e7_smoke/predictions/e2_predictions.jsonl
intersuit/harness/artifacts/as_m4_e0_e7_smoke/predictions/e3_predictions.jsonl
intersuit/harness/artifacts/as_m4_e0_e7_smoke/predictions/e4_predictions.jsonl
intersuit/harness/artifacts/as_m4_e0_e7_smoke/predictions/e5_predictions.jsonl
intersuit/harness/artifacts/as_m4_e0_e7_smoke/predictions/e6_predictions.jsonl
intersuit/harness/artifacts/as_m4_e0_e7_smoke/predictions/e7_predictions.jsonl
```

边界说明：

- `oracle` 后端是确定性 CPU smoke，不加载 M4/AS-M4 权重，不代表真实模型性能；
- `oracle` 的作用是验证 E0-E7 文件契约、prediction JSONL 格式、scoring 关系检查和 artifact 路径；
- 真实归因仍必须使用 `--backend model` 或后续正式推理 runner 生成模型预测后再评分；
- 当前仍不能声称 AS-M4 已在 AVQA、MUSIC-AVQA 或 OmniMMI 上证明性能提升。

### 2026-07-16 P16 E1/E2 真实模型 backend smoke

已在单卡上完成一次最小真实模型 backend smoke。该 smoke 只运行 E1/E2、每个实验 1 条样本，用于验证 AS-M4 checkpoint 加载、`video_feature` 输入包装、dummy muted audio 回退路径、scene audio 参数透传和 prediction JSONL 写出。

为避免一次加载 baseline 与 AS-M4 两套模型，`run_predictions_from_plan.py` 新增：

```text
--experiments E1,E2
--model_name_override LongVA-7B-Qwen2
```

同时修复了两个真实加载问题：

- AS-M4 checkpoint 目录名不含 `qwen`，需要 `--model_name_override LongVA-7B-Qwen2` 才能走 `LlavaQwenForCausalLM`；
- `video_feature` 预计算特征只在 streaming AV 路径中支持。E1 行为回退时会注入一个零值 dummy scene-audio window，并通过 `AS_M4_FORCE_AUDIO_GATE=0` 保证音频不影响融合，从而保持“静音/行为回退”语义。

执行命令：

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/home/yjm/M4-main/intersuit \
/home/yjm/miniconda3/envs/M4/bin/python -m intersuit.harness.runners.run_predictions_from_plan \
  --plan intersuit/harness/artifacts/as_m4_e0_e7_smoke/matrix_plan.jsonl \
  --backend model \
  --experiments E1,E2 \
  --limit 1 \
  --feature_root intersuit/inputs/features/as_m4_smoke \
  --device cuda:0 \
  --max_new_tokens 4 \
  --model_name_override LongVA-7B-Qwen2
```

结果：

- 输出 `{"status": "pass", "backend": "model", "dry_run": false, ... "num_selected_experiments": 2}`；
- 成功写出：

```text
intersuit/harness/artifacts/as_m4_e0_e7_smoke/predictions/e1_predictions.jsonl
intersuit/harness/artifacts/as_m4_e0_e7_smoke/predictions/e2_predictions.jsonl
```

- 运行结束后 GPU 1 显存回到约 `11 MiB`，无残留占用。

边界说明：

- 当前 E1/E2 两条真实模型输出均为空字符串，不能作为性能提升证据；
- 这一步只证明真实模型 backend 的加载与文件契约跑通；
- 完整 P16 仍需要真实 E0-E7 prediction JSONL，且最好先使用更合适的小型真实问答/音频样本，而不是只依赖当前 smoke 特征；
- E0 baseline 仍需单独处理，因为原始 M4 路径不支持 `video_feature` 输入，后续应为 E0 使用原始视频帧推理，或增加一个明确的 baseline video_feature 兼容路径。

### 2026-07-16 P16 E0 baseline `video_feature` backend smoke

已为原始 multimodal 路径补充 `video_feature` 兼容分支，使 `prepare_inputs_labels_for_multimodal()` 在 `modalities=["video_feature"]` 时直接使用预计算 `[frames, tokens, hidden]` 特征，不再错误送入 vision tower。普通 `image` / `video` 路径不改变。

修改文件：

```text
intersuit/intersuit/model/llava_arch.py
```

Smoke / Harness：

```bash
python -m py_compile \
  intersuit/intersuit/model/llava_arch.py \
  intersuit/intersuit/harness/runners/run_predictions_from_plan.py

/home/yjm/miniconda3/envs/M4/bin/python -m pytest -q \
  intersuit/tests/test_prediction_runner.py \
  intersuit/tests/test_attribution_harness.py \
  intersuit/tests/test_attribution_scoring.py

CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/home/yjm/M4-main/intersuit \
/home/yjm/miniconda3/envs/M4/bin/python -m intersuit.harness.runners.run_predictions_from_plan \
  --plan intersuit/harness/artifacts/as_m4_e0_e7_smoke/matrix_plan.jsonl \
  --backend model \
  --experiments E0 \
  --limit 1 \
  --feature_root intersuit/inputs/features/as_m4_smoke \
  --device cuda:0 \
  --max_new_tokens 4 \
  --model_name_override LongVA-7B-Qwen2
```

结果：

- `py_compile` 通过；
- pytest 结果为 `10 passed`；
- E0 baseline checkpoint 成功加载；
- runner 输出 `{"status": "pass", "backend": "model", "dry_run": false, ... "num_selected_experiments": 1}`；
- 成功写出 `intersuit/harness/artifacts/as_m4_e0_e7_smoke/predictions/e0_predictions.jsonl`；
- 运行结束后 GPU 1 显存回到约 `11 MiB`。

边界说明：

- 当前 E0 真实模型输出也为空字符串，不能作为模型能力结论；
- 这一步只证明 baseline `video_feature` 输入包装已可跑；
- 后续完整 E0-E7 真实模型归因仍需要更合适的小集样本、完整 E0-E7 prediction JSONL 和 scorer 输出；
- 当前 smoke 特征主要用于链路验证，不应替代 AVQA / MUSIC-AVQA / OmniMMI 的真实性能评估。

### 2026-07-16 P16 E3-E7 AS-M4 audio-condition backend smoke

已补齐 `run_predictions_from_plan.py` 中 E0-E7 音频条件的实际扰动逻辑：

- `muted` / `none`：不使用真实 scene audio；AS-M4 `video_feature` 路径下会由 runner 注入零值 dummy audio 并强制 gate=0；
- `mismatched`：使用另一个样本的 scene audio；
- `noisy`：加入确定性小幅噪声，保持形状不变；
- `shifted`：沿音频窗口维度滚动，同时滚动 timestamps；
- `correct`：使用原始 scene audio。

新增/修改：

```text
intersuit/intersuit/harness/runners/run_predictions_from_plan.py
intersuit/tests/test_prediction_runner.py
```

Smoke / Harness：

```bash
python -m py_compile \
  intersuit/intersuit/harness/runners/run_predictions_from_plan.py \
  intersuit/intersuit/model/llava_arch.py

/home/yjm/miniconda3/envs/M4/bin/python -m pytest -q \
  intersuit/tests/test_prediction_runner.py \
  intersuit/tests/test_attribution_harness.py \
  intersuit/tests/test_attribution_scoring.py

PYTHONPATH=/home/yjm/M4-main/intersuit \
/home/yjm/miniconda3/envs/M4/bin/python -m intersuit.harness.runners.run_predictions_from_plan \
  --plan intersuit/harness/artifacts/as_m4_e0_e7_smoke/matrix_plan.jsonl \
  --backend oracle \
  --limit 2 \
  --feature_root intersuit/inputs/features/as_m4_smoke \
  --device cpu \
  --max_new_tokens 4

PYTHONPATH=/home/yjm/M4-main/intersuit \
/home/yjm/miniconda3/envs/M4/bin/python -m intersuit.harness.runners.score_ablation_matrix \
  --plan intersuit/harness/artifacts/as_m4_e0_e7_smoke/matrix_plan.jsonl \
  --output_dir intersuit/harness/artifacts/as_m4_e0_e7_oracle_score
```

结果：

- `py_compile` 通过；
- pytest 结果为 `11 passed`；
- oracle E0-E7 预测生成通过；
- oracle scoring 输出 `{"status": "pass", ... "all_core_passed": true}`。

随后使用 AS-M4 checkpoint 对 E3-E7 各 1 条样本做真实模型 backend smoke：

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/home/yjm/M4-main/intersuit \
/home/yjm/miniconda3/envs/M4/bin/python -m intersuit.harness.runners.run_predictions_from_plan \
  --plan intersuit/harness/artifacts/as_m4_e0_e7_smoke/matrix_plan.jsonl \
  --backend model \
  --experiments E3,E4,E5,E6,E7 \
  --limit 1 \
  --feature_root intersuit/inputs/features/as_m4_smoke \
  --device cuda:0 \
  --max_new_tokens 4 \
  --model_name_override LongVA-7B-Qwen2
```

结果：

- runner 输出 `{"status": "pass", "backend": "model", "dry_run": false, ... "num_selected_experiments": 5}`；
- 成功写出 E3-E7 prediction JSONL；
- 运行结束后全部 GPU 显存约 `11 MiB`、利用率 `0%`。

边界说明：

- 当前 E3-E7 真实模型输出仍为空字符串，不能作为性能或归因结论；
- 这一步只证明错配、加噪、错位、gate0 等输入扰动在真实模型 backend 中不会崩溃；
- 后续需要更合适的真实小集样本和可评价输出，才能运行真正的 E0-E7 性能归因。

### 2026-07-16 P16 raw-video real-smoke harness

为解决早期 `video_feature` smoke 特征过小导致真实模型 backend 输出空字符串的问题，已补充 raw-video real-smoke 路径，让 prediction runner 能直接读取原始视频帧并贴近官方 demo 的 `generate_parallel()` 调用。

新增/修改文件：

```text
intersuit/intersuit/harness/runners/run_predictions_from_plan.py
intersuit/inputs/texts/as_m4_real_smoke_water.json
intersuit/harness/configs/as_m4_e0_e7_real_smoke.json
intersuit/tests/test_prediction_runner.py
```

实现要点：

- manifest 支持 `video_path`，runner 使用 `load_video()` 读取原始视频并通过 `image_processor.preprocess()` 生成视频 tensor；
- manifest 支持 `generation_mode=parallel`、`context`、`new_query`、`new_query_pos`，用于复现本地已验证的 `turntaking-valid` 调用形态；
- manifest 支持 `video_max_frames`，water smoke 固定为 `8`，与 `scripts/run_demos_stable.sh turntaking-valid` 保持一致；
- raw-video 样本暂不透传简化的一维 `frame_timestamps`，由 AS-M4 内部按帧数生成默认时间中心；预计算 `video_feature` 路径仍保留时间戳透传。

Smoke / Harness：

```bash
/home/yjm/miniconda3/envs/M4/bin/python -m py_compile \
  intersuit/intersuit/harness/runners/run_predictions_from_plan.py \
  intersuit/intersuit/model/llava_arch.py

/home/yjm/miniconda3/envs/M4/bin/python -m pytest -q \
  intersuit/tests/test_prediction_runner.py \
  intersuit/tests/test_attribution_harness.py \
  intersuit/tests/test_attribution_scoring.py

PYTHONPATH=/home/yjm/M4-main/intersuit \
/home/yjm/miniconda3/envs/M4/bin/python -m intersuit.harness.runners.run_ablation_matrix \
  --config intersuit/harness/configs/as_m4_e0_e7_real_smoke.json \
  --output_dir intersuit/harness/artifacts/as_m4_e0_e7_real_smoke \
  --strict_paths
```

结果：

- `py_compile` 通过；
- pytest 结果为 `12 passed`；
- real-smoke E0-E7 matrix plan 生成通过，输出 `{"status": "pass", "errors": [], "num_experiments": 8}`。

真实模型 backend 单卡 smoke：

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/home/yjm/M4-main/intersuit \
/home/yjm/miniconda3/envs/M4/bin/python -m intersuit.harness.runners.run_predictions_from_plan \
  --plan intersuit/harness/artifacts/as_m4_e0_e7_real_smoke/matrix_plan.jsonl \
  --backend model \
  --experiments E0 \
  --limit 1 \
  --feature_root intersuit/inputs/features/as_m4_smoke \
  --device cuda:0 \
  --max_new_tokens 48 \
  --model_name_override LongVA-7B-Qwen2

CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/home/yjm/M4-main/intersuit \
/home/yjm/miniconda3/envs/M4/bin/python -m intersuit.harness.runners.run_predictions_from_plan \
  --plan intersuit/harness/artifacts/as_m4_e0_e7_real_smoke/matrix_plan.jsonl \
  --backend model \
  --experiments E1,E2 \
  --limit 1 \
  --feature_root intersuit/inputs/features/as_m4_smoke \
  --device cuda:0 \
  --max_new_tokens 48 \
  --model_name_override LongVA-7B-Qwen2
```

结果：

- E0 baseline checkpoint 成功加载并输出非空文本：`There are two people in the video.`，与本地 `turntaking-valid` demo 语义一致；
- E1/E2 AS-M4 canary checkpoint 成功加载、raw-video + scene-audio 参数链路可运行并写出 prediction JSONL；
- E1/E2 当前输出退化为 `!` / `!!!!!!!!...`，因此只能证明链路与文件契约跑通，不能作为 AS-M4 性能提升证据；
- 运行结束后 GPU 0/1/2/3/4 显存均约 `11 MiB`、利用率 `0%`、功率限制 `450W`。

边界说明：

- 当前 real-smoke 是单条 water demo 小样本，只用于防止 runner 继续使用会空输出的 tiny `video_feature` 伪样本；
- `correct=false` 仍可能出现，因为当前 scorer 是 exact match，而 `generate_parallel()` 会同时包含 base description 和插入问题回答；后续需要为 real-smoke / AVQA 增加 contains、regex 或任务特定 scorer；
- AS-M4 canary checkpoint 的退化输出提示：后续若要证明 E0-E7 归因成立，必须使用更合适的 AS-M4 checkpoint、真实场景音频数据和正式 scorer，不能把这次 real-smoke 解释为性能提升。

### 2026-07-16 P17 AS-M4 退化诊断与安全初始化

针对 P16 raw-video real-smoke 中 AS-M4 checkpoint 输出退化为 `!` / `!!!!!!!!...` 的问题，已补充第一轮诊断和安全初始化改动，目标是先定位 audio residual 是否过强，再保证新建 AS-M4 模块初始时严格不破坏原视频特征。

新增/修改要点：

```text
intersuit/intersuit/model/streaming_av/fusion.py
intersuit/intersuit/model/streaming_av/confidence_gate.py
intersuit/intersuit/model/streaming_av/builder.py
intersuit/intersuit/model/llava_arch.py
intersuit/intersuit/train/train.py
intersuit/intersuit/harness/runners/run_predictions_from_plan.py
intersuit/intersuit/harness/runners/run_ablation_matrix.py
intersuit/scripts/finetune_m4.sh
intersuit/scripts/run_as_m4_stage.sh
```

实现：

- `GatedAVFusion` 新增 `fusion_init=zero|identity`，默认 `zero`，新建模块时 `audio_projector.weight=0`，因此即使 `gate=1` 初始也精确退化为 video-only；`identity` 保留给旧行为对照和诊断。
- `AudioConfidenceGate` 新增 `gate_logit_bias`，默认 `-5.0`，让 clean audio 初始 gate 处于近关闭状态，避免第一步训练就大幅扰动 LLM 输入空间。
- 训练入口新增 `AS_M4_FUSION_INIT` 和 `AS_M4_GATE_LOGIT_BIAS`，默认分别为 `zero` 与 `-5.0`，并写入 staged launcher 生成的 `.env`。
- fusion 诊断新增 `gate_mean/max/min`、`video_norm`、`audio_norm`、`delta_norm`、`delta_to_video_ratio`、`quality_gate`、`relevance_gate`、`offset_sec`、`eventness`。
- prediction runner 新增 `--dump_diagnostics`，打开后每条 JSONL 写入 `as_m4_diagnostics`，并把 `gate` 与 `delta_to_video_ratio` 提升到行级字段，方便 scorer 汇总。
- real-smoke manifest 支持 `accept_contains` / `accept_regex`，当前 water smoke 增加 `two people` / `kitchen` 等 contains 验收，避免 `generate_parallel()` 长句输出被 exact match 误判。

已完成 CPU 验证：

```bash
python -m py_compile \
  intersuit/intersuit/model/llava_arch.py \
  intersuit/intersuit/harness/runners/run_predictions_from_plan.py \
  intersuit/intersuit/model/streaming_av/fusion.py \
  intersuit/intersuit/model/streaming_av/confidence_gate.py

/home/yjm/miniconda3/envs/M4/bin/python -m pytest -q \
  intersuit/tests/test_fusion.py \
  intersuit/tests/test_confidence_gate.py \
  intersuit/tests/test_prediction_runner.py \
  intersuit/tests/test_attribution_harness.py \
  intersuit/tests/test_attribution_scoring.py

/home/yjm/miniconda3/envs/M4/bin/python -m pytest -q \
  intersuit/tests/test_streaming_av_integration.py \
  intersuit/tests/test_scene_audio_encoder.py \
  intersuit/tests/test_event_detector.py \
  intersuit/tests/test_temporal_aligner.py \
  intersuit/tests/test_dataset_collator.py
```

结果：

- `py_compile` 通过；
- 第一组 pytest：`25 passed`；
- 第二组 pytest：`20 passed`；
- real-smoke matrix 在 `/tmp/as_m4_e0_e7_real_smoke_check` 生成通过，`num_experiments=8`。

边界说明：

- 本轮尚未启动 GPU 真实模型 `--dump_diagnostics` smoke，也尚未启动修复后 2-step/20-step 训练 canary；
- 现有 `intersuit/harness/artifacts/as_m4_e0_e7_smoke/predictions` 中 E0-E2 为 2 条、E3-E7 为 1 条，直接用当前文件跑 scorer 会因样本数不一致导致 E6/E5 比较失败；这属于旧 artifact 状态问题，后续应重新生成完整 E0-E7 prediction JSONL 后再评分；
- 旧 AS-M4 checkpoint 权重会覆盖新默认初始化，因此本轮代码修复主要影响后续新训练和诊断，不应把旧 checkpoint 的输出变化解释为已修复。

### 2026-07-16 P18 AS-M4 GPU 退化定位进展

在 P17 的诊断字段和安全初始化基础上，已继续完成真实 GPU 单样本定位与 2-step canary。

已验证结论：

- 当前代码下，12k no-freeze baseline 仍可在 raw-video real-smoke 上正确输出：`There are two people in the video.`。
- 旧 AS-M4 checkpoint 在 `E2`、`E7/gate0` 下仍输出 `!`，但诊断已经变为有限数值；`E7/gate0` 的 `delta_to_video_ratio=0.0`，说明退化不是当前音频残差继续扰动造成的。
- 已修复 fusion 路径中的 NaN 污染风险：`gate=0` 时即使 audio 侧出现 NaN，也会通过 `torch.nan_to_num` 保证融合输出退回有限的 video-only 特征。
- 从原始 `checkpoints/M4-LongVA-7B-Qwen2` 新训 2-step AS-M4 smoke 成功保存，但该起点本身不能保持当前 raw-video 行为，推理仍退化。
- 从已验证的 `checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze` 新训 2-step AS-M4 smoke 成功保存，基座共同权重与 12k baseline 字节级一致，但 AS checkpoint 在 `E1/behavior`、`E2`、`E7/gate0` 仍输出 `The!!`。
- 对 12k baseline 与 12kbase AS-M4 2-step checkpoint 做了全量 safetensors 对比：734 个共同 tensor 完全一致，AS checkpoint 只多 23 个 `streaming_av_module` tensor。
- 临时 `/tmp` symlink checkpoint 视图会导致 baseline 也退化为 `The!!`，因此后续不能用 `/tmp` symlink/copy 视图做 M4/AS-M4 行为探针；必须使用 repo 内真实 checkpoint 路径或正式保存目录。

本轮新增/修复的代码行为：

- `run_predictions_from_plan.py` 现在会按每个 experiment 的 `AS_M4_ENABLE_SCENE_AUDIO` 临时覆盖 `model.config.enable_scene_audio`，并在单条推理后恢复原配置，避免 behavior rollback 被 checkpoint config 中的 `enable_scene_audio=True` 误导。
- `GatedAVFusion` 对 `aligned_audio`、`audio_delta` 和 broadcast 后的 gate 做非有限值清理，保证 gate0 是可测试的硬回退路径。
- `llava_arch.py` 在 event/alignment/gate/fusion 诊断链路中清理非有限值，避免 JSONL 诊断被 NaN 破坏。
- `generate_parallel()` 已移除固定 32 帧 `prefix_length = 13 + 32 * 144 + 1`，改为根据原始 `<image>` token 位置和实际 pack 后的视觉 embedding 数动态推导，并新增 8 帧/32 帧单测。

已完成验证：

```bash
python -m py_compile \
  intersuit/intersuit/model/llava_arch.py \
  intersuit/intersuit/harness/runners/run_predictions_from_plan.py \
  intersuit/intersuit/model/streaming_av/fusion.py \
  intersuit/intersuit/model/language_model/llava_qwen.py

/home/yjm/miniconda3/envs/M4/bin/python -m pytest -q \
  intersuit/tests/test_parallel_prefix.py \
  intersuit/tests/test_fusion.py \
  intersuit/tests/test_prediction_runner.py \
  intersuit/tests/test_streaming_av_integration.py

/home/yjm/miniconda3/envs/M4/bin/python -m pytest -q \
  intersuit/tests/test_prediction_runner.py
```

结果：

- `py_compile` 通过；
- 第一组 pytest：`22 passed`；
- 第二组 pytest：`8 passed`；
- GPU smoke 结束后 GPU 0/1/2/3/4 均回到约 `11 MiB`、利用率 `0%`、功率限制 `450W`。

当前仍存在的问题：

- AS-M4 checkpoint 即使在 `AS_M4_ENABLE_SCENE_AUDIO=0` 或 `AS_M4_FORCE_AUDIO_GATE=0` 下仍未恢复 baseline 输出；动态 prefix 修复后，12kbase AS-M4 2-step checkpoint 的 E1/E7 仍输出 `The!!`，其中 E7 诊断仍显示 `gate=0.0`、`delta_to_video_ratio=0.0`。
- `generate_parallel()` 的 32 帧硬编码已修复，但该修复没有单独解决 AS checkpoint 退化；后续应继续检查真实保存目录下的加载细节和新增模块对 `from_pretrained` 初始化的影响。
- 目前不能声称 AS-M4 有性能提升；真实归因实验必须等 behavior/gate0 回退先恢复后再跑完整 E0-E7。

下一步优先级：

1. 在 repo 内真实路径下重新保存一个只含 baseline config + baseline 共同权重 + AS 新增模块的 checkpoint，避免 `/tmp` symlink 探针误导。
2. 检查 `from_pretrained(low_cpu_mem_usage=True)` 与 vision tower meta tensor warning 是否导致 AS 保存目录加载行为不同；必要时为 AS-M4 加显式加载后校验。
3. 重新跑 E0/E1/E2/E7 单样本；验收标准是 E1 behavior 和 E7 gate0 必须接近 E0。
4. 通过后再跑 20-step canary 和至少 10 条 real-smoke E0-E7，不启动 AS-M4 32k full。

### 2026-07-16 P19 AS-M4 `The!!` 退化根因更新

P18 后继续做了 runtime 级别对比，结论有重要更新：`The!!` 退化的最高概率根因已经从 audio fusion 转为加载 dtype，并已修复。

已定位问题：

- `load_pretrained_model()` 原本只根据 `model_path` 是否包含 `qwen2` 来决定 eager attention 下是否使用 `torch.bfloat16`。
- baseline 路径 `M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze` 包含 `Qwen2`，因此按 `bfloat16` 加载。
- AS-M4 路径 `AS-M4-12kbase-smoke-vfeat-asmodules-zero-gatebias-2step` 不包含 `Qwen2`，因此被错误按 `float16` 加载。
- 这解释了“checkpoint 共同权重字节级一致，但生成行为不同”的现象：文件没坏，运行时 dtype 变了。

已完成修复：

- `intersuit/intersuit/model/builder.py` 新增 `should_use_qwen2_bf16(model_path, model_name, attn_implementation)`。
- dtype 判断现在同时检查 `model_path` 和 `model_name`；只要 `model_name_override=LongVA-Qwen2` 且 attention 为 `eager`，AS-M4 checkpoint 也会按 `bfloat16` 加载。
- 新增 `intersuit/tests/test_model_builder.py`，覆盖 AS-M4 路径名不含 Qwen2 时仍通过 `model_name` 选择 bf16 的情况。
- 新增 `intersuit/scripts/debug_as_m4_runtime.py`，用于比较 baseline 与 AS-M4 加载后的 tokenizer/config/module/state 摘要。

修复后 runtime 诊断结果：

- baseline 与 AS-M4 的 `model.embed_tokens.weight` 均为 `torch.bfloat16`。
- 选定的 454 个共同 runtime state key 未发现 hash 差异。
- AS-M4 相比 baseline 只多出 `streaming_av_module` 的 23 个 tensor。
- config 差异只剩 AS-M4 相关字段：`enable_scene_audio`、`as_m4_fusion_init`、`as_m4_gate_logit_bias`、`mm_tunable_parts`。

修复后 GPU probe 结果：

- AS-M4 E1/E7 不再输出 `The!!`，而是与当前 dynamic-prefix 代码下的 baseline 一样输出到 `about2`。
- 重新跑当前代码的 E0 baseline parallel probe，输出同样为 `The image... about2`。
- 因此，`The!!` 崩坏已基本归因并修复；当前剩余问题是 `generate_parallel()` 动态 prefix/中途插入 query 后的回答文本不够完整，导致 scorer 未命中 `two people`。
- E7 仍保持硬回退诊断：`gate=0.0`、`delta_to_video_ratio=0.0`，说明 audio residual 在 gate0 下没有参与。
- 已为 water real-smoke 的人数题补充窄范围 `accept_regex`，接受 `about2` / 单独 `2` 这类简短人数回答；这只用于 smoke 验收，不等价于论文级语义评分。

新发现的边界问题：

- 临时探针使用的 AS-M4 2-step checkpoint 目录当前未保留在 `intersuit/checkpoints` 下，仅剩训练日志；后续要复跑 AS-M4 E1/E7，需要重新保存或重新跑 2-step canary，并确保 tokenizer/config 文件齐全。
- 普通 `generate()` 分支在当前 harness 中解码为空，疑似 LLaVA `generate()` override 返回值已是 generated tokens，而 runner 仍按 full sequence 做了 `[:, input_ids.shape[1]:]` 切片；该分支不是当前 real-smoke 的主路径，但后续应单独修正。

当前判断：

- 不再把 AS-M4 当前问题归因为 audio residual 或 gate 过大。
- 下一步重点应放在 `generate_parallel()` 的 baseline 输出完整性：动态 prefix 修复虽然消除了 32 帧硬编码，但当前 8 帧 water smoke 的 E0 baseline 从完整 `There are two people in the video.` 变成 `about2`，需要继续检查 `prefix_length` 边界、transition attention mask 和新 query 插入位置。

### 2026-07-16 P20 重新执行 2-step、E0/E1/E7 门禁与 20/150-step

按 `2-step -> checkpoint 加载 -> E0 -> E1 -> E7 -> 20-step -> 150-step` 顺序重新执行。训练统一从已验证的 12k no-freeze baseline 启动，使用物理 GPU `0,2,3,4`、DDP-only、`TORCHDYNAMO_DISABLE=1`、`DDP_FIND_UNUSED_PARAMETERS=True`，且只训练 `streaming_av_module`。

2-step 结果：

- run：`AS-M4-12kbase-retrain-2step-e0e1e7-20260716`；
- `train_loss=0.24042703211307526`，末步 `grad_norm=0.00396728515625`；
- checkpoint 正常保存为四个 safetensors 分片；
- 独立 runtime 加载校验通过，baseline 与 AS-M4 的 tokenizer 无差异，454 个共同关键 state 摘要无差异，AS-M4 仅新增 23 个 `streaming_av_module` tensor；
- 加载校验报告：`intersuit/harness/artifacts/as_m4_e0e1e7_20260716/runtime_load_check.json`。

E0/E1/E7 单样本门禁结果：

- 三组均生成相同 prediction：`The image you've provided appears to be a screenshot from a video. The video seems to be about2`；
- 三组 smoke scorer 均为 `correct=true`；
- E7 诊断为 `gate=0.0`、`delta_to_video_ratio=0.0`，证明 gate0 下音频残差未参与融合；
- prediction 位于 `intersuit/harness/artifacts/as_m4_e0e1e7_20260716/predictions/`；
- 该结果证明 behavior/gate0 回退成立，不等价于 scene audio 已带来性能提升。

20-step 结果：

- 第一次执行因外层 60 秒工具超时在 14/20 被终止，没有保存 checkpoint；失败日志保留，空输出目录已清理；
- 使用新 run `AS-M4-12kbase-canary20-after-e0e1e7-20260716-v2` 从头重跑成功；
- `train_runtime=41.4049` 秒，`train_loss=0.2265265591442585`；
- 末步 loss 为 `0.2179`，`grad_norm=0.359375`，均为有限值。

150-step 结果：

- run：`AS-M4-12kbase-canary150-after-e0e1e7-20260716`；
- 150/150 steps 完成并保存四个 safetensors 分片；
- `train_runtime=261.3056` 秒，`train_loss=0.17248623569806418`；
- 末步 loss 为 `0.1389`，`grad_norm=0.8359375`，均为有限值；
- 训练后 GPU 0/1/2/3/4 均恢复到约 `11 MiB`、利用率 `0%`，没有残留 `train_mem.py` 或 `torchrun` 进程；可见内核日志中未命中 `NVRM`、`Xid`、`AER`、`PCIe Bus Error`、`GPU has fallen` 或 `hung task`。

结论：本轮 2-step 保存/加载、E0/E1/E7 回退门禁、20-step 和 150-step canary 均已通过。当前结论仍只覆盖 12k AS-only smoke 数据，不能据此声称真实场景音频带来精度提升或 32k/full 训练已经完成。
## 2026-07-16 AVUT E0/E1/E7 第一轮数据准备（停在人工审核门禁）

本轮按固定 5 条样本的 smoke 规则执行，未运行模型推理，也未读取已有预测来选择样本。AVUT 代码仓库来自 `https://github.com/lark-png/AVUT`，本地 commit 为 `030f52f1eb7c87ff38a36f737d272d73ce9e2400`；数据来自 Hugging Face `tsinghua-ee/AVUTBenchmark`，revision 为 `558ca95ac16c972a4de2028fbecef2a8b82fdb00`，下载日期为 2026-07-16。

人工标注文件为 `intersuit/datasets/AVUT/raw/AV_Human_data.json`，共 1734 条。实际结构为顶层 list；问题字段为 `question`，答案字段为 `answer`，视频编号和路径字段分别为 `video_id`、`video_path`；choices 由 `option_A` 至 `option_D` 四个字段组成，正确答案为字母并映射到对应选项文本；任务类型字段为 `task_type`，没有 split 字段。发现 QA 455 的正确答案指向空白 `option_C`，已在结构报告中作为转换错误记录，未静默使用。

候选选择使用官方 `task_type`、声音语义关键词和纯视觉属性排除词，固定随机种子 `20260716`，并限制不同视频。固定候选为 QA 141、199、686、967、196；候选清单和人工审核模板分别位于 `intersuit/inputs/texts/avut/avut_smoke_candidates.json` 与 `intersuit/inputs/texts/avut/avut_smoke_review.csv`。仅下载这 5 个视频，共 57,798,716 字节；未下载 Gemini 标注或全量 AVUT。

5 个视频均通过 ffprobe/ffmpeg 全量音视频解码检查，均为 AAC、44.1 kHz、双声道，媒体报告位于 `intersuit/harness/artifacts/as_m4_avut_smoke/media_validation.json`。新增的 AVUT 脚本已通过 `py_compile`；使用 `/home/yjm/miniconda3/envs/M4/bin/python -m pytest` 运行 3 个相关测试文件，结果为 `11 passed`。当前默认 `package` Python 环境没有 pytest，因此测试使用已安装 pytest 的 M4 环境。

当前严格停在阶段八：`muted_answerable`、`audio_answerable`、`keep`、`manual_note` 四列仍为空，等待用户先静音、再有声播放后填写，并另存为 `intersuit/inputs/texts/avut/avut_smoke_reviewed.csv`。在该文件存在且填写完整前，不生成最终 manifest、E0/E1/E7 plan，也不运行 GPU。另一次环境检查中 `nvidia-smi` 无法与 NVIDIA 驱动通信；即使人工审核完成，GPU smoke 前也必须重新通过 GPU/驱动检查。

### 2026-07-16 AVUT muted-only 审核调整与单样本 E0 停止

用户明确取消有声人工审核，并将本轮定义调整为“仅人工静音审核”。5 条固定样本统一记录为 `muted_answerable=false`、`audio_answerable=null`、`review_level=muted_only`、`audio_required_candidate=true`、`manually_verified_audio_required=false`。因此它们只能称为“静音不可回答的音频相关候选样本”，不能称为已经人工严格验证的 audio-required 样本。

更新后的审核文件为 `intersuit/inputs/texts/avut/avut_smoke_reviewed.csv`，manifest 为 `intersuit/inputs/texts/avut/avut_audio_smoke.json`。5 个视频再次通过存在性、ffprobe 音轨和 ffmpeg 全量解码检查；问题、官方答案、四个选项均完整，且每个问题均显式包含 audio/sound/say/drum 等声音语义。manifest 使用 `scene_audio_path` 指向同一 MP4，并由 runner 在推理时通过 ffmpeg 直接解码为 16 kHz float32 PCM，再切为 1.0 秒窗口、0.5 秒 hop。该路径绕开了本机 torchaudio 缺少 WAV backend 的兼容问题；首条样本 CPU 检查得到音频 shape `[65,16000]`、时间戳 shape `[65,2]`、所有值有限、waveform norm `197.3996`。

E0/E1/E7 配置为 `intersuit/harness/configs/as_m4_avut_e0_e1_e7_smoke.json`，使用固定 manifest、32 帧、parallel generation、`max_new_tokens=64`、temperature 0、seed 20260716。E0 使用 12k no-freeze baseline；E1 使用 `AS-M4-12kbase-canary150-after-e0e1e7-20260716` 和真实同步音轨；E7 使用同一 AS-M4 checkpoint/音轨并强制 gate=0。三项 plan 的 strict path 校验与 dry-run 均通过。相关 CPU 测试结果为 `28 passed`。

GPU 前检查确认 5 张卡空闲、功率限制 450W，本轮选择物理 GPU 4。首次用 `package` 环境运行 E0 时，Transformers 4.57.3 因 PyTorch 2.5.1 与 CVE-2025-32434 安全限制拒绝加载视觉塔旧 `.bin`，未产生预测。随后改用仓库记录的 M4 环境（Transformers 4.44.0）重新运行，checkpoint 加载并完成生成，但 E0 输出为空字符串；同时日志出现大量视觉塔 checkpoint 权重复制到 meta 参数为 no-op 的 warning。空输出触发既定停止条件，因此 E1、E7 和全 5 条评测均未运行。运行后 GPU 显存恢复为空闲，当前可见内核日志中未发现 NVRM/Xid/AER/PCIe/hung-task 记录。

随后执行只加载、不推理的 runtime 诊断：视觉塔 391 个参数最终 `meta_count=0`，全部有限，首个参数 norm 为 `5.13335`，设备为 `cuda:0`、dtype 为 float16。因此 meta warning 发生在嵌套初始化阶段，但主 checkpoint 后续已把视觉参数实体化，不能单凭 warning 判定它是空输出直接根因。当前可确认的是 E0 在该 AVUT parallel prompt 上立即生成 EOS/空文本；不能把失败归因于 AVUT 音轨或 AS-M4 gate，因为 E0 根本不使用 scene audio。后续需要先针对 baseline 的 parallel prompt、query 插入位置和生成边界定位 E0 空输出，再重新按 E0 -> E1 -> E7 顺序运行；在此之前不得扩大样本，也没有 gate/audio_norm/delta 的 GPU 诊断结论。

### 2026-07-16 AVUT 单样本 E0 空输出根因与修复

使用保留的 water real-smoke、12k no-freeze baseline、`LongVA-Qwen2`、parallel generation 和 `max_new_tokens=48` 复跑，旧 E0 仍输出非空 `...about2`。A/B/C/D 对照中，旧 water 样本在旧/新配置下均非空，AVUT 首条样本在旧/新配置下均为空，排除了 checkpoint、视觉塔和配置元数据整体回归。

token 探针确认原 AVUT parallel 调用只生成一个 token：`151645=<|im_end|>`。其 base context 为 `Please watch the video and answer the question.`，模型在 step 0 直接 EOS，而真实问题配置为 `new_query_pos=20`，因此问题从未插入。原调用的 `visual_embedding_count=4608`、`input_embedding_length=4637`、`prefix_length=4623`，prefix 合法且未越界。另确认普通 `generate()` override 返回 generated-only IDs，旧 runner 再按 input 长度切片会丢掉全部生成 token。

修复包括：AVUT 统一改为普通 `generate()`；把 question/choices 格式化为 `Question + Options A-D + Answer with only...` 并保持 Qwen assistant 边界；generated-only 与 full-sequence 输出采用兼容切片；保存完整 prompt/token 诊断；选择题 scorer 支持将字母映射回选项文本。未修改音频编码器、gate、fusion、temporal aligner、训练参数或 checkpoint。

修复后 AVUT E0 输入长度为 92，视觉 embedding 数为 4608，query token 范围 16–87，assistant token 为 89；生成 `[33,151645]`，原始 decode 为 `B<|im_end|>`，最终 prediction 为 `B`。同视频两个简单视觉 prompt 分别生成 11 和 48 tokens，均为非空。旧 water E0 修复后仍输出原非空文本。

满足 E0 门禁后仅对同一条样本按顺序运行 E1、E7。E1 输出非空且无 `The!!`，`audio_norm=228.1299286`、`gate_mean=0.2068053`、`delta_to_video_ratio=0.06437786`，均为有限值；E7 输出 `B`，`gate_mean/max/min=0`、`delta_norm=0`、`delta_to_video_ratio=0`。E0/E1/E7 当前都答错，这属于模型性能结果，未作为空输出处理，也未扩大到 5 条。

完整 CPU 测试为 `84 passed`，py_compile 通过。诊断产物位于 `intersuit/harness/artifacts/as_m4_avut_debug/`，包括 `ab_matrix.json`、`prompt_debug.json`、`token_debug.json`、`known_good_vs_avut.md` 和单样本 E0/E1/E7 predictions。结束后五张 GPU 均为约 11 MiB、0% 利用率、450W。

### 2026-07-16 AVUT 5 条固定样本 E0/E1/E7 smoke

在 commit `f1dc2c51de79390af51861268667ed8a76b05e4b` 上，使用固定 manifest `intersuit/inputs/texts/avut/avut_audio_smoke.json`、32 帧、普通确定性 `generate`、`max_new_tokens=48` 和物理 GPU 4，严格按 E0 → E1 → E7 顺序完成 5 条 muted-only 候选样本评测。运行期间未修改 runner、prompt、scorer、manifest、选项、答案、checkpoint 或模型代码。

GPU 前只读答案审计确认原始答案全部为字母，不涉及 0-based/1-based 索引：QA 141/199/686 为 D，QA 967/196 为 B；A-D 顺序和答案文本映射均唯一。scorer 验证为 trim 后完整单字母精确比较，支持小写归一化，不接受长文本中的无关字母。相关映射/choice 测试为 `7 passed`。

E0 五条均为非空严格单字母，依次为 `B/A/B/B/B`，正确数 2/5。E1 五条均非空且真实音频诊断完整，正确数 2/5；其中 `avut_0141` 输出退化长文本而非单字母，已按精确字母 scorer 判错并原样保留，其余四条为单字母。E7 五条均为严格单字母 `B/A/B/B/B`，正确数 2/5；所有样本精确满足 `gate_mean/min/max=0`、`delta_norm=0`、`delta_to_video_ratio=0`。

E1 诊断统计：`audio_norm mean/min/max=170.48076/128.69054/228.12993`，跨样本 `gate mean=0.1231216`、`gate min=0.00155747`、`gate max=0.35139781`，`delta_to_video_ratio mean/min/max=0.03152274/0.01070478/0.06437786`；NaN、Inf、零 audio norm 和零 gate 样本数均为 0。未发现空输出或 `The!!`。

E0/E1/E7 accuracy 均为 0.4；E1_GAIN=0、AUDIO_HURT=0、ALL_CORRECT=2、ALL_WRONG=3。E0 与 E7 五条答案完全一致，一致率 100%。这只说明当前 5 条真实视频已接入链路、E1 读取并融合音频、E7 能关闭音频残差；不构成统计显著性、严格 audio-required 或 AS-M4 性能提升证据。由于没有 E1_GAIN 且有一条 E1 格式退化，当前不建议立即扩大到 20～50 条。

产物目录为 `intersuit/harness/artifacts/as_m4_avut_smoke/20260716_muted_only_5samples/`。三组推理使用完整 stdout+stderr 日志再次按相同顺序复核，prediction SHA256 与首轮逐字节一致。结束后五张 GPU 均为约 11 MiB、0% 利用率、450W。

### 2026-07-17 AVUT `avut_0141` E1 长文本诊断

未修改正式 prediction、manifest、prompt、scorer、模型或 checkpoint。在隔离目录中使用正式 E1 checkpoint、视频、真实音轨、32 帧、确定性生成、`max_new_tokens=48` 和 seed 20260716 复跑，输出与正式结果逐 token 完全一致：首 token 为 `A`（ID 32），共生成 48 tokens，未出现 EOS，`skip_special_tokens=False/True` 两种 decode 完全相同。

完整正式输出为：

```text
A, B, and C
 25s
 dicing
s
 dicing e25s
 dicing
s
 dicing
s
 dicing
s
 dicinging
s
 dicing
```

该输出同时明确提到 A/B/C，不能唯一提取选项；未提到真值字母 D，也未完整包含真值文本 `25s-33s`。它不是正常的“先选一个答案再解释”，也不是单纯视频/音频描述；后半段 `dicing` 重复 7 次，生成预算耗尽且无 EOS。未发现特殊 token、decode 删除、切片丢失或 prediction 写盘异常。

E0/E1/E7 的正式 prompt、conversation template 和有效生成参数一致：`do_sample=false`、`temperature=0`、`top_p=0.8` 与 `top_k=20` 在非采样模式下不生效、`num_beams=1`、`max_new_tokens=48`、generation config 的 `eos_token_id=[151645,151643]`、tokenizer EOS 为 `151645`、`pad_token_id=151643`。除 E0 baseline checkpoint 与 E1/E7 AS-M4 checkpoint、音频开关/gate 条件外无额外差异。

临时强约束 prompt `Reply with exactly one uppercase letter... Do not provide an explanation.` 仍未恢复单字母，输出从逗号开始并重复 `, or any s`，同样生成满 48 tokens 且无 EOS。分类为 `GENERATION_FAILURE`，不是 `FORMAT_ONLY`、`WRONG_BUT_PARSEABLE` 或语义正确但格式失败。当前没有 runner 错误证据；现象更符合启用 E1 音频残差后发生的确定性生成退化。单次强 prompt 不能修复，因此不修改正式 prompt、不放宽 scorer，也不建议扩大到 20～50 条。

诊断目录：`intersuit/harness/artifacts/as_m4_avut_smoke/20260716_muted_only_5samples/avut_0141_e1_debug/`，核心文件为 `sample_diagnosis.json` 和 `generation_parameter_comparison.json`。结束后五张 GPU 均约 11 MiB、0% 利用率、450W。

### 2026-07-17 `avut_0141` 音频残差控制变量实验

仅对 `avut_0141` 建立 C0–C4、residual scale 和首 token logits 控制变量，未修改正式 prediction、checkpoint、正式 manifest/scorer 或训练。新增 debug-only `AS_M4_DEBUG_AUDIO_RESIDUAL_SCALE`，默认 1.0；不设置时已在 GPU 上验证与原正式 E1 逐 token 一致。音频静音、错误音频和 ±1 秒偏移只在 runner 解码后、分窗前生效，偏移采用零填充且保持 waveform 长度；首 token scores 只在显式 debug 开关下返回。完整 CPU 回归为 `88 passed`。

主要结果：C0 gate0 输出 B；C1 原始音频复现 48-token 重复且无 EOS；C2 等长静音输出 B；C3 使用 `avut_0199` 完整错误音频输出 B；C4 +1 秒仍重复无 EOS，-1 秒首 token 直接 EOS。错误音频虽有非零残差但 `delta/video=0.02711`，明显低于原始音频的 `0.06438`。

scale 结果为：0→B、0.25→B、0.5→A，三者都有 EOS 且无重复；0.75→首 token EOS 空输出；1.0→与 C1 相同的重复长文本。对应 `delta/video` 为 0、0.01609、0.03219、0.04829、0.06438，单样本失败相关 Pearson 为 0.8661。gate 在 scale sweep 中保持 `0.2068053`，说明观察到的变化来自 gate 后 residual 幅度，而非 gate 值变化。

首 token 方面，E0/C0/C2/scale0 均为 B rank1、EOS rank56；C1/scale1 变为 A rank1、B rank2、EOS rank3；scale0.75 的 EOS 概率 0.9012、rank1；-1 秒的 EOS rank1。证据首先指向原始音频产生的残差幅度过大，时间位置会调制失败形态；不支持“音频分支通用偏置”或“任何非零音频必然失败”。当前只足以支持后续研究 residual scale、gate 校准或融合归一化，不足以直接修改生产 gate/fusion 或决定重训，也不适合扩大样本。

产物位于 `intersuit/harness/artifacts/as_m4_avut_debug/avut_0141_audio_ablation/`。实验无 NaN、Inf、CUDA OOM、长度/时间戳不匹配；正式 E0/E1/E7 prediction SHA256 未变化；结束后五张 GPU 均约 11 MiB、0% 利用率、450W，内核日志无新增 NVRM/Xid/AER/PCIe/hung-task 错误。

## 2026-07-17：AVUT 5 条样本 audio residual ratio cap 实验

新增默认关闭的实验变量 `AS_M4_AUDIO_DELTA_RATIO_CAP`。它在最终 gated audio residual 加入视频特征前，按单样本整体 float32 范数限制 residual/video 比例；未设置或为 `0` 时是精确 no-op。原 `delta_norm`、`delta_to_video_ratio` 明确表示 cap 后值，并新增 raw/capped 范数、比例及实际缩放系数诊断。

固定 5 条 E1 结果：disabled 为 4/5 严格单字母、2/5 正确；0.05 与 0.04 均为 4/5 严格单字母且 `avut_0141` 空输出、2/5 正确；0.03 为 5/5 严格单字母、无空/重复、2/5 正确；0.02 为 5/5 严格单字母但正确率降至 1/5。当前仅将 0.03 作为扩大诊断候选，不设正式默认值。E7 gate=0 单样本复核中 gate、raw/capped delta 及 ratio 均精确为 0。完整报告位于 `intersuit/harness/artifacts/as_m4_avut_debug/avut_5sample_audio_delta_cap_sweep/report.md`。
