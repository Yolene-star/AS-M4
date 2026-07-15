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

### 32k OOM 时的保精度降级顺序

如果 AS-M4 新增模块导致 32k OOM，优先牺牲时间来保精度，每次只改一个降级项，并重新执行 `2-step -> 20-step -> 150-step` canary。

优先顺序：

1. 离线预计算冻结场景音频特征，训练时不再运行 scene audio encoder forward；
2. 开启或扩大新增模块的 activation checkpointing；
3. 将 `M4_CHUNKED_LM_LOSS_TOKENS` 从 `512` 降到 `256`，仍 OOM 再降到 `128`；
4. micro batch 降到 `1`，通过增加 gradient accumulation 保持有效 batch；
5. 对 temporal aligner 使用 chunked similarity，分块计算音频/视频相似度矩阵；
6. 缩短训练时的音频 history，但保留评测配置，并记录训练/评测差异；
7. 减少对齐候选范围，但只在事件窗口内使用，不削减原始 32k 文本上下文；
8. 受控启用 CPU optimizer offload，并监控系统内存；
9. 最后才考虑降低 LoRA rank 或冻结更多 AS-M4 新增模块。

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
