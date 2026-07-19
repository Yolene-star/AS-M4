# 冻结音视频任务基线快照

记录时间：2026-07-19（Asia/Shanghai）

## 代码状态

- 快照前提交：`232f0f3687f09145add9a2bfea16534ace84d9e1`
- 分支：`feat/audio-event-aligner-v1`
- 本文件所在提交用于固定本轮基线运行代码；实际提交号以 `git log -1` 为准。
- 数据、日志、checkpoint、媒体和 `harness/artifacts` 运行产物不纳入 Git。

## 冻结评测集

- manifest：`intersuit/harness/artifacts/frozen_av_task_eval_dev300/frozen_eval.json`
- manifest SHA256：`ca344ceaec00c30790a3174aae1359c95db18eafe799d62932c`
- 有序样本 ID SHA256：`2db69e5009729f9d00e7fd40d24db88413caa195763b78ce6004cf9d5dfd7394`
- 样本数：300
- 分类：`audio_necessary=100`、`video_necessary=100`、`audio_visual_joint=100`
- 媒体校验：300 条视频均可解码；200 条音频相关样本均包含且可解码音轨。

## 固定模型与计划

- 原始 M4：`intersuit/checkpoints/M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze`
- dummy AS-M4：`intersuit/checkpoints/AS-M4-12kbase-canary150-after-e0e1e7-20260716`
- canary 计划：`intersuit/harness/artifacts/frozen_av_task_eval_dev300/baseline_canary_plan.jsonl`
- 300 条计划：`intersuit/harness/artifacts/frozen_av_task_eval_dev300/baseline_plan.jsonl`

canary 固定为同一条样本、同一 prompt 和同一解码参数：

1. `M4_ORIGINAL`：`AS_M4_ENABLE_SCENE_AUDIO=0`，不读取场景音频。
2. `AS_M4_DUMMY`：`AS_M4_ENABLE_SCENE_AUDIO=1`，使用正确同步音频和 `dummy_waveform_statistics`。

300 条固定必跑条件：

1. 原始 M4，关闭场景音频。
2. dummy AS-M4，使用正确同步音频。
3. dummy AS-M4，使用确定性错配音频。

## 执行约束

- canary 未通过时不启动 300 条完整基线。
- 完整基线只记录结果，不修改 prompt、解码参数、模型参数或 scorer。
- 300 条基线完成前不下载新数据，不开始动态窗口、复杂 Gate 或 MUG 训练。
- 每组结果必须分别报告三类准确率、总体准确率、相对原始 M4 的新增答对、新增答错和答案变化率。
