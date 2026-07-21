# 冻结 300 条音视频任务基线报告

运行日期：2026-07-19（Asia/Shanghai）

## 固定状态

- 运行代码提交：`32fa77aaeb11852b9fd9bae197b5cbe4921216c3`
- 工程快照提交：`3ecc2e6`
- 冻结 manifest SHA256：`ca344ceaec00c30790f1c7179f72a90a3174aae1359c95db18eafe799d62932c`
- 有序样本 ID SHA256：`2db69e5009729f9d00e7fd40d24db88413caa195763b78ce6004cf9d5dfd7394`
- 三组基线 plan SHA256：`0b34956dbad10cda8aaa88062aaa0f6433aa7d8bb358ca7b37251a10012b29cf`
- 运行设备：物理 GPU 4，功率限制 `450W`
- 解码：确定性生成，`max_new_tokens=64`，`model_name_override=LongVA-Qwen2`
- 本轮没有修改 prompt、解码参数、scorer、checkpoint 或模型参数。

模型条件：

1. `M4_ORIGINAL`：12k no-freeze 原始 M4，关闭场景音频。
2. `AS_M4_DUMMY`：AS-M4 150-step checkpoint，正确同步音频，`dummy_waveform_statistics`。
3. `AS_M4_DUMMY_MISMATCHED`：同一 AS-M4 checkpoint，确定性选择下一条不同来源有效音频。

错配映射在运行前逐条校验：300/300 条都有可用的不同来源音频，复用自身音频路径的样本数为 0。

## Canary

两组 canary 使用同一条 `avut_0663`、同一 prompt 和解码参数。

- 原始 M4：视频加载正常，视觉 embedding 数为 4608；没有加载场景音频，输出严格单字母 `A` 后正常 EOS。
- dummy AS-M4：解码 111 个音频窗口，`audio_norm=149.2798`、`gate=0.09879`、`delta_to_video_ratio=0.02110`。
- dummy residual 明确非零，332 个 canary 诊断数值全部有限。
- 两组均无 OOM、NaN、路径错误或残留 GPU 占用。

canary 的单样本答案均错误；这里只用于验证运行链路，不作为精度结论。

## 300 条结果

| 条件 | 音频必要 | 视频必要 | 音视频联合 | 总体 |
|---|---:|---:|---:|---:|
| 原始 M4 | 31/100（31%） | 96/100（96%） | 53/100（53%） | 180/300（60.0%） |
| dummy 正确音频 | 29/100（29%） | 96/100（96%） | 36/100（36%） | 161/300（53.7%） |
| dummy 错配音频 | 26/100（26%） | 96/100（96%） | 38/100（38%） | 160/300（53.3%） |

相对原始 M4：

| 条件 | 新增答对 | 新增答错 | 净变化 | 答案变化 |
|---|---:|---:|---:|---:|
| dummy 正确音频 | 10 | 29 | -19 | 71/300（23.7%） |
| dummy 错配音频 | 5 | 25 | -20 | 65/300（21.7%） |

逐类变化：

| 条件与类别 | 新增答对 | 新增答错 | 答案变化 | 空输出 | 非严格单字母 |
|---|---:|---:|---:|---:|---:|
| 正确音频 / 音频必要 | 8 | 10 | 37 | 12 | 18 |
| 正确音频 / 视频必要 | 0 | 0 | 0 | 0 | 0 |
| 正确音频 / 音视频联合 | 2 | 19 | 34 | 14 | 26 |
| 错配音频 / 音频必要 | 5 | 10 | 33 | 8 | 19 |
| 错配音频 / 视频必要 | 0 | 0 | 0 | 0 | 0 |
| 错配音频 / 音视频联合 | 0 | 15 | 32 | 17 | 27 |

输出完整性：

- 原始 M4：300/300 非空，300/300 为严格单字母。
- dummy 正确音频：274/300 非空，256/300 为严格单字母。
- dummy 错配音频：275/300 非空，254/300 为严格单字母。
- 两组 dummy 均有 300/300 非零 residual；各自 106,270 个诊断数值全部有限。
- 三组日志退出码均为 0，运行后 GPU 4 回到约 11 MiB。

## 产物校验

- 原始 M4 prediction SHA256：`07cba97e1a18f73d1b190bf0693f3e8987a8a1160078041d7179a4ce79277210`
- dummy 正确音频 prediction SHA256：`219b244a137e5a12e71d4a93fc906ed09c3e92c9899a8757ba21b2dea711cf4d`
- dummy 错配音频 prediction SHA256：`86b8ad5ce49ae4ee75234675d8fa238e97aca44f2539b064c3e76a3180b1e7c9`

预测、完整诊断和日志位于
`intersuit/harness/artifacts/frozen_av_task_eval_dev300/`，按项目规则不纳入 Git。

## 结论

当前 dummy AS-M4 没有产生正向收益。正确音频相对原始 M4 净少 19
题，主要退化集中在音视频联合类，并出现空输出和非严格单字母格式失败。
正确音频只比错配音频多答对 1 题，不能证明
`dummy_waveform_statistics` 学到了可靠的音频语义。

因此本轮不调整 dummy gate、residual scale、prompt 或 scorer。下一阶段只接入
“冻结 BEATs → audio projector → 基础 residual 融合 → M4”，第一版只训练新增
projector 和融合层；暂不接动态窗口、复杂 Gate 或 MUG。
