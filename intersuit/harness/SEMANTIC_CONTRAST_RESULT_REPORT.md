# Fixed BEATs 语义对比最终结果报告

日期：2026-07-21（Asia/Shanghai）

## 最终结论

Fixed BEATs semantic contrast `checkpoint-150` 的正式结果门禁为 **FAIL**，不得晋级为
AS-M4 后续主实验 checkpoint。

本次评测严格按预注册顺序执行：先在 reserve326 上校准固定 gate，再冻结
`gate=0.005`，最后在 dev760 上只运行一次 correct/mismatched 正式评测。dev760 结果未用于
重新调 gate，也未补跑其他 gate。

虽然 correct 音频产生了 5 个相对 mismatched 的独有答对样本，但总体准确率低于
video-only，并且 mismatched 反而明显优于 correct。这说明当前 semantic ranking 训练没有
形成可泛化的正确音频语义选择性。

## 被测对象

- checkpoint：`intersuit/checkpoints/AS-M4-BEATs-semantic-contrast-canary150-v1/checkpoint-150`；
- 训练阶梯：`2-step -> 20-step -> 150-step`；
- 训练结果：150/150，`train_loss=0.336477`，loss/grad 全部 finite；
- 推理模式：Fixed Window、固定 gate、无 Offset、无 Dynamic Window、无 learned Gate；
- dev 清单：`intersuit/datasets/AS_M4_STAGE2/dev_manifest.json`；
- dev 清单 SHA256：`0237cdebfb54c2766cd17ab709293862be381fa17ca809d101fbce37d617fb10`。

## Reserve326 校准

只评测预定的 `gate=0.005` 后即停止 reserve 搜索并冻结 gate；未继续窥探
`gate=0.01/0.02`。

| 条件 | 正确数 | 相对 video-only gain/harm/net | 独有答对 |
| --- | ---: | ---: | ---: |
| video-only | 128/326 | - | - |
| correct | 128/326 | 8/8/0 | 5 |
| mismatched | 127/326 | 6/7/-1 | 4 |

correct 与 mismatched 均无空输出或首 token EOS。reserve 门禁通过，因此正式冻结
`gate=0.005`。

## Dev760 正式结果

| 条件 | 正确数 | 准确率 | 相对 video-only gain/harm/net | 相同预测率 |
| --- | ---: | ---: | ---: | ---: |
| video-only | 254/760 | 33.4211% | - | - |
| correct | 251/760 | 33.0263% | 7/10/-3 | 70.0000% |
| mismatched | 259/760 | 34.0789% | 13/8/+5 | 69.4737% |
| silence | 254/760 | 33.4211% | 0/0/0 | 100.0000% |
| gate0 | 254/760 | 33.4211% | 0/0/0 | 100.0000% |

- correct 相对 mismatched 独有答对：5；
- mismatched 相对 correct 独有答对：13；
- correct 平均 gate：`0.0049647444`；
- mismatched 平均 gate：`0.0049654942`；
- correct/mismatched 均为 1 个空输出、1 个首 token EOS，与 video-only 相同；
- 所有条件的非有限诊断数均为 0。

## 分任务观察

| 任务类型 | 样本数 | video-only | correct | mismatched |
| --- | ---: | ---: | ---: | ---: |
| audio | 316 | 100 (31.65%) | 94 (29.75%) | 97 (30.70%) |
| audio_visual | 228 | 79 (34.65%) | 81 (35.53%) | 85 (37.28%) |
| visual | 216 | 75 (34.72%) | 76 (35.19%) | 77 (35.65%) |

correct 在 `audio_visual` 和 `visual` 子集略高于 video-only，但两个子集仍低于
mismatched；在最直接依赖音频的 `audio` 子集上，correct 同时低于 video-only 和
mismatched。因此不能把局部增益解释为正确音频语义选择性。

## 门禁判定

| 门禁 | 结果 |
| --- | --- |
| correct accuracy >= video-only | FAIL |
| correct accuracy > mismatched | FAIL |
| correct 独有增益 > 0 | PASS |
| correct 相对 video-only net gain >= 0 | FAIL |
| correct net gain > mismatched net gain | FAIL |
| silence 精确回退率 >= 99% | PASS |
| gate0 精确回退 | PASS |
| 输出与诊断有效 | PASS |

总门禁按全部条件合取判定，因此最终状态为 **FAIL**。

## 处理决定

1. 不再根据 dev760 结果调整 gate，也不对该 checkpoint 补做新的 dev gate sweep；
2. `checkpoint-150` 仅保留为本次失败实验的可复现结果，不作为主线权重；
3. silence/gate0 精确回退已验证，失败点是语义选择性，而不是回退链路或数值稳定性；
4. 下一轮若继续该方向，应回到训练目标或负样本构造设计，并使用新的实验编号和新的
   reserve/dev 协议，不能复用本次 dev760 作为调参集。

## 收尾验证

- 相关回归测试：`50 passed, 2 warnings`；两条 warning 均为既有 SWIG 类型弃用提示；
- `git diff --check`：PASS；
- postflight：5 张 GPU 均为 `11 MiB`、`0%` 利用率，功率限制记录为 `450 W`；
- 内核日志：未发现本次运行产生的 NVRM Xid、AER、PCIe Bus Error、GPU fallen 或
  hung-task 异常；
- 已清理 2-step 和 20-step 临时 checkpoint，只保留正式 `checkpoint-150`。

## 结果位置

- 正式配置：`intersuit/harness/configs/as_m4_semantic_contrast_dev760_gate0005.json`；
- 汇总 JSON：`intersuit/harness/artifacts/as_m4_semantic_contrast_dev760_gate0005/selectivity_summary.json`；
- 自动报告：`intersuit/harness/artifacts/as_m4_semantic_contrast_dev760_gate0005/selectivity_report.md`；
- correct/mismatched 原始预测：
  `intersuit/harness/artifacts/as_m4_semantic_contrast_dev760_gate0005/predictions/`。
