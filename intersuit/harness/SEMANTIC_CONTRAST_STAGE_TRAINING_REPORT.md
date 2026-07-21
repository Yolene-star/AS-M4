# Fixed BEATs 语义对比训练阶段报告

日期：2026-07-21（Asia/Shanghai）

## 结论

Fixed BEATs 语义配对辅助目标已完成 `2-step -> 20-step -> 150-step` 训练阶梯。
本阶段只增加训练期 semantic ranking loss，不改变推理结构；Fixed Window、固定 gate、
scene-audio projector 和 residual fusion 保持原链路，Dynamic Window、learned Gate、
Event Aligner 均未启用。

最终 checkpoint：

```text
intersuit/checkpoints/AS-M4-BEATs-semantic-contrast-canary150-v1/checkpoint-150
```

## 数据与目标

- 正样本：原训练清单中的正确场景音频；
- 负样本：同 `source_dataset`、同 `task_type`，且不同物理媒体、不同事件标签的音频；
- 配对清单：`intersuit/datasets/AS_M4_STAGE2/train_manifest_semantic_negative.json`；
- 样本数：2430；
- 清单 SHA256：`c7c74202272ffef53b56277fbd2c0d106e30b1e563b6a7c7f0a87ed70b32725c`；
- 泄漏/媒体审计：PASS，`error_count=0`；
- ranking weight：`1.0`；
- margin：`0.2`；
- 可训练模块：`scene_audio_projector,streaming_av_fusion`；
- 冻结父 checkpoint：`AS-M4-BEATs-fixed-stage2-full-v1-20260720`。

## 阶梯结果

| 阶段 | 状态 | 训练 loss | 备注 |
| --- | --- | ---: | --- |
| 2-step | PASS | 0.3917, 0.4048 | projector/fusion 权重均更新 |
| 20-step | PASS | 0.396217 | 20/20，loss/grad 全部有限 |
| 150-step | PASS | 0.336477 | 150/150，运行 1407.3014 秒 |

150-step 共记录 150 个 loss 和 grad norm：

- loss 范围：`0.2311–0.8098`，全部 finite；
- grad norm 范围：`0.287109375–29.5`，全部 finite；
- 最大 grad norm 出现在 step 22，之后恢复到稳定范围，未出现 NaN/Inf、OOM 或 NCCL 失败；
- 最后一步 loss：`0.2825`；
- 最后一步 grad norm：`0.66796875`。

## 权重核验

相对父 checkpoint：

| 参数 | 父 norm | 新 norm | delta norm | finite |
| --- | ---: | ---: | ---: | --- |
| scene-audio projector weight | 35.642944 | 35.653900 | 1.000345 | 是 |
| scene-audio projector bias | 0.183915 | 0.198726 | 0.019769 | 是 |
| fusion audio projector weight | 4.289510 | 5.221072 | 2.290654 | 是 |

最终配置明确保存：

- `scene_audio_contrastive_weight=1.0`；
- `scene_audio_contrastive_margin=0.2`；
- `scene_audio_window_mode=fixed`；
- `enable_audio_confidence_gate_v1=false`；
- `enable_audio_event_aligner_v1=false`。

## 启动问题与修复

首次启动暴露出训练加载器只按 checkpoint 路径名识别 Qwen；AS-M4 checkpoint 目录名不含
`qwen`，导致模型类和 tokenizer 分支无法识别。加载器现同时读取 `config.model_type` 和
`architectures`，并已增加不依赖目录命名的回归测试。

## 后续边界

本报告只证明训练阶段完成且 checkpoint 有效，不代表 reserve326/dev760 语义选择性效果
门禁已经通过。下一阶段应冻结本 checkpoint，在 reserve326 上校准固定推理 gate，再对
dev760 只执行一次正式 correct/mismatch/silence/gate0 门禁。
