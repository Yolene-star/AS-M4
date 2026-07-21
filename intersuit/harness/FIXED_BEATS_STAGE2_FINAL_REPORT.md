# Fixed Window BEATs Stage 2 结果报告

日期：2026-07-20（Asia/Shanghai）

## 结论

Fixed Window BEATs 基线已完成 Stage 2 正式训练并通过 dev20 小规模效果门禁，
但在锁定的 dev760 语义选择性门禁上未通过，不能作为最终通过方案。

最终链路为：

```text
固定 1 秒音频窗口
-> 冻结 BEATs
-> 可训练 scene-audio projector
-> 固定顺序线性映射
-> 可训练 residual fusion
-> M4
```

训练时固定 gate 为 `1.0`，保证 Projector/Fusion 获得足够梯度。原始推理 gate
为 `0.25`；reserve326 校准后冻结为 `0.005`，再执行一次 dev760。原始音频窗口
被判定为静音时，固定链路会把对应 gate 硬置零。

Dynamic Window Selector、事件对齐和 learned confidence gate 均未用于本次 Fixed 正式训练或效果门禁。

## 正式训练

- 数据：`intersuit/datasets/AS_M4_STAGE2/train_manifest.json`
- 数据量：2430 QA / 1000 个物理媒体
- 基座：`M4-LongVA-7B-Qwen2-train-12k-lowmem-nofreeze`
- 输出：`intersuit/checkpoints/AS-M4-BEATs-fixed-stage2-full-v1-20260720`
- 训练步数：151/151，1 epoch
- `train_loss`：0.41948150596673917
- 训练 gate：1.0
- 原始推理 gate：0.25
- reserve326 冻结推理 gate：0.005
- 可训练模块：`scene_audio_projector,streaming_av_fusion`
- 冻结 BEATs 参数未写入 M4 checkpoint

最终权重检查：

| 参数 | norm | finite | nonzero |
|---|---:|---|---:|
| scene-audio projector weight | 35.6429 | 是 | 2,752,512 |
| scene-audio projector bias | 0.1839 | 是 | 3,584 |
| fusion audio projector weight | 4.2895 | 是 | 12,845,056 |

独立重载时 tokenizer 无差异，454 个共同关键权重与 12k baseline 一致；新增参数只属于 scene-audio/streaming 模块。

## Dev20 门禁

评测使用 Stage 2 dev 前 20 条，覆盖 AVE、AVUT 和 MUSIC。判分器为显式 `label_terms`：单字母 A-D 保持严格匹配；普通标签使用词边界；逗号标签按完整短语匹配；括号限定可被去除。

| 条件 | 答对 | 新增答对 | 新增答错 | 与 baseline 相同 | 空输出 | 首 token EOS | 重复 | mean ratio | max ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M4 video-only | 6 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 |
| Fixed 正确音频 | 7 | 1 | 0 | 7 | 0 | 0 | 0 | 0.02293 | 0.02568 |
| Fixed 静音 | 6 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 |
| Fixed 错配音频 | 7 | 1 | 0 | 8 | 0 | 0 | 0 | 0.02273 | 0.02664 |
| Fixed gate=0 | 6 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 |

本轮门禁通过项：

- 正确音频新增答对 1，多于新增答错 0；
- 静音和 gate=0 均 20/20 精确回退；
- 所有条件均无空输出、首 token EOS、重复文本或非有限诊断；
- 正式训练和评测后 GPU 全部回到约 11 MiB / 0%；
- 当前可见内核日志没有 NVRM、Xid、AER、PCIe Bus Error 或 hung task。

## 限制与下一阶段门禁

正确音频和错配音频在本次 dev20 上都新增答对同一条 `Frying (food)` 样本，因此当前结果证明了 Fixed 路径可稳定接入并保持回退，但尚未证明模型对音频内容具有充分选择性。

后续 Dynamic Window Selector 或 learned gate 不得只优化训练 loss。进入正式接入前至少需要新增以下门禁：

1. 正确音频准确率严格高于错配音频；
2. 正确音频的新增答对中，必须存在错配音频不能复现的增益；
3. 静音继续保持精确回退；
4. 错配音频不得引入空输出、重复文本或 EOS 退化；
5. Dynamic Selector 必须先作为独立旁路评测，不与 Fixed 正式 checkpoint 训练同时改变。

## Dev760 正式选择性门禁

原始 gate `0.25` 的 correct 为 `233/760`，低于 video-only 的 `254/760`，并新增
空输出/EOS，失败。随后只在 reserve326 上校准固定 gate；`0.10`、`0.05`、`0.02`、
`0.01` 均未满足基线约束，`0.005` 在 reserve 上达到 correct 与 baseline 持平，故冻结
该值并只重跑一次 dev760。

| 条件 | 答对 | 相对 video-only gain/harm/net | 空输出/首 EOS |
| --- | ---: | ---: | ---: |
| video-only | 254 | - | 1/1 |
| correct, gate=0.005 | 255 | 10/9/+1 | 1/1 |
| semantic mismatch, gate=0.005 | 259 | 13/8/+5 | 1/1 |
| silence | 254 | 0/0/0 | 1/1 |
| gate=0 | 254 | 0/0/0 | 1/1 |

回退和输出有效性通过，但 correct 未严格高于 mismatch，且 correct 净收益未高于
mismatch，因此总门禁为 **FAIL**。这证明固定推理缩放只能控制退化，不能产生音频内容
选择性。按锁定协议不得继续使用 dev760 调 gate。

下一版仅新增训练期语义配对 ranking loss：同 source/task、不同物理媒体、不同事件标签的
音频作为显式负样本，约束正确音频与答案语义的相似度高于错配音频。Offset、Dynamic 和
learned Gate 继续关闭；必须重新执行 `2-step -> 20-step -> 150-step -> reserve326 -> dev760`。

正式结果：

- `intersuit/harness/artifacts/as_m4_beats_fixed_stage2_dev760/selectivity_report.md`
- `intersuit/harness/artifacts/as_m4_beats_fixed_stage2_dev760_gate0005/selectivity_report.md`

## 代码提交

- `f742766`：保留消融实验的显式环境配置；
- `29699d3`：稳定 Fixed BEATs residual fusion，增加静音屏蔽和 label-term 判分；
- `cf1577c`：分离训练 gate 与推理 gate。
