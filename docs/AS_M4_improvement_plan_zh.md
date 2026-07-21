# AS-M4 音视频流式模型改进思路

本文将“音频事件感知对齐、音频置信度融合、伪标签对齐训练”整理为一条可逐步验证的 AS-M4 改进路线。重点是先建立可归因的增量实验，再做联合训练，避免多个模块同时变化后无法判断收益来源。

## 1. 当前基础与主要缺口

仓库已经具备 AS-M4 的最小工程闭环：

- `audio_stream.py` 和 `av_buffer.py` 负责场景音频读取、切窗以及音视频时间轴管理；
- `event_detector.py`、`dynamic_window_selector.py` 负责音频事件强度和动态窗口；
- `temporal_aligner.py`、`audio_event_aligner.py` 负责局部因果时间对齐和 offset 诊断；
- `confidence_gate.py` 计算质量/相关性门控；
- `fusion.py` 以不增加视频 token 数量的残差方式融合音频。

当前更像“模块可运行 + 诊断可观测”的原型，主要缺口是：

1. 事件强度尚未稳定地转化为对齐窗口、候选 offset 和训练目标；
2. 门控信号仍需要校准，尤其要区分“音频清晰但与问题无关”和“音频相关但质量差”；
3. 对齐模块缺少规模化监督，单靠语言模型 loss 很难学到稳定的时间对应关系；
4. 流式延迟、错位、噪声、多说话人等条件下的性能边界还需要系统评测。

## 2. 推荐总体路线

建议采用以下顺序：

```text
锁定原始 M4 与 Fixed BEATs checkpoint
    -> 完整验证正确/错配/静音/video-only
    -> 证明正确音频具有独有收益
    -> 人工错位与 Oracle 硬窗口
    -> 自动 Offset 硬选择
    -> Dynamic Selector
    -> learned Gate
    -> MUG 式事件伪标签与模态解耦预训练
    -> 小规模联合微调与端到端消融
```

Fixed BEATs 尚未证明正确音频优于错配音频之前，不得把事件强度接入对齐控制，也不得直接训练 Offset、Dynamic 或联合 Gate。对齐错误会直接污染融合输入；先确认音频确实带来可归因收益，再提高时间对应关系，最后让门控学习“是否应该相信音频”，风险和归因成本最低。

### 2.1 Fixed 语义选择性验收

正式进入对齐开发前，必须在锁定的 Fixed checkpoint 上按下述用途完成开发集评测：

```text
video-only / 正确音频 / 错配音频 / 静音 / gate=0
```

至少满足：正确音频的独有新增答对高于错配音频；静音和 `gate=0` 保持回退；无空输出、EOS 退化和重复文本。若正确音频与错配音频收益相同，应先检查 projector、融合位置、时间聚合和音频配对，不得继续叠加新模块。

评测集合用途固定如下，禁止混用：

| 集合 | 用途 |
| --- | --- |
| dev20 | 工程门禁和推理 gate `0.25` 的初步校准集 |
| dev 剩余 760 条 | Fixed 语义选择性正式评测集，不再调整 Fixed gate |
| reserve 326 条 | 后续 Gate、阈值、residual scale 和其他超参数校准集 |
| 冻结 300 条 | 阶段性最终验收集，禁止据此调参 |

## 3. 改进一：音频事件感知的流式对齐

### 3.1 核心思路

在每个流式时间步维护音频窗口和视频帧片段。必须显式区分运行模式：

- 严格实时：`V[t-r:t] + A[t-r:t]`，`lookahead_sec=0.0`；
- 准实时：`V[t-r:t+0.5] + A[t-r:t+0.5]`，`lookahead_sec=0.5`。

公式、日志和延迟指标不得混用两种模式。

```text
严格实时：
视频窗口 V[t-r:t] + 音频窗口 A[t-r:t]

准实时：
视频窗口 V[t-r:t+L] + 音频窗口 A[t-r:t+L]
其中 L = 0.5 秒

    -> 音频事件强度 e_t
    -> 局部候选 offset 分数 s(t, delta)
    -> 因果平滑后的 offset delta_t
    -> 对齐后的音频特征 A(t + delta_t)
```

音频事件强度不应只表示音量，且必须拆分以下信号：

- 短时能量/非静音比例：检测说话、敲击、爆音等有效活动；
- 音频特征变化率：检测事件起止和突发变化；
- 音视频同步证据：音频变化峰与视频运动/视觉事件峰的局部一致性；
- `event_saliency`：事件是否显著；
- `event_duration`：事件持续时间；
- `event_boundary`：是否接近事件起止点；
- `audio_quality`：声音是否可靠。

短促且边界明确的事件可使用短窗口；持续且语义稳定的事件可使用长窗口；低质量或静音段降低更新频率或回退。`event_strength` 只控制时间关注和更新速度，不等同于音频可信度；高强度但与问题或画面无关的背景声不能因此提高融合权重。

### 3.2 建议的对齐机制

第一版保持因果和轻量，只搜索有限 offset 集合，例如 `{-0.5, 0, +0.5}` 秒，并加入：

```text
s(t, delta) = semantic_similarity
            + lambda_event * event_sync_score
            + lambda_change * change_peak_score
            - lambda_smooth * |delta - delta_(t-1)|
```

`+0.5` 秒候选若需要访问未来音频或视频，只能在 `lookahead_sec=0.5` 的准实时模式启用。严格实时模式必须将其标记为不可用，或通过延迟输出获得所需上下文；不得以零延迟名义读取未来输入。

第一版不要直接对多个 offset 做 softmax 混合。先进行人工错位实验和 Oracle 硬窗口，再做自动硬选择：`margin >= threshold` 时使用预测 offset，低置信度时回退到 `0` 秒或上一稳定 offset。只有硬选择改善最终 QA 后，才比较 soft weighting。

建议新增或明确记录以下诊断量：`event_strength`、`candidate_scores`、`best_offset`、`offset_confidence`、`offset_jump`、`alignment_entropy`。这些量必须按窗口写入 JSONL，支持按事件类型和延迟条件分组统计。

### 3.3 训练目标

在原有语言模型 loss 之外，先加入小权重辅助目标：

- `L_offset`：伪标签或合成错位条件下的 offset 分类/回归损失；
- `L_sync`：事件窗口内音视频同步分数高于错配窗口的 ranking 或 InfoNCE 损失；
- `L_smooth`：约束相邻时间步 offset 不无故跳变。

总损失可写为：

```text
L = L_lm + alpha * L_offset + beta * L_sync + gamma * L_smooth
```

其中 `alpha/beta/gamma` 从很小值开始，先确认不会破坏原 M4 行为。

## 4. 改进二：音频置信度与问题相关性感知融合

### 4.1 门控信号拆分

门控不应只输出一个不可解释的权重，建议显式拆成：

```text
q_quality = 音频质量
r_av      = 音频与画面相关性
r_query   = 音频与当前问题相关性
c_align   = 时间同步置信度
```

其中：

- `q_quality`：信噪比、静音比例、削波、重叠程度、语音可懂度等；
- `r_av`：音频事件与画面事件的相关性；
- `r_query`：问题、音频、视频和任务/模态类型共同决定的相关性，不应只用简单文本相似度；
- `c_align`：对齐 margin、offset 稳定性、alignment entropy；
- `event_strength`：用于避免把纯静音段误判为高价值音频。

多人说话不能统一视为低质量，应分别记录 `speech_overlap_score`、`speaker_count_estimate` 和 `speech_intelligibility`，再结合问题类型决定是否降权。当前正式路径没有场景 ASR，暂不把 ASR confidence、WER proxy 或 transcript consistency 放入 Gate；必须与用户问题语音 `speech` 和视频环境音频 `scene_audio` 严格区分。

可解释规则基线：

```text
g_rule = q_quality * r_av * r_query * c_align
```

Learned Gate：

```text
g_learned = sigmoid(MLP(
    q_quality,
    r_av,
    r_query,
    c_align,
    event_strength,
    residual_norm,
    offset_margin
))
```

`event_strength` 只作为 Dynamic Selector 和 learned Gate 的辅助特征，不直接乘入规则基线。

最终融合继续采用残差形式：

```text
Z_fused = Z_video + g * s * Projector(Z_audio_aligned)
```

其中 `s` 是受控的 residual scale。建议保留同一 AS-M4 checkpoint 内 `gate=0` 关闭音频残差的能力，并继续记录 `delta_to_video_ratio`；这不自动等价于恢复原始 M4 权重。

### 4.2 门控训练方式

门控监督可以来自三类信号：

1. 规则质量标签：静音、强噪声、削波、明显错位；
2. 音频-问题相关性标签：问题是否必须依赖场景声音才能回答；
3. 任务收益标签：加入音频后答案是否变好，只能在独立开发集、固定教师和高置信 QA 标签上做配对校准，不能把当前模型自身判断直接作为第一阶段监督，避免形成循环依赖。

第一阶段建议训练质量头和相关性头，第二阶段再训练联合 gate。门控的目标不是“音频越大越好”，而是让音频在可靠且有用时影响模型，在不可靠或无关时接近 video-only。

### 4.3 必做的融合消融

至少比较以下条件：

- video-only；
- neutral-video + audio（仅用于诊断音频编码器能力；当前 residual 架构不直接定义 audio-only）；
- 无 gate 的直接融合；
- 仅质量 gate；
- 质量 + 问题相关性 gate；
- 质量 + 相关性 + 对齐置信度 gate；
- 相同配置下 `gate=0` 行为回退。

评测必须覆盖 clean、加噪、静音、语音不清晰、重叠语音、语音片段丢失、音频语义扰动、音频延迟和跨样本错配，而不能只在同步干净音频上比较。只有未来正式加入 `scene_audio -> ASR -> transcript + confidence` 路径后，才增加 transcript corruption、ASR confidence、WER proxy 和 transcript consistency 条件。

## 5. 改进三：参考 MUG 的伪标签增强训练

参考论文：Langyu Wang 等，`MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing`，ICCV。这里借鉴的是事件伪标签和模态解耦增强，不应称作“MUG 伪标签对齐预训练”。

MUG 的核心不是直接为每个样本回归连续时间 offset，而是针对弱监督音视频解析构造“分段级伪标签 + 跨模态随机组合”数据。迁移到 AS-M4 时，应保留这一核心，再把伪标签用于事件对齐和门控训练。

### 5.1 MUG 的具体机制

MUG 的流程可以概括为：

1. 使用冻结的 CLIP 和 CLAP，将预定义事件文本编码后与每个视频/音频分段计算相似度，阈值化得到视觉伪标签 `y_hat_v` 和音频伪标签 `y_hat_a`；
2. 将明显异常的空视觉标签等伪标签人工检查并修正；无法可靠归入类别集合的样本不参与组合；
3. 从视频甲取视觉轨迹、从视频乙取音频轨迹，做 Cross-Modality Random Combination（CMRC）；新样本的标签由两条单模态伪标签合并得到；
4. 将伪标签对应的类别转换为自然语言概念，再通过 CLIP/CLAP 文本编码器生成语义特征，使用 PLSIM 对音频/视觉特征进行缩放和偏置调制，以抑制无关模态噪声。

因此，MUG 提供的是“分段级事件监督和模态解耦数据增强”，并不等价于已有 AS-M4 的 offset 标注器。

### 5.2 迁移到 AS-M4 的伪标签设计

针对 AS-M4，可在 MUG 的分段伪标签之上增加以下软标签：

- 事件强度和事件边界：由 CLAP/CLIP 相似度曲线、能量变化和视觉运动变化联合得到；
- 对齐候选标签：保留原始同步片段作为正样本，使用时间平移片段作为 hard negative，训练候选 offset 排序。这是 AS-M4 基于 MUG 思想的扩展，不是 MUG 原论文目标；
- 模态相关性标签：根据音频伪标签与视频伪标签是否一致，估计互补、无关或冲突关系；
- 伪标签置信度：由相似度 margin、教师模型一致性和增强前后稳定性共同计算。

这里的 offset 监督是 AS-M4 的扩展，不应声称是 MUG 论文原始的直接目标。

### 5.3 CMRC 在 AS-M4 中的安全用法

CMRC 可以帮助模型学习“视频和音频不一定来自同一事件”，但不能直接把随机组合音频当作真实同步音频。建议：

- 将 CMRC 样本用于训练事件分类、相关性 gate 和抗错配能力；
- 将原始同步样本用于 temporal aligner 的正样本和语言模型任务；
- 对跨视频组合使用明确的 `mismatch` 标签或低相关性标签，避免模型把错配样本学成正常融合；
- CMRC 标签不能直接当作 QA 答案：纯视觉题可保留原答案，纯音频题必须按新音频重建答案，音视频联合题通常丢弃，除非重新生成并人工验证；
- 控制增强数据规模，逐步测试不同组合比例；MUG 的结果显示，组合数据过多会重新引入伪标签噪声和过拟合。

### 5.4 伪标签语义交互

可以参考 MUG 的 PLSIM，但将其改造成 AS-M4 的轻量语义门控：

```text
audio_modulated = audio_feature * scale_audio(event_text) + bias_audio(event_text)
video_modulated = video_feature * scale_video(event_text) + bias_video(event_text)
```

其中 `event_text` 来自预定义事件文本集合中经 CLIP/CLAP 相似度筛选出的分段伪标签，不是模型开放式生成的事件名称。该模块的作用是给 `confidence_gate` 提供语义先验，不能替代问题相关性判断；最终 gate 仍应同时考虑 `q_quality`、`r_av`、`r_query` 和 `c_align`。

### 5.5 教师-学生和稳定性筛选

建议采用冻结教师、可训练学生的方式：

- 教师：冻结 CLIP/CLAP、音频事件模型、视觉事件模型以及现有对齐诊断器；
- 学生：`streaming_av_module` 中的 event detector、temporal aligner、confidence gate；
- 对音频施加增益、噪声、混响和时间抖动，对视频施加帧丢弃和轻微速度变化；
- 只有在不同增强下类别伪标签稳定、相似度 margin 足够大时，才提高监督权重；
- 对连续 offset 使用 ranking/分类损失，而不是把低置信度教师的 argmax 当作精确时间真值。

### 5.6 数据配比

建议分三阶段：

1. MUG 式事件伪标签与模态解耦预训练：冻结 Qwen2、视觉塔和大部分 projector，直接训练 `event detector`、`AV match head`、`relevance head`、`confidence gate` 和模态解耦模块；
2. 真实任务微调：加入 AVQA、AVE、LLP、MUSIC-AVQA 或带原始音轨的 M4 扩展数据；
3. 小学习率联合微调：解冻必要的 projector/融合层，保留原 M4 12k 和 32k no-freeze 基线做独立对照。

`offset scorer`、`synchronizability head` 和 `temporal aligner` 属于 AS-M4 扩展，不应表述为 MUG 直接支持。Offset 监督主要来自原始同步样本、合成 `+/-0.5/+/-1.0` 秒偏移和高置信同步教师；CMRC 跨视频错配主要用于训练 `av_match=0` 或低 Gate，不作为精确 offset 类别。伪标签数据不能替代真实问答数据，语言模型 loss 仍负责回答能力。

## 6. 推荐实验矩阵与验收标准

每个阶段都保留原 M4、video-only 和 `gate=0` 回退结果。推荐最小矩阵：

| 实验 | 音频条件 | Offset | Dynamic | Gate | MUG | 目的 |
| --- | --- | --- | --- | --- | --- | --- |
| E0 | 关闭 | 否 | 否 | 否 | 否 | 原始 M4 |
| E1 | 正确 | 否 | 否 | 固定 gate | 否 | Fixed BEATs 基线与独有收益 |
| E2 | 错配 | 否 | 否 | 固定 gate | 否 | Fixed 选择性 |
| E3a | 人工错位 | 不修正 | 否 | 固定 gate | 否 | 证明错位是否损害 QA |
| E3b | 人工错位 | Oracle 修正 | 否 | 固定 gate | 否 | 测试窗口修正理论上限 |
| E3c | 人工错位 | 自动硬修正 | 否 | 固定 gate | 否 | 测试 Offset Scorer 实际收益 |
| E3d | 正确同步 | 自动硬修正 | 否 | 固定 gate | 否 | 检查 0 秒样本误修正 |
| E4a | 统一对齐评测集 | Oracle Offset | 关闭 | 固定 gate | 否 | Fixed + Oracle Offset |
| E4b | 统一对齐评测集 | 自动 Offset | 关闭 | 固定 gate | 否 | Fixed + Auto Offset |
| E4c | 统一对齐评测集 | 自动 Offset | 开启 | 固定 gate | 否 | Dynamic + Auto Offset |
| E5 | 正确/噪声/错配 | 自动 Offset | Dynamic | learned | 否 | Gate 收益与抗扰动 |
| E6 | 多条件 | 自动 Offset | Dynamic | learned | 是 | Full AS-M4 |
| E7 | 任意 | 任意 | 任意 | 强制 0 | 任意 | 同一 AS-M4 checkpoint 下的音频路径消融 |

建议同时报告：任务准确率、音视频 offset MAE、同步检索 Recall@1、门控校准误差、平均 gate、`delta_to_video_ratio`、流式额外延迟和峰值显存。

上线或扩大训练前至少满足：

- 同一 AS-M4 checkpoint 下 `gate=0` 与该 checkpoint 的 video-only 音频路径消融保持精确一致；与原始 M4 的逐值一致性另行通过 E0 checkpoint 验证；
- clean 音频不低于 video-only，噪声/错配条件下不能显著低于 video-only；
- offset MAE 和同步检索指标相对无对齐版本改善；
- 门控在静音、噪声和无关音频上明显下降；
- 32k 主实验仍沿用现有 no-freeze 基线和 OOM 降级顺序，不覆盖基线 checkpoint。

Offset 验收需单独报告：正确同步、Oracle 修正、自动修正和错位不修正的 QA，以及 0 秒样本误修正率。最低要求是 Oracle 修正优于错位不修正，自动修正也优于错位不修正。Dynamic 只有在相同 Offset 条件下准确率更高，或准确率不下降且音频 token、延迟或计算量下降时才算通过。

当前 residual 架构没有自然的 audio-only 定义，优先评测 `video-only / 正确音频 / 静音 / 错配音频 / gate=0`。如需 audio-only，应另行定义 neutral-video 承载位置或增加独立 audio token 路径。

E7 只证明同一 AS-M4 checkpoint 内 `gate=0` 关闭该 checkpoint 的音频残差，不证明它等于原始 M4 checkpoint。若联合训练更新了 M4 主体、视觉塔或 projector，精确权重回退必须显式加载保留的 12k/32k baseline checkpoint；只有全程冻结这些原视频路径参数时，才可能保持逐值回退。

## 7. 实施优先级

近期最值得先做的是：

1. 先完成 Fixed checkpoint 在 dev 剩余 760 条上的正确/错配/静音/video-only 选择性验收；
2. 完成人工错位和 Oracle 硬窗口实验，再接自动 Offset；
3. 把 `event_saliency`、`event_duration`、`event_boundary`、`offset_confidence`、`alignment_entropy` 接入对齐日志；
4. 将 Gate 明确拆成质量、音画相关性、问题相关性、对齐置信度四路，并完成合成扰动校准；
5. 构造时间平移/跨视频错配伪标签集，分别训练 MUG 直接支持的事件/匹配模块和 AS-M4 扩展的对齐模块；
6. 在 12k no-freeze 基线上完成重排后的 E0-E7 小规模归因；
7. 只有当回退、选择性和抗扰动门禁通过后，才在 32k no-freeze 基线上做完整联合训练。

不建议优先通过冻结 `lm_head`、删除场景音频、缩短上下文或削减视频 token 来获得表面上的训练稳定性。这些做法会掩盖对齐和门控问题，也会损害最终回答能力。
