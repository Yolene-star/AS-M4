# 训练数据与代码适配性审查

审查日期：2026-07-19（Asia/Shanghai）

代码提交：`e9341ac26f41b1c37a158f807245163a8d2937a9`

本次审查只读取本地数据、标注、媒体、脚本和训练代码。没有联网下载数据，
没有构建 train/dev manifest，没有启动训练，也没有修改正式推理路径。

## 结论

**FAIL**

当前数据不能直接、严格地构建“至少 1000 条独立 train 和约 300 条独立
dev”的 BEATs audio projector 最终回答训练集。主要阻断项：

1. AVUT 本地 243 个媒体 ID 全部属于冻结 dev、冻结 300 条或历史 21 条，
   独立可训练 ID 为 0。
2. 本地 `datasets/AVQA` 为空；MUSIC-AVQA 和 AVHBench 均没有本地数据、
   可用 manifest 或已实现的训练转换脚本。
3. `AVE_HF_EXPANDED` 有 1000 个媒体 ID，但与冻结 300 条重叠 100 个，
   与现有 dev manifest 重叠 60 个，两者有 13 个交集。按 ID 合并排除后
   只剩 853 个；再排除 1 个音视频时长差超过 0.1 秒的样本后，只剩
   **852 个严格候选 ID**。
4. 这 852 条只有视频、音频和事件标签，没有现成的
   `question`、`answer`、`task_type` 和 `conversations`。可以生成音频题和
   视觉题，但没有经过验证的真实音视频联合题。
5. BEATs 启动器目前只拒绝冻结 300 条 manifest 的同一路径，不能按
   `youtube_id` 或媒体 SHA256 阻断内容泄漏；训练数据模块也没有挂接
   dev dataset。

因此本轮禁止构建正式 train/dev manifest，禁止启动训练。

## 数据盘点

| 数据源 | 本地状态 | 标注/媒体规模 | 独立训练可用性 |
|---|---|---:|---|
| AVUT 原始人工标注 | 存在 | 1734 QA，691 个视频 ID | 标注可转换 |
| AVUT 本地媒体 | 存在 | 243 个视频，647 QA 可落到本地媒体 | 0 个独立 ID |
| AVUT 冻结 dev | 存在 | 222 个视频 ID | 全部排除 |
| AVUT 历史样本 | 存在 | 21 个视频 ID | 全部排除 |
| AVUT 冻结 300 条中的部分 | 存在 | 200 个 AVUT 视频 ID | 全部排除 |
| AVUT 外部网页示例 | 存在 | 12 个示例视频 | 没有兼容训练 QA manifest |
| AVQA / MUSIC-AVQA | 本地目录为空 | 缺少 `train_qa.json`、`val_qa.json` 和媒体 | 0 |
| AVHBench | 未发现 | 无目录、manifest 或下载/转换脚本 | 0 |
| AVE_HF_300 | 存在 | 300 个视频与独立 WAV | 已用于开发，排除 |
| AVE_HF_EXPANDED | 存在 | 1000 个视频与独立 WAV | 严格排除后 852 |
| AVE_HF_SELECTIVE_1200 | 存在 | 1200 个视频与独立 WAV | 591 dev + 609 test，全部排除 |
| LLP 统一事件 manifest | 存在 | 有窗口事件标注 | 当前 `video_path=null`，不能直接训练 |

MUSIC-AVQA 只在设计文档中出现；设计中的 `convert_music_avqa.py` 和
`convert_avqa.py` 尚未落地。现有
`prepare_audio_visual_event_manifests.py` 对 AVQA 只做 Sound/View/Both
统计，不生成最终回答训练 manifest。

## 字段适配

目标字段：

```text
video_path
scene_audio_path
question
answer
video_id/youtube_id
task_type
```

| 数据源 | video_path | scene_audio_path | question/answer | ID | task_type | conversations |
|---|---:|---:|---:|---:|---:|---:|
| AVUT 原始标注 | 有 | 无 | 有 | 有 | 有 | 无 |
| AVE_HF_EXPANDED | 1000/1000 | 0；有 `audio_path` | 0 | 1000/1000 | 0 | 0 |
| 冻结 300 条 | 300/300 | 200/300 | 300/300 | 300/300 | 300/300 | 300/300 |

AVE 候选需要离线、确定性转换：

1. `audio_path -> scene_audio_path`；
2. 使用 `datasets/AVE/data/Annotations.txt` 将全部 1000 个 ID 映射到
   28 个语义事件名称；当前映射覆盖率为 1000/1000；
3. 生成 `question`、`answer`、`task_type`；
4. 生成 M4 训练实际读取的 `conversations`；
5. 按 `youtube_id` 划分，禁止同一媒体跨 train/dev；
6. 保留 `source_dataset`、`original_split` 和转换规则版本。

直接可用且通过排除规则的原生最终回答题数量为：

| 类别 | 原生可用 | 可模板转换但尚未批准 |
|---|---:|---:|
| 音频必要 | 0 | 852 |
| 视频必要 | 0 | 852 |
| 音视频联合 | 0 | 0 |

两种模板可产生 1704 条 QA，但只来自 852 个独立媒体 ID。不能把同一媒体的
两种题型当成两个独立媒体来满足 1000 train + 300 dev 的严格容量门槛。

AVUT 本地 647 条 QA 中，音频必要题 227 条、音视频联合题 420 条、视觉题
0 条；这些 QA 对应的媒体全部属于排除集合，不能用于训练。

## 重叠审查

冻结与候选锁定值：

- 冻结 300 条 SHA256：
  `ca344ceaec00c30790f1c7179f72a90a3174aae1359c95db18eafe799d62932c`
- SELECTIVE1200 dev SHA256：
  `92cbc03c54bb3965f6682dd15e35f6d88122ac70ccccdbe3a23e9762bdaace03`
- SELECTIVE1200 test SHA256：
  `89b32906a5a700d33a45908628657b8632a0102fe4dc423a60b6528a149656df`
- EXPANDED 候选 SHA256：
  `7c5a67e34c6dbd53892cb3a66f0fa3886dac77dec3d17986391ec30c7509aa1e`
- AVUT 人工标注 SHA256：
  `b952c0893df409cbe716476d0a0b12cb97fd47ba0e75a6898cb99c81f29182e9`

ID 检查结果：

| 检查项 | 重叠数量 |
|---|---:|
| EXPANDED vs 冻结 300 条 | 100 |
| EXPANDED vs 所有现有 dev manifest | 60 |
| 上述两组重复排除 ID | 13 |
| EXPANDED 合并排除 | 147 |
| EXPANDED vs SELECTIVE1200 dev/test | 0 |
| EXPANDED vs AVUT 历史 21 条 | 0 |
| AVUT 本地 vs 冻结 300 条 | 200 |
| AVUT 本地 vs 历史 21 条 | 21 |
| AVUT 本地 vs AVUT 冻结 dev | 222 |

媒体 SHA256 检查覆盖 2000 个 EXPANDED 候选文件路径、2743 个排除文件路径，
合计读取 4,646,930,624 字节：

- EXPANDED 与排除集合完全相同的文件哈希：100 个，均为冻结评测视频；
- EXPANDED 内部重复文件哈希：0；
- 哈希重叠和 ID 重叠一致，没有发现只靠改名隐藏的额外完全相同文件。

## 媒体检查

全量已有验证记录与本轮路径检查：

| 数据源 | 样本对 | 缺失/零字节 | 空音频 | 时长差 > 0.1 秒 |
|---|---:|---:|---:|---:|
| AVE_HF_300 | 300 | 0 | 0 | 0 |
| AVE_HF_EXPANDED | 1000 | 0 | 0 | 1 |
| AVE_HF_SELECTIVE_1200 | 1200 | 0 | 0 | 4 |

EXPANDED 时长异常候选：

```text
youtube_id=0-I1-DOC-r8
video_duration=10.000000
audio_duration=9.729188
abs_diff=0.270812
```

该样本可解码且音频非空，但按本审查的 0.1 秒严格阈值排除。

AVUT 本地 243 个 MP4 已逐个执行 ffprobe：243/243 同时包含有效视频流和
音频流，损坏或缺流数量为 0。

本轮另外按事件标签和来源确定性抽查 40 条：

- 30 条 EXPANDED 独立候选；
- 5 条 AVUT 冻结 dev；
- 5 条 SELECTIVE1200 冻结 dev。

40/40 视频完整解码成功，40/40 音频解码成功，40/40 音频非空；
抽查中的音视频时长差最大为 0.031 秒。候选池已知损坏媒体为 0，
严格时长不匹配为 1。

## 代码适配

### Dataset：部分通过

当前 `LazySupervisedDataset`：

- 可以读取 JSON list；
- 可以读取 `video_path`；
- 可以读取 `scene_audio_path`，解码并生成固定窗口和时间戳；
- 不能直接识别 AVE 的 `audio_path` 别名；
- 训练文本实际依赖 `conversations`，不能只提供顶层
  `question` 和 `answer`。

因此数据源不能原样训练，但无需修改正式推理路径；只需新增离线转换器。

### DataCollator：通过

当前 `DataCollatorForSupervisedDataset` 可以：

- padding 不同长度的 scene-audio 窗口；
- 生成 `scene_audio_mask`；
- padding `scene_audio_timestamps`；
- 将这些字段传入模型。

相关 Dataset、scene encoder 和融合回归为 `30 passed`。

### 训练启动器：不通过严格泄漏门禁

`run_as_m4_beats_stage.sh` 已正确限制只训练：

```text
scene_audio_projector,streaming_av_fusion
```

并关闭动态对齐、learned Gate 和事件辅助路径。但当前门禁只比较
`DATA_PATH` 是否等于冻结 300 条 manifest 的路径，不能检查：

- manifest 内部 `youtube_id/video_id`；
- `video_path/scene_audio_path` 文件 SHA256；
- 历史 21 条；
- SELECTIVE1200 dev/test；
- 其他冻结 dev manifest。

此外 `make_supervised_data_module()` 当前固定 `eval_dataset=None`，不能在
训练过程中读取独立 dev manifest。正式训练前必须先补离线内容审计门禁；
是否把 dev 接入 Trainer 可另行决定，不能用 train loss 代替 dev 结果。

## 容量判断

严格候选媒体 ID：

```text
EXPANDED 原始                    1000
- 冻结/现有 dev ID 合并排除       147
- 时长差 > 0.1 秒                  1
= 严格候选                         852
```

若要求 train 和 dev 在媒体 ID 级别独立，则目标至少需要约 1300 个 ID，
当前短缺 **448 个有效独立媒体 ID**。因此容量门槛不通过。

## 推荐数据组合

优先组合：

1. 保留 852 个严格 EXPANDED ID，转换为音频事件题和视觉事件题；
2. 从 AVUT 尚未落地的 448 个官方视频 ID 中选择并下载有效媒体。对应现有
   人工标注 1087 条，其中音频必要题 387 条、音视频联合题 700 条；
3. 若 AVUT 下载或媒体校验有失败，再从 MUSIC-AVQA 官方 train split 补足，
   保留其 Sound/View/Both 原始题型；
4. AVHBench 先作为鲁棒性评测候选，不默认混入 projector 训练；
5. 所有来源先按媒体 ID 合并去重，再做 train/dev 分层划分。

预计新增下载量：

- 数量硬下限：448 个有效独立媒体；
- 建议准备：500 个媒体的下载/校验余量；
- 若补充同规格 10 秒 AVE 媒体，本地均值约 1.74 MB/视频+WAV，
  500 条约 0.87 GB，建议预留至少 1.2 GB；
- 若下载剩余 448 个 AVUT 官方视频，按官方全量约 24.0 GB 和 691 个
  视频粗略比例估算约 15.6 GB，建议预留至少 20 GB。实际下载前必须先用
  官方文件清单计算精确路径和大小；
- MUSIC-AVQA 本地没有文件清单，不能离线给出可信精确字节数。应先只拉取
  官方元数据/manifest 做许可、split、ID 和大小审查，再批准媒体下载。

## 下一步命令

当前结论为 FAIL，下面的训练命令**不得执行**。下一步应先实现两个离线工具：

```text
intersuit/scripts/convert_beats_projector_training_data.py
intersuit/scripts/validate_beats_training_manifest.py
```

转换器负责字段映射、问题模板、语义标签和媒体级划分；验证器必须接受冻结
300 条、历史 21 条、所有 dev/test manifest 作为排除输入，并同时检查 ID、
文件 SHA256、媒体解码、音频非空和时长差。

工具完成后的预期审查命令形式：

```bash
python intersuit/scripts/validate_beats_training_manifest.py \
  --candidate /path/to/candidate.json \
  --frozen intersuit/harness/artifacts/frozen_av_task_eval_dev300/frozen_eval.json \
  --exclude intersuit/harness/artifacts/ave_hf_selective_1200_split/dev_manifest.jsonl \
  --exclude intersuit/harness/artifacts/ave_hf_selective_1200_split/test_manifest.jsonl \
  --require-train 1000 \
  --require-dev 300 \
  --group-key youtube_id \
  --check-media-sha256
```

只有验证器返回 PASS、媒体 ID 级 train/dev 无交集、且所有排除集合命中数为
0 后，才允许生成正式 manifest，随后才可运行：

```bash
DATA_PATH=/absolute/path/to/passed_train.json \
bash intersuit/scripts/run_as_m4_beats_stage.sh 12k-smoke
```
