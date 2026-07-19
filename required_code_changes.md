# 新增数据源所需代码变更

本文件只定义后续实现范围。本次专项预检没有修改正式推理路径、没有生成
正式 train/dev manifest，也没有启动训练。

## 当前支持情况

| 能力 | 当前状态 | 结论 |
|---|---|---|
| AVUT Hugging Face 固定路径下载 | 无批量 allowlist 下载器 | 需要新增 |
| MUSIC YouTube 定时截取 | 本机 yt-dlp/ffmpeg 可用，无项目脚本 | 需要新增 |
| MP4 场景音频读取 | Dataset 可直接从 `scene_audio_path` 解码 | 已支持 |
| 离线 WAV 提取 | 无两数据源专用批处理工具 | 建议新增 |
| AVUT 原始字段转换 | 只有 smoke/frozen eval 转换 | 需要新增 |
| MUSIC 模板与 `anser` 转换 | 未实现 | 需要新增 |
| DataCollator 音频窗口 padding | 已支持 | 通过 |
| 启动器排除门禁 | 只比较冻结 manifest 文件路径 | 不通过 |
| Trainer dev dataset | 固定为 `eval_dataset=None` | 未支持 |

`prepare_ave_hf_pilot.py` 只适用于 AVE；`download_omnimmi_assets.py` 只适用于
OmniMMI；`prepare_audio_visual_event_manifests.py` 对 AVQA 只做统计。它们
不能直接承担本白名单的下载和最终回答字段转换。

## 1. 白名单下载器

新增：

```text
intersuit/scripts/download_new_dataset_allowlist.py
```

必须满足：

1. 输入只能是 `download_allowlist.json` 或 `download_reserve_list.json`；
2. 验证 artifact kind，拒绝普通 JSON 或训练 manifest；
3. 按 `source_dataset` 分派 AVUT/MUSIC 下载逻辑；
4. 默认输出到 `quarantine`，不能直接写入正式媒体目录；
5. 支持 `--limit`，第一轮最多 5 条；
6. 支持断点续传，但临时文件使用 `.part`；
7. 每条保存 stdout、stderr、命令、开始时间和退出码；
8. AVUT 必须验证固定 revision、字节数和远程 LFS SHA256；
9. MUSIC 必须严格使用 CSV 的 start/duration，不允许整段视频混入；
10. 失败条目不能静默替换；只能按 reserve 顺序补位；
11. 禁止访问白名单之外的 URL、repo ID 或 YouTube ID；
12. 默认不启用代理，使用代理时必须显式记录环境。

建议命令：

```bash
python intersuit/scripts/download_new_dataset_allowlist.py \
  --allowlist download_allowlist.json \
  --output-root intersuit/datasets/NEW_AV_TRAIN/quarantine \
  --limit 5 \
  --workers 1
```

## 2. 媒体后处理与验收

新增：

```text
intersuit/scripts/validate_new_dataset_media.py
```

每个物理媒体必须检查：

- 文件存在且非零；
- ffprobe 能解析；
- 视频流存在并可解码；
- 音频流存在并可解码；
- 解码后音频样本非空；
- 音频 RMS/mean-abs 不是全零；
- AVUT 文件大小和 SHA256 等于白名单；
- MUSIC 截取时长与 CSV 目标差不超过明确阈值；
- MUSIC suffix 与派生文件名一致；
- `has_flip=Y` 时原始和 `_flip` 均可解码；
- 下载后 SHA256 不在排除集合；
- 同批媒体 SHA256 不重复。

验收输出应是独立 validation JSONL。只有 validation 状态为 accepted 的媒体
才能从 `quarantine` 移到 `accepted`。

不建议训练时反复从长视频提取音轨。可在验收后额外生成 16 kHz mono WAV，
但 MP4 仍是来源真值，WAV 必须记录：

```text
source_media_sha256
audio_sha256
sample_rate
sample_count
duration
ffmpeg_version
```

## 3. 字段转换器

新增：

```text
intersuit/scripts/convert_new_dataset_training_fields.py
```

第一阶段只能读取：

- AVUT `AV_Human_data.json`；
- MUSIC `train_balance.json`；
- 已通过验收的媒体 validation JSONL。

禁止读取 AVUT Gemini 标注作为第一阶段主标签。

AVUT 转换要求：

- 按 `video_path` stem 得到 `youtube_id`；
- 保留原始 `video_id` 和 `QA_id`；
- 把四个选项写入 prompt；
- 保留答案字母，不擅自改成生成式长答案；
- 官方三类纯音频任务映射为 `audio`；
- Character/Object/OCR Matching 映射为 `audio_visual`。

MUSIC 转换要求：

- `video_id + suffix` 是物理截取段 ID；
- `has_flip=Y` 时允许 base 和 `_flip` 两个派生 ID；
- `youtube_id` 始终保留 CSV 原始 ID；
- 按顺序使用 `templ_values` 替换 `<Object>/<FL>`；
- 上游答案字段读取 `anser`；
- `type[0]` 映射到 audio/visual/audio_visual；
- `type[1]` 保留为 subtype；
- `question_deleted` 非 0 的条目必须删除；
- 任何残留占位符都要硬失败。

转换器第一阶段只输出字段转换候选，不直接输出正式 train/dev manifest。

## 4. 严格泄漏检测

新增：

```text
intersuit/scripts/validate_training_data_leakage.py
```

门禁必须同时检查三个层面。

### video_id

`video_id` 在同一 `source_dataset` 命名空间内比较。AVUT 的整数 video ID
不能与其他数据集的数字 ID 做无意义的全局碰撞判断。

### youtube_id

`youtube_id` 做全局比较：

- candidate vs 冻结 300；
- candidate vs 历史 21；
- candidate vs 全部 dev/test/eval；
- train vs dev；
- AVUT vs MUSIC；
- base ID vs suffix/flip 派生 ID。

### media_sha256

下载后对实际媒体文件计算 SHA256：

- candidate vs 全部排除媒体；
- train vs dev；
- 同批内部；
- AVUT 远程 LFS SHA256 vs 实际下载 SHA256；
- MUSIC base vs flip 记录派生关系，但不能把二者误判为同一文件。

验证器输出必须包含：

```text
status
candidate_count
video_id_overlap_count
youtube_id_overlap_count
media_sha256_overlap_count
missing_media_count
invalid_media_count
exclusion_set_sha256
candidate_set_sha256
```

任一 overlap 非 0 时退出码必须非 0。

## 5. 启动器门禁

修改范围只限训练启动器，不改正式推理路径。

`run_as_m4_beats_stage.sh` 在调用 `run_as_m4_stage.sh` 前必须：

1. 要求 `TRAIN_MANIFEST_AUDIT`；
2. 验证 audit 状态为 PASS；
3. 验证 audit 的 candidate SHA256 等于当前 `DATA_PATH`；
4. 验证冻结/历史/dev/test 排除集合 SHA256 与当前集合一致；
5. 验证 `video_id_overlap_count=0`；
6. 验证 `youtube_id_overlap_count=0`；
7. 验证 `media_sha256_overlap_count=0`；
8. 验证所有媒体状态为 accepted；
9. 验证 train/dev 媒体 ID 和 SHA256 均无交集；
10. audit 过期或任一输入变化时拒绝启动。

建议入口形式：

```bash
DATA_PATH=/absolute/path/to/train.json \
TRAIN_MANIFEST_AUDIT=/absolute/path/to/train.audit.json \
bash intersuit/scripts/run_as_m4_beats_stage.sh 12k-smoke
```

仅比较 `DATA_PATH` 路径的现有门禁必须保留为第一层快速检查，但不能替代内容
门禁。

## 6. Dev 接入

当前 `make_supervised_data_module()` 固定：

```text
eval_dataset=None
```

两种可接受方案：

1. 增加显式 `--eval_data_path`，Trainer 读取独立 dev；
2. 训练阶段仍不跑 eval，但用冻结的独立 harness 在每个关键 checkpoint 后
   评测。

无论选择哪种方案，dev 都不能并入训练 JSON，不能用 train loss 冒充 dev
结果。

## 7. 测试要求

下载前必须补齐：

- allowlist 非法 source/URL 拒绝测试；
- AVUT revision/大小/SHA256 错误测试；
- MUSIC start/duration/suffix/flip 测试；
- `anser` 和模板替换测试；
- `question_deleted` 过滤测试；
- 20 条字段转换 fixture；
- video_id source namespace 测试；
- youtube_id 全局泄漏测试；
- 改名但同 SHA256 泄漏测试；
- train/dev 同 ID 和同 SHA256 测试；
- launcher audit 缺失、过期和失败状态测试；
- quarantine 不能被 Dataset 读取的测试。

在这些测试完成前，不允许执行 550 条批量下载；最多只允许下载器自身的
1～5 条隔离 smoke。
