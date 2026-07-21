# 新增数据源专项预检

日期：2026-07-19（Asia/Shanghai）

代码基线：`305dd6f`

## 结论

**PASS**

本结论只授权后续按 `download_allowlist.json` 下载到隔离目录，不代表媒体已经
验收，也不允许生成正式 train/dev manifest 或启动训练。

通过项：

- 主下载白名单包含 550 个物理候选媒体；
- AVUT 430 个，MUSIC-AVQA-v2.0 120 个；
- MUSIC-AVQA-v2.0 另有 200 个确定性替补候选；
- 主白名单 `video_id/youtube_id` 与全部排除集合重叠为 0；
- AVUT 远程 LFS SHA256 与本地媒体哈希排除集合重叠为 0；
- 20 条字段转换干跑全部通过；
- 预计 QA 中 audio、visual、audio_visual 均非 0；
- 没有无法解决的字段或媒体映射问题。

MUSIC 的 YouTube 来源不提供下载后文件 SHA256，因此只能先进入
`quarantine`。下载后必须计算实际媒体 SHA256，与排除集合逐文件比较；
只有零命中、媒体校验通过的文件才能转入 `accepted`。这不是可选步骤。

本次只下载了约 12.4 MB 的标注和仓库元数据用于预检，没有下载视频。

输出锁定值：

- `download_allowlist.json`：
  `bb75475528ca87aec1eac40f7ce3172365b078bc505f9363ae042d8a285ad37f`
- `download_reserve_list.json`：
  `6e7c83f8fc7321df75da56612e3ca2e7a929722ef2ba7c3630810bdf65dfea83`
- `expected_task_distribution.json`：
  `58931d24e795c52ee8a072d43904da35f1dfbe2f011d344a3dba2544c8d6326b`

## 固定来源

### AVUT

- Hugging Face：`tsinghua-ee/AVUTBenchmark`
- revision：`558ca95ac16c972a4de2028fbecef2a8b82fdb00`
- GitHub：`lark-png/AVUT`
- GitHub 本地镜像提交：`030f52f1eb7c87ff38a36f737d272d73ce9e2400`
- 第一阶段标注：`AV_Human_data.json`
- 人工标注 SHA256：
  `b952c0893df409cbe716476d0a0b12cb97fd47ba0e75a6898cb99c81f29182e9`
- `AV_Gemini_data.json` 和 `AV_Gemini_filtered_data.json` 不进入第一阶段
  主训练标签。

远程人工标注与本地
`intersuit/datasets/AVUT/raw/AV_Human_data.json` SHA256 完全一致。
GitHub 仓库只提供数据入口和 12 个网页演示视频，不提供独立的完整训练
标注；正式媒体路径以固定 revision 的 Hugging Face 文件树为准。

### MUSIC-AVQA-v2.0

- Hugging Face：`DraculaDragon/MUSIC-AVQA-v2.0`
- revision：`76613c9f7f840524c11dd9465018533f9cb1d18d`
- license metadata：`GPL-3.0`
- 视频 CSV：
  `MUSIC-AVQA-v2.0_additional_videos.csv`
- CSV SHA256：
  `4672f5994308d32c2a2942662be90803176195b96f6176f3d890566f77b9e508`
- QA：
  `train_balance.json`
- QA SHA256：
  `34218d7aa2f60ff4aa1ece5f968c1e2a7c8e4c5e9db18ac6273ab7b5a3e9ef00`

仓库只保存 CSV 和 QA JSON，不保存逐视频媒体。视频需要按 CSV 中的
YouTube ID、起点和时长截取；官方另提供完整预处理压缩包，但本轮不下载。

## 排除集合

排除集合读取范围：

1. 冻结 300 条；
2. AVUT 历史 21 条；
3. 本地全部 dev/test/val/eval 候选 JSON/JSONL；
4. 本地现有相关媒体 ID；
5. 本地现有相关媒体 SHA256；
6. OmniMMI 现有媒体只扫描文件 ID，不重新哈希约 181 GB 媒体。

锁定结果：

| 项目 | 数量/值 |
|---|---:|
| 冻结样本 | 300 |
| 历史 AVUT ID | 21 |
| 被扫描的 dev/test/eval manifest | 57 |
| ID 排除集合 | 5375 |
| ID 集合 SHA256 | `3ca782a0b42875cca199e378b3940966d6383a2da9aee6888ccf95588473a262` |
| 用于 ID 扫描的现有媒体文件 | 7814 |
| 被实际哈希的相关媒体路径 | 5495 |
| 实际读取媒体字节 | 5,467,668,016 |
| 不同媒体 SHA256 | 5255 |
| SHA256 集合摘要 | `31296be8437132cad27433a4dc31f39135c02223058ddc6a1d61dedb5d6c971a` |

## AVUT 枚举

`AV_Human_data.json`：

- 1734 条人工 QA；
- 691 个唯一 YouTube 媒体 ID；
- 691/691 均可映射到 Hugging Face MP4；
- 其中 1 个 ID 在远程文件树中有两个路径，但该 ID 已本地落地并处于排除
  集合，不影响新增候选；
- 本地已有 243 个 AVUT 媒体 ID；
- 初始未落地媒体为 448 个。

对 448 个初始候选继续执行全部 dev/eval 和媒体哈希排除：

- 12 个 ID 出现在冻结 dev300 的候选/备用集合中；
- 6 个远程 MP4 的 LFS SHA256 与 GitHub 网页演示视频完全相同；
- 最终合法 AVUT 候选为 **430 个**；
- 合计远程大小为 **6,274,058,408 字节**；
- 最终候选 ID 重叠为 0；
- 最终候选远程 SHA256 重叠为 0；
- 媒体路径映射错误为 0。

430 个 AVUT 候选对应 1042 条人工 QA：

| 类别 | QA |
|---|---:|
| audio | 370 |
| audio_visual | 672 |
| visual | 0 |

白名单直接保存固定 revision、远程相对路径、下载 URL、字节数和远程 LFS
SHA256。下载后必须同时校验大小和 SHA256。

## MUSIC 枚举

视频 CSV：

- 1040 个物理截取段；
- 1035 个唯一 YouTube ID；
- 1035 个无 suffix 段；
- 5 个额外 suffix 段：`_#02` 到 `_#05`；
- 190 个段标记 `has_flip=Y`；
- 1037 个段目标时长 60 秒，另外 3 个分别为 54、26、46 秒。

`train_balance.json`：

- 42,775 条 QA；
- 字段为 `video_id/type/question_content/templ_values/anser`；
- `anser` 是上游真实字段名，不是本地拼写错误；
- `type` 的第一维可稳定映射为 Audio、Visual、Audio-Visual；
- 问题模板使用 `<Object>` 或 `<FL>`，可按 `templ_values` 顺序替换；
- 1000 个物理 CSV 段可映射到至少一条有效 QA；
- 664 个物理段同时覆盖 audio、visual、audio_visual；
- 40 个 CSV 段没有可用 train QA，不进入候选；
- 所有候选模板均可完整替换，无残留占位符。

确定性选择规则：

```text
seed=20260719
必须同时覆盖 audio、visual、audio_visual
必须通过 video_id 和 youtube_id 排除
按 SHA256(seed:candidate_id) 排序
前 120 个进入主白名单
后 200 个进入替补列表
```

MUSIC 主候选 120 个物理段预计产生 766 条 QA：

| 类别 | QA |
|---|---:|
| audio | 223 |
| visual | 192 |
| audio_visual | 351 |

部分 `has_flip=Y` 候选会由一个下载源生成原图和水平翻转两个派生媒体。
主白名单合计 550 个物理下载候选，预计生成 574 个训练前媒体实例。

## 总任务分布

主白名单对应的预计 QA：

| 类别 | AVUT | MUSIC | 总计 |
|---|---:|---:|---:|
| audio | 370 | 223 | 593 |
| visual | 0 | 192 | 192 |
| audio_visual | 672 | 351 | 1023 |
| 总计 | 1042 | 766 | 1808 |

以上是下载前的上限分布。下载失败、媒体损坏、空音轨、时长错误和哈希命中
都会使实际数量下降；替补列表用于恢复到至少 550 个合法媒体。

## 字段转换干跑

`expected_task_distribution.json` 保存了 20 条完整干跑结果：

- AVUT 10 条；
- MUSIC-AVQA-v2.0 10 条；
- 9 个必需字段全部非空；
- 未生成正式 `conversations` 或 train/dev manifest。

字段映射：

| 目标字段 | AVUT | MUSIC-AVQA-v2.0 |
|---|---|---|
| sample_id | `avut_<QA_id>` | 来源、派生 video ID 和 JSON 行号 |
| source_dataset | `AVUT` | `MUSIC-AVQA-v2.0` |
| video_id | 人工标注 `video_id` | suffix/flip 后的段 ID |
| youtube_id | `video_path` 文件名 | CSV `video_id` |
| video_path | 白名单本地目标 | 截取/翻转后的本地目标 |
| scene_audio_path | 与 MP4 相同 | 与 MP4 相同 |
| question | 人工问题 | 模板替换后的问题 |
| answer | 人工答案字母 | 上游 `anser` |
| task_type | AVUT 官方任务映射 | modality + subtype |

不存在无法解决的字段映射。

## 下载与后处理

当前本机具备：

- `yt-dlp 2025.10.14`；
- `ffmpeg/ffprobe 3.1.3-static`；
- `huggingface_hub 0.24.7`；
- yt-dlp 支持 `--download-sections`、`--force-keyframes-at-cuts` 和 MP4 remux。

AVUT 单条下载必须使用白名单中的固定 URL，并校验：

```bash
curl -fL --retry 5 \
  -o /path/to/quarantine/AVUT/<youtube_id>.mp4 \
  '<download_url_from_allowlist>'

sha256sum /path/to/quarantine/AVUT/<youtube_id>.mp4
ffprobe -v error -show_streams -show_format \
  /path/to/quarantine/AVUT/<youtube_id>.mp4
```

MUSIC 单条下载/截取形式：

```bash
yt-dlp \
  --download-sections '*<start_seconds>-<end_seconds>' \
  --force-keyframes-at-cuts \
  --merge-output-format mp4 \
  --remux-video mp4 \
  -o '/path/to/quarantine/MUSIC/<video_id>.%(ext)s' \
  'https://www.youtube.com/watch?v=<youtube_id>'
```

`end_seconds = start_seconds + duration_seconds`。若 `has_flip=true`，在原始截取
媒体通过校验后再用 ffmpeg 生成 `_flip.mp4`，不得重新下载另一份来源。

批量执行前需要先实现
`required_code_changes.md` 中的 allowlist downloader 和严格验证器。
当前不允许手工循环整批下载。

## 通过边界

本预检 PASS 后，允许的下一阶段只有：

```text
实现并测试下载器/转换器/泄漏验证器
-> 按白名单下载到 quarantine
-> 下载后 SHA256、解码、音轨、时长检查
-> 合法媒体转入 accepted
```

仍然禁止：

```text
生成正式 train/dev manifest
启动 2-step 或 20-step 训练
修改正式推理路径
下载 AV_Gemini_data.json 作为主训练标签
绕过 reserve 顺序自行补样本
```
