# 新增音视频训练数据落地工具链报告

## 结论

**PASS（仅完成 5 条隔离区烟测，未启动训练，未进入正式训练媒体目录）**

本阶段没有修改 BEATs、M4、动态窗口、Gate、offset scorer 或 MUG 推理路径。新增的 WAV 读取回退只用于解决当前 CPU 环境缺少 torchaudio IO backend 的 Dataset 兼容问题。

## 实施内容

| 项目 | 结果 |
|---|---|
| 白名单下载器 | `intersuit/scripts/download_new_dataset_allowlist.py`，固定 revision、Range 续传、重试、超时、幂等、失败记录；MUSIC 使用固定时间段 `yt-dlp` 截取并隔离保存 |
| 媒体验证器 | `intersuit/scripts/validate_new_dataset_media.py`，ffprobe/ffmpeg 视频音频解码、时长、非空音轨、flip、原始 SHA256 和排除集门禁 |
| 字段转换器 | `intersuit/scripts/convert_new_dataset_training_fields.py`，严格输出 12 个训练字段及 16 kHz mono WAV |
| 泄漏门禁 | `intersuit/scripts/validate_training_data_leakage.py`，同时检查 `video_id`、`youtube_id`、`media_sha256`、train/dev、音频可解码和记录 SHA |
| 排除集合重建 | `intersuit/scripts/build_new_dataset_exclusion_inventory.py`，从仓库根目录现有 manifest/媒体重建可复现摘要 |
| 启动器门禁 | `run_as_m4_beats_stage.sh` 在检查 checkpoint 和调用训练前验证审计状态、manifest 路径和 SHA256 |

## 烟测结果

- 下载主白名单：AVUT 2 条、MUSIC 3 条；`download_status.json` 为 PASS。
- MUSIC 中 1 条带 `_flip`，因此隔离区实际物理文件为 6 个派生媒体。
- 媒体验证：5 个白名单条目全部 PASS，视频/音轨均可解码，音频非空，MUSIC 时长与截取规则通过。
- 字段转换：23 条 QA；audio 7、visual 5、audio_visual 11；AVUT 3 条、MUSIC 20 条。
- `scene_audio_path` 有效率 100%；WAV 为 16 kHz、单声道，实际 Dataset 音频窗口均非空。
- Dataset 读取 23 条成功；DataCollator 组批成功，音频批形状为 `(23, 120, 16000)`。
- 泄漏门禁：`video_id`、`youtube_id`、`media_sha256` 重叠均为 0；train/dev 交叉检查通过。
- 启动器非法 manifest 测试在训练进程和 GPU 初始化前拒绝；合法审计 smoke 仅通过内容门禁后在本地 checkpoint 检查处停止，未进入训练。
- 全部项目测试：**217 passed, 3 skipped**；新增/相关测试：**11 passed**。
- quarantine 目录已加入 `.gitignore`，没有大媒体被 Git 跟踪。

## 产物

以下 JSON 和 SHA 文件位于 `intersuit/datasets/NEW_AV_TRAIN/quarantine/`，并已复制到仓库根目录同名文件以便审阅：

- `download_status.json`
- `download_failures.json`
- `quarantine_validation_report.json`
- `smoke_manifest.json`
- `smoke_manifest.sha256`
- `leakage_gate_report.json`

排除集合原始清单：`intersuit/datasets/NEW_AV_TRAIN/exclusion_inventory.json`（构建时排除本次 quarantine，记录 ID、媒体 SHA、路径数量及字节数摘要）。

## 下一阶段批量下载命令

工具链 PASS 后，才允许执行主名单批量下载；该命令仍只写入隔离区：

```bash
PYTHONPATH=intersuit/scripts \
python intersuit/scripts/download_new_dataset_allowlist.py \
  --allowlist download_allowlist.json \
  --reserve download_reserve_list.json \
  --output-root intersuit/datasets/NEW_AV_TRAIN/quarantine \
  --avut-limit 430 \
  --music-limit 120 \
  --timeout 30 \
  --retries 2
```

下载完成后必须依次运行：

```bash
PYTHONPATH=intersuit/scripts python intersuit/scripts/validate_new_dataset_media.py \
  --download-status intersuit/datasets/NEW_AV_TRAIN/quarantine/download_status.json \
  --exclude-manifest intersuit/datasets/NEW_AV_TRAIN/exclusion_inventory.json \
  --output intersuit/datasets/NEW_AV_TRAIN/quarantine/quarantine_validation_report.json

PYTHONPATH=intersuit/scripts python intersuit/scripts/convert_new_dataset_training_fields.py \
  --validation-report intersuit/datasets/NEW_AV_TRAIN/quarantine/quarantine_validation_report.json \
  --avut-annotations intersuit/datasets/AVUT/raw/AV_Human_data.json \
  --music-annotations intersuit/datasets/NEW_AV_TRAIN/quarantine/metadata/train_balance.json \
  --audio-root intersuit/datasets/NEW_AV_TRAIN/quarantine/audio \
  --output intersuit/datasets/NEW_AV_TRAIN/quarantine/candidate_fields.json
```

只有验证和字段转换均 PASS，且重新生成泄漏审计 PASS 后，才可人工确认并构建正式 train/dev manifest。当前阶段不自动移动媒体、不启动训练。
