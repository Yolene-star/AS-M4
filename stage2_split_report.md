# 阶段 2 正式训练集划分与门禁报告

## 结论

**PASS**

- 内容级去重：1420 → 1420，丢弃物理组 0
- 物理媒体 train/dev/reserve：1000/300/120
- QA train/dev/reserve：2430/780/326
- 三集合 video ID、YouTube ID、派生媒体 ID、物理组和媒体 SHA256 重叠：0
- 冻结 300、历史 21、其他 dev/test 重叠：0
- scene_audio_path 有效率：100%
- 视频与音轨完整解码率：100%
- sample_id 重复：0
- Dataset 加载：PASS
- DataCollator 组批：PASS
- 启动器内容门禁：PASS（gate-only，未启动训练）

## 分层结果

### train

- 任务 QA：audio=1003，visual=753，audio_visual=674
- 来源 QA：AVE_HF_EXPANDED=1200，AVUT=664，MUSIC-AVQA-v2.0=566

### dev

- 任务 QA：audio=322，visual=222，audio_visual=236
- 来源 QA：AVE_HF_EXPANDED=360，AVUT=256，MUSIC-AVQA-v2.0=164

### reserve

- 任务 QA：audio=134，visual=91，audio_visual=101
- 来源 QA：AVE_HF_EXPANDED=144，AVUT=105，MUSIC-AVQA-v2.0=77

所有划分均在 `physical_media_id` 组级完成；同源 flip、裁剪和 suffix 派生媒体没有跨集合。阶段 2 只生成正式 manifest 和门禁证据，阶段 3 训练尚未开始。
