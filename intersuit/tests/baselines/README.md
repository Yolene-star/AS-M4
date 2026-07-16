# AS-M4 固定回归基线

`avut_e0_e1_e7_cap003.json` 固定 2026-07-17 的 5 条 AVUT E0/E1/E7 smoke 结果，用于后续代码或 checkpoint 变更后的人工/自动回归比较。

- `audio_delta_ratio_cap` 的正式默认值固定为 `0.0`，即默认关闭并保持精确 no-op。
- `0.03` 仅作为已验证的候选实验值保留，不自动启用。
- E1 记录 `cap=0.03` 的结果；E0/E7 记录同一固定样本集的基线结果。
- E7 必须保持 `gate=0` 且 `delta_to_video_ratio=0`，E0 与 E7 的 5 条预测应完全一致。
- 原始预测文件位于被 Git 忽略的运行产物目录；本文件夹只保存必要结果、运行参数和 SHA256，不提交模型、媒体、日志或大体积诊断张量。

固定 CPU 回归命令：

```bash
/home/yjm/miniconda3/envs/M4/bin/python -m pytest -q intersuit/tests
```

基线冻结时结果为 `94 passed, 2 warnings`。如测试数量或结果变化，必须说明新增、删除或行为变化的原因，不能静默覆盖本基线。
