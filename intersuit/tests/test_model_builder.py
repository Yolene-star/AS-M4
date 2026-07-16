from intersuit.model.builder import should_use_qwen2_bf16


def test_qwen2_dtype_detection_uses_model_name_when_path_is_as_m4_run():
    assert should_use_qwen2_bf16(
        "checkpoints/AS-M4-12kbase-smoke-vfeat-asmodules-zero-gatebias-2step",
        "LongVA-Qwen2",
        "eager",
    )


def test_qwen2_dtype_detection_requires_eager_attention():
    assert not should_use_qwen2_bf16(
        "checkpoints/AS-M4-12kbase-smoke-vfeat-asmodules-zero-gatebias-2step",
        "LongVA-Qwen2",
        "flash_attention_2",
    )
