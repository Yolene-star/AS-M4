import torch

from intersuit.constants import IMAGE_TOKEN_INDEX
from intersuit.model.language_model.llava_qwen import (
    infer_parallel_prefix_length,
    transition_attention_mask_pt,
)


def test_parallel_prefix_length_uses_actual_visual_token_count():
    input_ids = torch.tensor([[1, 2, IMAGE_TOKEN_INDEX, 3, 4, 5]])
    inputs_embeds = torch.zeros(1, 6 - 1 + 8 * 144, 4)

    prefix = infer_parallel_prefix_length(input_ids, inputs_embeds)

    assert prefix == 2 + 8 * 144 + 1


def test_parallel_prefix_length_handles_thirty_two_frames_without_constant():
    input_ids = torch.tensor([[1, 2, IMAGE_TOKEN_INDEX, 3]])
    inputs_embeds = torch.zeros(1, 4 - 1 + 32 * 144, 4)

    prefix = infer_parallel_prefix_length(input_ids, inputs_embeds)

    assert prefix == 2 + 32 * 144 + 1


def test_parallel_prefix_length_without_image_token_keeps_prompt_prefix():
    input_ids = torch.tensor([[1, 2, 3, 4]])
    inputs_embeds = torch.zeros(1, 4, 4)

    prefix = infer_parallel_prefix_length(input_ids, inputs_embeds)

    assert prefix == 4


def test_transition_attention_mask_treats_prefix_length_as_count():
    # prefix_length is a count, so valid prefix indices are 0, 1, 2.
    channel = [0, 0, 0, 1, 2]

    mask = transition_attention_mask_pt(
        1,
        1,
        q_idx=[4],
        kv_idx=[2, 3],
        prefix_length=3,
        channel=channel,
        device=torch.device("cpu"),
    )

    assert mask.tolist() == [[True, False]]


def test_parallel_prefix_with_avut_prompt():
    # Qwen wrapper tokens before/after one image placeholder plus a 32-frame visual block.
    input_ids = torch.tensor([[151644, 8948, IMAGE_TOKEN_INDEX, 198, 1450, 30, 151645, 151644, 77091]])
    visual_embedding_count = 32 * 144
    inputs_embeds = torch.zeros(1, input_ids.shape[1] - 1 + visual_embedding_count, 4)

    prefix = infer_parallel_prefix_length(input_ids, inputs_embeds)

    assert prefix == 2 + visual_embedding_count + 1
    assert 0 < prefix < inputs_embeds.shape[1]
