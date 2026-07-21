"""Fixed BEATs 语义配对辅助目标的 CPU 测试。"""

import torch

from intersuit.model.language_model.llava_qwen import scene_audio_semantic_ranking_loss


def test_semantic_ranking_prefers_matching_audio():
    target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    mask = torch.tensor([[True, True]])
    positive = target.clone().requires_grad_()
    negative = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], requires_grad=True)

    loss, diagnostics = scene_audio_semantic_ranking_loss(
        positive, mask, negative, mask, target, mask, margin=0.2
    )

    assert loss.item() == 0.0
    assert diagnostics["positive_similarity"] > diagnostics["negative_similarity"]


def test_semantic_ranking_penalizes_reversed_pair_and_backpropagates():
    target = torch.tensor([[[1.0, 0.0]]])
    mask = torch.tensor([[True]])
    positive = torch.tensor([[[0.0, 1.0]]], requires_grad=True)
    negative = torch.tensor([[[1.0, 0.0]]], requires_grad=True)

    loss, _ = scene_audio_semantic_ranking_loss(
        positive, mask, negative, mask, target, mask, margin=0.2
    )
    loss.backward()

    assert loss.item() > 0.0
    assert positive.grad is not None
    assert negative.grad is not None
