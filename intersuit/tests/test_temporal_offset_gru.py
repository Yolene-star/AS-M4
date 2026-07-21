"""轻量时序 offset 诊断模块的 CPU 单元测试。"""

from __future__ import annotations

import torch

from intersuit.model.streaming_av.temporal_offset_gru import (
    TemporalOffsetGRUDiagnostic,
    ordered_offset_emd_loss,
)


def _inputs(batch: int = 2, steps: int = 6):
    generator = torch.Generator().manual_seed(20260719)
    logits = torch.randn(batch, steps, 3, generator=generator)
    features = torch.randn(batch, steps, 3, 128, generator=generator)
    evidence = torch.randn(batch, steps, 8, generator=generator)
    return logits, features, evidence


def test_forward_step_matches_whole_sequence():
    torch.manual_seed(7)
    model = TemporalOffsetGRUDiagnostic().eval()
    logits, features, evidence = _inputs()

    whole = model(logits, features, evidence)
    state = model.initial_state(logits.shape[0])
    step_logits = []
    step_sync = []
    for index in range(logits.shape[1]):
        output = model.forward_step(
            logits[:, index],
            features[:, index],
            evidence[:, index],
            state,
        )
        state = output.state
        step_logits.append(output.offset_logits)
        step_sync.append(output.synchronizability_logits)

    assert torch.allclose(torch.stack(step_logits, dim=1), whole.offset_logits, atol=1e-6)
    assert torch.allclose(torch.stack(step_sync, dim=1), whole.synchronizability_logits, atol=1e-6)
    assert torch.allclose(state, whole.state, atol=1e-6)


def test_reset_state_clears_only_selected_videos():
    model = TemporalOffsetGRUDiagnostic()
    state = torch.ones(1, 3, 128)

    reset = model.reset_state(state, torch.tensor([False, True, False]))

    assert torch.equal(reset[:, 0], state[:, 0])
    assert torch.equal(reset[:, 1], torch.zeros_like(reset[:, 1]))
    assert torch.equal(reset[:, 2], state[:, 2])
    assert torch.equal(model.reset_state(state), torch.zeros_like(state))


def test_acceptance_requires_sync_nonzero_and_frozen_margin():
    model = TemporalOffsetGRUDiagnostic(synchronizability_threshold=0.5)
    with torch.no_grad():
        model.offset_head.weight.zero_()
        model.offset_head.bias.copy_(torch.tensor([5.0, 0.0, 0.0]))
        model.synchronizability_head.weight.zero_()
        model.synchronizability_head.bias.fill_(10.0)
    _, features, evidence = _inputs(batch=1, steps=1)

    accepted = model(torch.tensor([[[1.0, 0.0, -1.0]]]), features, evidence)
    low_margin = model(torch.tensor([[[0.10, 0.0, -0.10]]]), features, evidence)
    with torch.no_grad():
        model.synchronizability_head.bias.fill_(-10.0)
    unsyncable = model(torch.tensor([[[1.0, 0.0, -1.0]]]), features, evidence)

    assert accepted.accepted.item() is True
    assert accepted.suggested_offset.item() == -0.5
    assert low_margin.accepted.item() is False
    assert low_margin.suggested_offset.item() == 0.0
    assert unsyncable.accepted.item() is False
    assert unsyncable.suggested_offset.item() == 0.0


def test_emd_penalizes_cross_direction_more_than_adjacent_error():
    target = torch.tensor([0])
    adjacent = torch.tensor([[0.0, 8.0, 0.0]])
    opposite = torch.tensor([[0.0, 0.0, 8.0]])

    adjacent_loss = ordered_offset_emd_loss(adjacent, target)
    opposite_loss = ordered_offset_emd_loss(opposite, target)

    assert opposite_loss > adjacent_loss


def test_outputs_are_finite_and_have_expected_shapes():
    model = TemporalOffsetGRUDiagnostic()
    logits, features, evidence = _inputs(batch=3, steps=5)

    output = model(logits, features, evidence)

    assert output.offset_logits.shape == (3, 5, 3)
    assert output.synchronizability_logits.shape == (3, 5)
    assert output.state.shape == (1, 3, 128)
    for value in output:
        if torch.is_tensor(value) and value.is_floating_point():
            assert torch.isfinite(value).all()
