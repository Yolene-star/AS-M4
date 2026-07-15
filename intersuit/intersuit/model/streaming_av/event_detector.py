"""Audio event detector for AS-M4 streaming scene audio."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F


class AudioEventDetectorOutput(NamedTuple):
    """Window-level event predictions."""

    event_logits: torch.Tensor
    eventness_logits: torch.Tensor
    eventness: torch.Tensor
    boundary_logits: torch.Tensor
    mask: torch.Tensor


class AudioEventDetector(nn.Module):
    """A lightweight MLP event head over scene-audio features.

    The first-pass detector is intentionally small. It exposes stable outputs
    for downstream scheduling, alignment, and loss code while heavier temporal
    models can replace it later.
    """

    def __init__(
        self,
        input_dim: int,
        num_events: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        eventness_prior_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_events <= 0:
            raise ValueError("num_events must be positive")
        hidden = int(hidden_dim or max(32, min(512, input_dim)))
        self.input_dim = int(input_dim)
        self.num_events = int(num_events)
        self.eventness_prior_scale = float(eventness_prior_scale)

        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.event_head = nn.Linear(hidden, num_events)
        self.eventness_head = nn.Linear(hidden, 1)
        self.boundary_head = nn.Linear(hidden, 2)

    def forward(
        self,
        audio_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> AudioEventDetectorOutput:
        if audio_features.ndim != 3:
            raise ValueError(f"Expected audio_features shaped [B,T,D], got {tuple(audio_features.shape)}")
        batch, steps, dim = audio_features.shape
        if dim != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {dim}")

        if mask is None:
            valid_mask = torch.ones(batch, steps, dtype=torch.bool, device=audio_features.device)
        else:
            valid_mask = mask.to(device=audio_features.device, dtype=torch.bool)
            if valid_mask.shape != (batch, steps):
                raise ValueError(f"mask shape {tuple(valid_mask.shape)} does not match {(batch, steps)}")

        hidden = self.trunk(audio_features)
        event_logits = self.event_head(hidden)
        boundary_logits = self.boundary_head(hidden)

        energy_prior = audio_features.abs().mean(dim=-1, keepdim=True)
        eventness_logits = self.eventness_head(hidden) + self.eventness_prior_scale * energy_prior
        eventness_logits = eventness_logits.squeeze(-1)
        eventness = torch.sigmoid(eventness_logits)

        event_logits = event_logits.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        boundary_logits = boundary_logits.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        eventness_logits = eventness_logits.masked_fill(~valid_mask, 0.0)
        eventness = eventness.masked_fill(~valid_mask, 0.0)

        return AudioEventDetectorOutput(
            event_logits=event_logits,
            eventness_logits=eventness_logits,
            eventness=eventness,
            boundary_logits=boundary_logits,
            mask=valid_mask,
        )


def compute_event_loss(
    output: AudioEventDetectorOutput,
    event_labels: torch.Tensor | None = None,
    eventness_labels: torch.Tensor | None = None,
    boundary_labels: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute masked auxiliary losses for event detector training."""

    valid_mask = output.mask if mask is None else mask.to(device=output.eventness.device, dtype=torch.bool)
    losses: list[torch.Tensor] = []

    if event_labels is not None:
        labels = event_labels.to(device=output.event_logits.device, dtype=torch.long)
        per_item = F.cross_entropy(
            output.event_logits.reshape(-1, output.event_logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        losses.append(_masked_mean(per_item, valid_mask))

    if eventness_labels is not None:
        labels = eventness_labels.to(device=output.eventness_logits.device, dtype=torch.float32)
        per_item = F.binary_cross_entropy_with_logits(
            output.eventness_logits,
            labels,
            reduction="none",
        )
        losses.append(_masked_mean(per_item, valid_mask))

    if boundary_labels is not None:
        labels = boundary_labels.to(device=output.boundary_logits.device, dtype=torch.long)
        per_item = F.cross_entropy(
            output.boundary_logits.reshape(-1, output.boundary_logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        losses.append(_masked_mean(per_item, valid_mask))

    if not losses:
        return output.eventness.sum() * 0.0
    return torch.stack(losses).sum()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(device=values.device, dtype=values.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (values * mask_f).sum() / denom
