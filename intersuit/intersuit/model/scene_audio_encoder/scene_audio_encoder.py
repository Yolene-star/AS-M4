"""Lightweight scene-audio encoder backends for AS-M4.

These classes define the stable interface used by downstream AS-M4 modules.
Heavy pretrained encoders such as BEATs, PANNs, or CLAP can later implement the
same contract without changing event detection, alignment, or fusion code.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys
from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F


class SceneAudioEncoderOutput(NamedTuple):
    """Scene-audio features plus masks and optional timestamps."""

    features: torch.Tensor
    mask: torch.Tensor
    timestamps: torch.Tensor | None = None
    feature_kind: str = "unspecified"


class DummySceneAudioEncoder(nn.Module):
    """Deterministic waveform-statistics encoder used for smoke tests.

    Input shape is ``[B, T, S]`` where ``T`` is the number of audio windows and
    ``S`` is samples per window. The encoder has no trainable parameters.
    """

    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = int(hidden_size)

    def forward(
        self,
        audio_windows: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
    ) -> SceneAudioEncoderOutput:
        windows = _ensure_batched_windows(audio_windows)
        batch, steps, _ = windows.shape

        if sample_mask is None:
            mask = torch.ones(batch, steps, dtype=torch.bool, device=windows.device)
        else:
            mask = sample_mask.to(device=windows.device, dtype=torch.bool)
            if mask.shape != (batch, steps):
                raise ValueError(f"sample_mask shape {tuple(mask.shape)} does not match {(batch, steps)}")

        stats = _window_statistics(windows)
        features = _expand_to_hidden(stats, self.hidden_size)
        features = features.masked_fill(~mask.unsqueeze(-1), 0.0)
        valid_timestamps = _validate_timestamps(timestamps, batch, steps, windows.device)
        return SceneAudioEncoderOutput(
            features=features,
            mask=mask,
            timestamps=valid_timestamps,
            feature_kind="dummy_waveform_statistics",
        )


class PrecomputedSceneAudioEncoder(nn.Module):
    """Adapter for precomputed scene-audio features.

    If ``input_dim`` differs from ``hidden_size``, a frozen deterministic linear
    projection is applied. This keeps first-pass harnesses dependency-free while
    exposing the same shape as future pretrained encoders.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        input_dim: int | None = None,
        shared_semantic_space: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = int(hidden_size)
        self.input_dim = int(input_dim) if input_dim is not None else None
        self.shared_semantic_space = bool(shared_semantic_space)
        self.proj: nn.Linear | None = None
        if self.input_dim is not None and self.input_dim != self.hidden_size:
            self.proj = nn.Linear(self.input_dim, self.hidden_size, bias=False)
            nn.init.zeros_(self.proj.weight)
            diag = min(self.input_dim, self.hidden_size)
            with torch.no_grad():
                self.proj.weight[:diag, :diag] = torch.eye(diag)
            for param in self.proj.parameters():
                param.requires_grad = False

    def forward(
        self,
        features: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
    ) -> SceneAudioEncoderOutput:
        encoded = _ensure_batched_features(features).to(dtype=torch.float32)
        batch, steps, dim = encoded.shape
        if not torch.isfinite(encoded).all():
            raise ValueError("precomputed scene-audio features must be finite")

        if self.proj is not None:
            if dim != self.input_dim:
                raise ValueError(f"Expected precomputed dim {self.input_dim}, got {dim}")
            encoded = self.proj(encoded)
        elif dim != self.hidden_size:
            encoded = _expand_to_hidden(encoded, self.hidden_size)

        if sample_mask is None:
            mask = torch.ones(batch, steps, dtype=torch.bool, device=encoded.device)
        else:
            mask = sample_mask.to(device=encoded.device, dtype=torch.bool)
            if mask.shape != (batch, steps):
                raise ValueError(f"sample_mask shape {tuple(mask.shape)} does not match {(batch, steps)}")

        encoded = encoded.masked_fill(~mask.unsqueeze(-1), 0.0)
        valid_timestamps = _validate_timestamps(timestamps, batch, steps, encoded.device)
        feature_kind = "shared_precomputed_semantic" if self.shared_semantic_space else "precomputed_audio_features"
        return SceneAudioEncoderOutput(
            features=encoded,
            mask=mask,
            timestamps=valid_timestamps,
            feature_kind=feature_kind,
        )


class FrozenTorchaudioSceneAudioEncoder(nn.Module):
    """Frozen torchaudio Wav2Vec2/HuBERT speech acoustic baseline encoder."""

    def __init__(
        self,
        hidden_size: int = 768,
        bundle_name: str = "WAV2VEC2_BASE",
        sample_rate: int = 16000,
        weight_path: str | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        try:
            import torchaudio
        except ImportError as exc:
            raise ImportError("torchaudio is required for frozen_torchaudio scene audio encoder") from exc

        bundle = getattr(torchaudio.pipelines, str(bundle_name), None)
        if bundle is None:
            raise ValueError(f"Unknown torchaudio pipeline bundle: {bundle_name}")
        local_weight = _resolve_torchaudio_bundle_weight(bundle, weight_path)
        model = bundle.get_model(dl_kwargs={"model_dir": str(local_weight.parent), "progress": False})
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        self.encoder = model
        self.bundle_name = str(bundle_name)
        self.feature_kind = "speech_acoustic_baseline"
        self.local_weight_path = str(local_weight)
        self.sample_rate = int(sample_rate)
        self.bundle_sample_rate = int(getattr(bundle, "sample_rate", sample_rate))
        self.hidden_size = int(hidden_size)
        self.encoder_dim = int(getattr(bundle, "_params", {}).get("encoder_embed_dim", hidden_size))
        self.proj: nn.Linear | None = None
        if self.encoder_dim != self.hidden_size:
            self.proj = nn.Linear(self.encoder_dim, self.hidden_size, bias=False)
            nn.init.zeros_(self.proj.weight)
            diag = min(self.encoder_dim, self.hidden_size)
            with torch.no_grad():
                self.proj.weight[:diag, :diag] = torch.eye(diag)

    def forward(
        self,
        audio_windows: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
    ) -> SceneAudioEncoderOutput:
        windows = _ensure_batched_windows(audio_windows)
        batch, steps, samples = windows.shape
        if sample_mask is None:
            mask = torch.ones(batch, steps, dtype=torch.bool, device=windows.device)
        else:
            mask = sample_mask.to(device=windows.device, dtype=torch.bool)
            if mask.shape != (batch, steps):
                raise ValueError(f"sample_mask shape {tuple(mask.shape)} does not match {(batch, steps)}")

        flat = windows.reshape(batch * steps, samples)
        lengths = torch.full((flat.shape[0],), samples, dtype=torch.long, device=flat.device)
        with torch.no_grad():
            encoded, encoded_lengths = self.encoder(flat, lengths)
            encoded_mask = _lengths_to_mask(encoded_lengths.to(encoded.device), encoded.shape[1])
            denom = encoded_mask.to(encoded.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (encoded * encoded_mask.unsqueeze(-1).to(encoded.dtype)).sum(dim=1) / denom
        pooled = pooled.reshape(batch, steps, -1)
        if self.proj is not None:
            pooled = self.proj(pooled.to(dtype=self.proj.weight.dtype)).to(dtype=torch.float32)
        pooled = pooled.masked_fill(~mask.unsqueeze(-1), 0.0)
        valid_timestamps = _validate_timestamps(timestamps, batch, steps, pooled.device)
        return SceneAudioEncoderOutput(
            features=pooled,
            mask=mask,
            timestamps=valid_timestamps,
            feature_kind=self.feature_kind,
        )


class FrozenBEATsSceneAudioEncoder(nn.Module):
    """Frozen local BEATs encoder followed by a trainable audio projector.

    BEATs is intentionally kept outside the registered module tree. Its
    external checkpoint remains the source of truth and is not duplicated in
    every M4 checkpoint; only ``audio_projector`` is trainable and saved.
    """

    def __init__(
        self,
        hidden_size: int,
        checkpoint_path: str,
        code_root: str,
        sample_rate: int = 16000,
        expected_sha256: str | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if sample_rate != 16000:
            raise ValueError("BEATs scene audio currently requires 16000 Hz input")

        checkpoint = _resolve_local_file(checkpoint_path, "BEATs checkpoint")
        source = _resolve_local_file(Path(code_root) / "beats/BEATs.py", "BEATs source")
        checkpoint_sha256 = _sha256_file(checkpoint)
        if expected_sha256 and checkpoint_sha256 != str(expected_sha256).lower():
            raise ValueError(
                f"BEATs checkpoint SHA256 mismatch: expected={expected_sha256}, "
                f"actual={checkpoint_sha256}"
            )

        source_root = source.parents[1]
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        beats_module = importlib.import_module("beats.BEATs")
        beats_config = getattr(beats_module, "BEATsConfig")
        beats_class = getattr(beats_module, "BEATs")

        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("BEATs checkpoint must contain a dictionary")
        cfg = payload.get("cfg") or payload.get("config") or {}
        state = payload.get("model") or payload.get("state_dict") or payload
        if not isinstance(state, dict):
            raise ValueError("BEATs checkpoint does not contain model weights")

        beats = beats_class(beats_config(cfg))
        incompatible = beats.load_state_dict(_strip_module_prefix(state), strict=False)
        allowed_missing = {"predictor.weight", "predictor.bias"}
        disallowed_missing = sorted(set(incompatible.missing_keys) - allowed_missing)
        if disallowed_missing or incompatible.unexpected_keys:
            raise ValueError(
                "BEATs checkpoint is incompatible with the local source: "
                f"missing={disallowed_missing}, unexpected={incompatible.unexpected_keys}"
            )
        beats.eval()
        for param in beats.parameters():
            param.requires_grad = False
        object.__setattr__(self, "_beats_model", beats)

        encoder_dim = int(getattr(beats.cfg, "encoder_embed_dim", 0))
        if encoder_dim <= 0:
            raise ValueError("BEATs config does not expose encoder_embed_dim")
        self.audio_projector = nn.Linear(encoder_dim, int(hidden_size))
        nn.init.xavier_uniform_(self.audio_projector.weight)
        nn.init.zeros_(self.audio_projector.bias)

        self.hidden_size = int(hidden_size)
        self.encoder_dim = encoder_dim
        self.sample_rate = int(sample_rate)
        self.checkpoint_path = str(checkpoint)
        self.code_root = str(source_root)
        self.checkpoint_sha256 = checkpoint_sha256
        self.feature_kind = "beats_projected_semantic"

    @property
    def beats_model(self) -> nn.Module:
        return object.__getattribute__(self, "_beats_model")

    def _apply(self, fn):
        super()._apply(fn)
        self.beats_model._apply(fn)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.beats_model.eval()
        return self

    def forward(
        self,
        audio_windows: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
    ) -> SceneAudioEncoderOutput:
        windows = _ensure_batched_windows(audio_windows)
        batch, steps, _ = windows.shape
        if sample_mask is None:
            mask = torch.ones(batch, steps, dtype=torch.bool, device=windows.device)
        else:
            mask = sample_mask.to(device=windows.device, dtype=torch.bool)
            if mask.shape != (batch, steps):
                raise ValueError(f"sample_mask shape {tuple(mask.shape)} does not match {(batch, steps)}")

        flat = windows.detach().float().cpu().reshape(batch * steps, -1)
        fbanks = torch.stack([_waveform_to_fbank(window, self.sample_rate) for window in flat])
        beats_param = next(self.beats_model.parameters())
        fbanks = fbanks.to(device=beats_param.device, dtype=beats_param.dtype)
        with torch.no_grad():
            encoded, _, _ = self.beats_model.extract_features(
                fbanks,
                padding_mask=None,
                feature_only=True,
            )
            pooled = encoded.float().mean(dim=1)
        if not torch.isfinite(pooled).all():
            raise ValueError("BEATs produced NaN/Inf embeddings")

        projected = self.audio_projector(
            pooled.to(
                device=self.audio_projector.weight.device,
                dtype=self.audio_projector.weight.dtype,
            )
        )
        projected = projected.reshape(batch, steps, self.hidden_size)
        projected = projected.masked_fill(~mask.to(projected.device).unsqueeze(-1), 0.0)
        valid_timestamps = _validate_timestamps(timestamps, batch, steps, projected.device)
        return SceneAudioEncoderOutput(
            features=projected,
            mask=mask.to(projected.device),
            timestamps=valid_timestamps,
            feature_kind=self.feature_kind,
        )


def _ensure_batched_windows(audio_windows: torch.Tensor) -> torch.Tensor:
    if audio_windows.ndim == 2:
        audio_windows = audio_windows.unsqueeze(0)
    if audio_windows.ndim != 3:
        raise ValueError(f"Expected audio windows shaped [B,T,S] or [T,S], got {tuple(audio_windows.shape)}")
    return audio_windows.to(dtype=torch.float32)


def _ensure_batched_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 2:
        features = features.unsqueeze(0)
    if features.ndim != 3:
        raise ValueError(f"Expected features shaped [B,T,D] or [T,D], got {tuple(features.shape)}")
    return features


def _validate_timestamps(
    timestamps: torch.Tensor | None,
    batch: int,
    steps: int,
    device: torch.device,
) -> torch.Tensor | None:
    if timestamps is None:
        return None
    values = timestamps.to(device=device, dtype=torch.float32)
    if values.ndim == 2:
        values = values.unsqueeze(0).expand(batch, -1, -1)
    if values.shape != (batch, steps, 2):
        raise ValueError(f"timestamps shape {tuple(values.shape)} does not match {(batch, steps, 2)}")
    if not torch.isfinite(values).all():
        raise ValueError("timestamps must be finite")
    if (values[..., 0] > values[..., 1]).any():
        raise ValueError("timestamps must satisfy start <= end for every window")
    return values


def _resolve_torchaudio_bundle_weight(bundle, weight_path: str | None) -> Path:
    expected_name = getattr(bundle, "_path", None)
    if not expected_name:
        raise ValueError("torchaudio bundle does not expose a local weight filename")
    if weight_path is not None:
        path = Path(weight_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"Torchaudio speech_acoustic_baseline weight file not found: {path}. "
                "Automatic downloads are disabled; prepare the Wav2Vec2/HuBERT weights manually."
            )
        if path.name != expected_name:
            raise ValueError(
                f"Torchaudio bundle expects cached filename {expected_name}, got {path.name}. "
                "Place or link the weight with the expected bundle filename to avoid automatic downloads."
            )
        return path

    cache_path = Path(torch.hub.get_dir()).expanduser() / "checkpoints" / expected_name
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Torchaudio speech_acoustic_baseline weights are not cached at {cache_path}. "
            "Automatic downloads are disabled; cache the Wav2Vec2/HuBERT bundle weights manually "
            "or pass scene_audio_torchaudio_weight_path."
        )
    return cache_path


def _window_statistics(windows: torch.Tensor) -> torch.Tensor:
    mean = windows.mean(dim=-1)
    std = windows.std(dim=-1, unbiased=False)
    rms = torch.sqrt(torch.clamp((windows * windows).mean(dim=-1), min=0.0))
    max_abs = windows.abs().amax(dim=-1)
    min_value = windows.amin(dim=-1)
    max_value = windows.amax(dim=-1)
    zero_cross = ((windows[..., 1:] * windows[..., :-1]) < 0).float().mean(dim=-1)
    return torch.stack([mean, std, rms, max_abs, min_value, max_value, zero_cross], dim=-1)


def _expand_to_hidden(features: torch.Tensor, hidden_size: int) -> torch.Tensor:
    if features.shape[-1] == hidden_size:
        return features
    if features.shape[-1] > hidden_size:
        return features[..., :hidden_size]
    repeat = (hidden_size + features.shape[-1] - 1) // features.shape[-1]
    expanded = features.repeat(1, 1, repeat)
    return expanded[..., :hidden_size]


def _lengths_to_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def _resolve_local_file(path_value: str | Path, label: str) -> Path:
    path = Path(path_value).expanduser()
    repo_root = Path(__file__).resolve().parents[4]
    candidates = [path] if path.is_absolute() else [
        (Path.cwd() / path).resolve(),
        (repo_root / path).resolve(),
        (repo_root / "intersuit" / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{label} not found: {path_value}. Automatic downloads are disabled."
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_module_prefix(state: dict) -> dict:
    return {
        key.removeprefix("module."): value
        for key, value in state.items()
        if torch.is_tensor(value)
    }


def _waveform_to_fbank(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    import torchaudio.compliance.kaldi as ta_kaldi

    mono = waveform.float().flatten().unsqueeze(0)
    return ta_kaldi.fbank(
        mono * (2**15),
        num_mel_bins=128,
        sample_frequency=sample_rate,
        frame_length=25,
        frame_shift=10,
    )
