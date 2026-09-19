"""Explicit SSL decoder, target, and masked-loss interfaces over ``s_t``.

The module intentionally does not choose a biological prediction target.  A
project-specific implementation supplies a decoder and target builder while
this file enforces the stable ``s_t [B,T,D]`` plus ``time_mask [B,T]``
boundary and prevents padded timesteps from contributing to the loss.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

try:  # Keep the base interpreter importable without the optional extra.
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - depends on installation
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: ImportError | None = exc

    class _MissingTorchModule:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_torch()

    class _MissingNN:
        Module = _MissingTorchModule

    nn = _MissingNN()  # type: ignore[assignment]
else:
    _TORCH_IMPORT_ERROR = None


def _require_torch() -> Any:
    if torch is None:  # pragma: no cover - depends on installation
        raise ImportError(
            "interpreter.ssl requires the optional dependency 'torch'. "
            "Install this project with the 'social-gnn' extra."
        ) from _TORCH_IMPORT_ERROR
    return torch


def _normalise_time_mask(
    value: Any,
    *,
    batch_size: int,
    timesteps: int,
    device: "torch.device",
) -> "torch.Tensor":
    runtime = _require_torch()
    if not isinstance(value, runtime.Tensor):
        try:
            value = runtime.as_tensor(value)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TypeError("time_mask must be tensor-like") from exc
    if tuple(value.shape) != (batch_size, timesteps):
        raise ValueError(
            f"time_mask must have shape {(batch_size, timesteps)}, got {tuple(value.shape)}"
        )
    if value.dtype != runtime.bool:
        if value.is_floating_point() and not bool(runtime.isfinite(value).all().item()):
            raise ValueError("time_mask contains a non-finite value")
        if not bool(((value == 0) | (value == 1)).all().item()):
            raise ValueError("time_mask must contain only boolean/0/1 values")
    return value.detach().clone().to(device=device, dtype=runtime.bool)


def _normalise_element_mask(
    value: Any,
    *,
    reference: "torch.Tensor",
) -> "torch.Tensor":
    runtime = _require_torch()
    if not isinstance(value, runtime.Tensor):
        try:
            value = runtime.as_tensor(value)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TypeError("target_mask must be tensor-like") from exc
    if value.ndim < 2 or tuple(value.shape[:2]) != tuple(reference.shape[:2]):
        raise ValueError("target_mask must start with the target [B,T] dimensions")
    if value.ndim > reference.ndim:
        raise ValueError("target_mask cannot have more dimensions than targets")
    for mask_size, target_size in zip(value.shape[2:], reference.shape[2:]):
        if mask_size not in (1, target_size):
            raise ValueError("target_mask must be broadcastable across target elements")
    if value.dtype != runtime.bool:
        if value.is_floating_point() and not bool(runtime.isfinite(value).all().item()):
            raise ValueError("target_mask contains a non-finite value")
        if not bool(((value == 0) | (value == 1)).all().item()):
            raise ValueError("target_mask must contain only boolean/0/1 values")
    return value.detach().clone().to(device=reference.device, dtype=runtime.bool)


def _validate_temporal_values(value: Any, name: str) -> "torch.Tensor":
    runtime = _require_torch()
    if not isinstance(value, runtime.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim < 2:
        raise ValueError(f"{name} must have leading [B,T] dimensions")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{name} dimensions B and T must be positive")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not bool(runtime.isfinite(value).all().item()):
        raise ValueError(f"{name} contains a non-finite value")
    return value


def _expanded_mask(mask: "torch.Tensor", reference: "torch.Tensor") -> "torch.Tensor":
    while mask.ndim < reference.ndim:
        mask = mask.unsqueeze(-1)
    try:
        return mask.expand_as(reference)
    except RuntimeError as exc:
        raise ValueError("mask is not broadcastable to the prediction target shape") from exc


@dataclass(frozen=True)
class PredictionBatch:
    """Decoder predictions paired with the exact temporal validity mask."""

    predictions: "torch.Tensor"
    time_mask: "torch.Tensor"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        predictions = _validate_temporal_values(self.predictions, "predictions")
        time_mask = _normalise_time_mask(
            self.time_mask,
            batch_size=int(predictions.shape[0]),
            timesteps=int(predictions.shape[1]),
            device=predictions.device,
        )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("PredictionBatch.metadata must be a mapping")
        # Predictions retain their graph; only masks/metadata are isolated.
        object.__setattr__(self, "time_mask", time_mask)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def values(self) -> "torch.Tensor":
        """Neutral alias useful for generic objective tooling."""

        return self.predictions


@dataclass(frozen=True)
class TargetBatch:
    """Immutable SSL targets with temporal and optional element masks.

    ``targets`` are always cloned and detached.  Consequently, dynamic decoder
    masking or augmentation cannot silently alter the original supervision.
    ``target_mask`` may be ``[B,T]`` or include broadcastable element axes.
    """

    targets: "torch.Tensor"
    time_mask: "torch.Tensor"
    target_mask: "torch.Tensor | None" = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        targets = _validate_temporal_values(self.targets, "targets").detach().clone()
        time_mask = _normalise_time_mask(
            self.time_mask,
            batch_size=int(targets.shape[0]),
            timesteps=int(targets.shape[1]),
            device=targets.device,
        )
        target_mask = (
            None
            if self.target_mask is None
            else _normalise_element_mask(self.target_mask, reference=targets)
        )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("TargetBatch.metadata must be a mapping")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "time_mask", time_mask)
        object.__setattr__(self, "target_mask", target_mask)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def values(self) -> "torch.Tensor":
        """Neutral alias useful for generic objective tooling."""

        return self.targets


class TemporalDecoder(nn.Module, ABC):  # type: ignore[misc]
    """Abstract decoder whose only primary input is temporal state ``s_t``."""

    @abstractmethod
    def forward(
        self,
        states: "torch.Tensor",
        time_mask: "torch.Tensor",
    ) -> PredictionBatch | "torch.Tensor":
        """Decode ``states [B,T,D_s]`` without treating padding as data."""


class PredictionTargetBuilder(ABC):
    """Build a project-specific target from an isolated copy of ``s_t``."""

    @abstractmethod
    def build(
        self,
        states: "torch.Tensor",
        time_mask: "torch.Tensor",
        *,
        context: Mapping[str, Any] | None = None,
    ) -> TargetBatch:
        """Return detached targets; no biological target is assumed here."""

    def __call__(
        self,
        states: "torch.Tensor",
        time_mask: "torch.Tensor",
        *,
        context: Mapping[str, Any] | None = None,
    ) -> TargetBatch:
        return self.build(states, time_mask, context=context)


class MaskedLoss(nn.Module, ABC):  # type: ignore[misc]
    """Abstract loss over explicit prediction/target containers."""

    @abstractmethod
    def forward(self, prediction: PredictionBatch, target: TargetBatch) -> "torch.Tensor":
        """Return a scalar loss or a documented masked loss tensor."""


class MaskedMSELoss(MaskedLoss):
    """MSE over valid timesteps and optional valid target elements only."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(self, prediction: PredictionBatch, target: TargetBatch) -> "torch.Tensor":
        runtime = _require_torch()
        if not isinstance(prediction, PredictionBatch):
            raise TypeError("prediction must be a PredictionBatch")
        if not isinstance(target, TargetBatch):
            raise TypeError("target must be a TargetBatch")
        if tuple(prediction.predictions.shape) != tuple(target.targets.shape):
            raise ValueError("prediction and target values must have identical shapes")
        if prediction.predictions.device != target.targets.device:
            raise ValueError("prediction and target values must be on the same device")
        if not runtime.equal(prediction.time_mask, target.time_mask):
            raise ValueError("prediction and target time_mask values must match exactly")

        valid = _expanded_mask(target.time_mask, target.targets)
        if target.target_mask is not None:
            valid = valid & _expanded_mask(target.target_mask, target.targets)
        squared_error = (prediction.predictions - target.targets).square()
        masked = runtime.where(valid, squared_error, runtime.zeros_like(squared_error))
        if self.reduction == "none":
            return masked
        total = masked.sum()
        if self.reduction == "sum":
            return total
        count = valid.sum()
        if int(count.item()) == 0:
            raise ValueError("masked MSE has no valid target elements")
        return total / count.to(dtype=total.dtype)


class SSLObjective(nn.Module):  # type: ignore[misc]
    """Orchestrate target snapshot, decoding, and masked loss computation."""

    def __init__(
        self,
        decoder: TemporalDecoder,
        target_builder: PredictionTargetBuilder,
        loss: MaskedLoss | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(decoder, TemporalDecoder):
            raise TypeError("decoder must implement TemporalDecoder")
        if not isinstance(target_builder, PredictionTargetBuilder):
            raise TypeError("target_builder must implement PredictionTargetBuilder")
        if loss is not None and not isinstance(loss, MaskedLoss):
            raise TypeError("loss must implement MaskedLoss")
        self.decoder = decoder
        self.target_builder = target_builder
        self.loss = loss if loss is not None else MaskedMSELoss()

    @staticmethod
    def _inputs(
        states: Any, time_mask: Any
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        states = _validate_temporal_values(states, "states")
        if states.ndim != 3:
            raise ValueError("states must have shape [B,T,D_s]")
        normalised_mask = _normalise_time_mask(
            time_mask,
            batch_size=int(states.shape[0]),
            timesteps=int(states.shape[1]),
            device=states.device,
        )
        return states, normalised_mask

    def prepare(
        self,
        states: "torch.Tensor",
        time_mask: "torch.Tensor",
        *,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[PredictionBatch, TargetBatch]:
        """Create inspectable prediction and target batches without scoring."""

        runtime = _require_torch()
        states, time_mask = self._inputs(states, time_mask)

        # Snapshot supervision first and without autograd.  The decoder gets a
        # differentiable clone, so in-place augmentation cannot alter either
        # the caller's states or the target snapshot.
        with runtime.no_grad():
            target = self.target_builder(
                states.detach().clone(),
                time_mask.detach().clone(),
                context=context,
            )
        if not isinstance(target, TargetBatch):
            raise TypeError("target_builder must return TargetBatch")
        if not runtime.equal(target.time_mask, time_mask):
            raise ValueError("target_builder must preserve the supplied time_mask")

        decoded = self.decoder(states.clone(), time_mask.detach().clone())
        prediction = (
            decoded
            if isinstance(decoded, PredictionBatch)
            else PredictionBatch(predictions=decoded, time_mask=time_mask)
        )
        if not runtime.equal(prediction.time_mask, time_mask):
            raise ValueError("decoder must preserve the supplied time_mask")
        return prediction, target

    def compute(
        self,
        states: "torch.Tensor",
        time_mask: "torch.Tensor",
        *,
        context: Mapping[str, Any] | None = None,
    ) -> tuple["torch.Tensor", PredictionBatch, TargetBatch]:
        """Return loss plus the two structured objects used to compute it."""

        prediction, target = self.prepare(states, time_mask, context=context)
        return self.loss(prediction, target), prediction, target

    def forward(
        self,
        states: "torch.Tensor",
        time_mask: "torch.Tensor",
        *,
        context: Mapping[str, Any] | None = None,
    ) -> "torch.Tensor":
        loss, _prediction, _target = self.compute(
            states, time_mask, context=context
        )
        return loss


__all__ = [
    "MaskedLoss",
    "MaskedMSELoss",
    "PredictionBatch",
    "PredictionTargetBuilder",
    "SSLObjective",
    "TargetBatch",
    "TemporalDecoder",
]
