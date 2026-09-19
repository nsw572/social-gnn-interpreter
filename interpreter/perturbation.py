"""Representation-space interventions and trial-preserving experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .core import BehaviorSample, ClusterAdapter, ClusterEvidence, ObservationRef
from .validation import ConstraintResult, ValidationResult, Validator


REPRESENTATION_SCOPE = "representation_space"


def _predict_many(cluster_adapter: ClusterAdapter, states: np.ndarray) -> np.ndarray:
    predictions = np.asarray(cluster_adapter.predict(np.asarray(states)))
    if predictions.ndim == 0:
        predictions = predictions.reshape(1)
    predictions = predictions.reshape(-1)
    if predictions.shape[0] != np.asarray(states).shape[0]:
        raise ValueError("ClusterAdapter.predict must return one label per state")
    return predictions


def _predict_one(cluster_adapter: ClusterAdapter, state: np.ndarray) -> Any:
    prediction = _predict_many(cluster_adapter, np.asarray(state)[None, :])
    value = prediction[0]
    return value.item() if isinstance(value, np.generic) else value


def _get_evidence(
    cluster_adapter: ClusterAdapter, states: np.ndarray
) -> ClusterEvidence | None:
    evidence_method = getattr(cluster_adapter, "evidence", None)
    if evidence_method is None:
        return None
    return evidence_method(np.asarray(states))


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    item = array.reshape(-1)[0]
    try:
        result = float(item)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _score_from_evidence(evidence: Any, cluster: Any) -> float | None:
    """Extract one raw cluster score without assuming its direction/meaning."""

    if evidence is None:
        return None
    score_for = getattr(evidence, "score_for", None)
    if callable(score_for):
        try:
            return _scalar(score_for(cluster))
        except (KeyError, IndexError, TypeError, ValueError):
            return None
    if isinstance(evidence, Mapping):
        return _scalar(evidence.get(cluster))

    scores = getattr(evidence, "scores", None)
    if scores is None:
        return None
    score_array = np.asarray(scores)
    if score_array.ndim >= 2:
        score_array = score_array.reshape(-1, score_array.shape[-1])[0]
    else:
        score_array = score_array.reshape(-1)

    labels = None
    for attribute in ("cluster_ids", "labels", "classes"):
        candidate = getattr(evidence, attribute, None)
        if candidate is not None:
            labels = np.asarray(candidate).reshape(-1)
            break
    if labels is not None and labels.size == score_array.size:
        matches = np.flatnonzero(labels == cluster)
        return _scalar(score_array[matches[0]]) if matches.size else None
    if isinstance(cluster, (int, np.integer)) and 0 <= int(cluster) < score_array.size:
        return _scalar(score_array[int(cluster)])
    return None


def _patch_value(values: Any, timestep: int) -> float | None:
    if values is None:
        return None
    array = np.asarray(values)
    if array.ndim == 0:
        value = array.item()
    else:
        value = array[timestep]
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


class Intervention(ABC):
    """A coherent edit applied to a complete trial at one timestep."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable intervention family name for aggregation."""

    @abstractmethod
    def apply(self, sample: BehaviorSample, timestep: int) -> BehaviorSample:
        """Return an independent modified sample; never mutate ``sample``."""


@dataclass(frozen=True)
class LatentCoordinateIntervention(Intervention):
    """Interpolate one coordinate of ``s_t`` toward a requested target value."""

    index: int
    target_value: float
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.index, (int, np.integer)) or int(self.index) < 0:
            raise ValueError("index must be a non-negative integer")
        if not np.isfinite(self.target_value):
            raise ValueError("target_value must be finite")
        if not np.isfinite(self.alpha) or not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")

    @property
    def name(self) -> str:
        return f"latent_coordinate[{int(self.index)}]"

    def apply(self, sample: BehaviorSample, timestep: int) -> BehaviorSample:
        states = np.asarray(sample.states)
        mask = np.asarray(sample.time_mask, dtype=bool)
        if states.ndim != 2:
            raise ValueError("sample.states must have shape [T, D]")
        if mask.ndim != 1 or mask.shape[0] != states.shape[0]:
            raise ValueError("sample.time_mask must have shape [T]")
        if not isinstance(timestep, (int, np.integer)):
            raise TypeError("timestep must be an integer")
        timestep = int(timestep)
        if timestep < 0 or timestep >= states.shape[0]:
            raise IndexError("timestep is out of bounds")
        if not mask[timestep]:
            raise ValueError("cannot intervene on a padded timestep")
        if int(self.index) >= states.shape[1]:
            raise IndexError("latent coordinate is out of bounds")

        modified_states = np.array(states, dtype=float, copy=True)
        original_value = modified_states[timestep, int(self.index)]
        modified_states[timestep, int(self.index)] = original_value + self.alpha * (
            float(self.target_value) - original_value
        )
        return sample.with_states(modified_states)


@dataclass(frozen=True)
class PerturbationRecord:
    """One intervention attempt on one valid source-cluster observation."""

    sample_id: str
    timestep: int
    patch_start_s: float | None
    patch_end_s: float | None
    source_cluster: Any
    target_cluster: Any
    original_cluster: Any
    new_cluster: Any | None
    intervention_name: str
    original_evidence: ClusterEvidence | None
    modified_evidence: ClusterEvidence | None
    original_cluster_score: float | None
    new_cluster_score: float | None
    source_score_before: float | None
    source_score_after: float | None
    target_score_before: float | None
    target_score_after: float | None
    latent_l2: float | None
    validation: ValidationResult
    valid: bool
    failure_reason: str | None = None
    scope: str = REPRESENTATION_SCOPE

    @property
    def observation(self) -> ObservationRef:
        """Canonical provenance view shared with contrast-layer observations."""

        return ObservationRef(
            sample_id=self.sample_id,
            timestep=self.timestep,
            start_s=self.patch_start_s,
            end_s=self.patch_end_s,
        )


@dataclass(frozen=True)
class PerturbationResult:
    """Cluster-level aggregate plus all inspectable observation records."""

    source_cluster: Any
    target_cluster: Any
    intervention_name: str
    records: tuple[PerturbationRecord, ...]
    leave_source_rate: float
    enter_target_rate: float
    mean_latent_shift: float
    valid_intervention_rate: float
    scope: str = REPRESENTATION_SCOPE

    @property
    def attempted_count(self) -> int:
        return len(self.records)

    @property
    def valid_count(self) -> int:
        return sum(record.valid for record in self.records)


class PerturbationRunner:
    """Apply an intervention only to observations assigned to a source cluster."""

    def __init__(
        self,
        cluster_adapter: ClusterAdapter,
        validator: Validator | None = None,
    ) -> None:
        self.cluster_adapter = cluster_adapter
        self.validator = Validator() if validator is None else validator

    @staticmethod
    def _application_failure(
        intervention_name: str, error: Exception
    ) -> ValidationResult:
        return ValidationResult(
            valid=False,
            constraint_results=(
                ConstraintResult(
                    name="intervention_application",
                    valid=False,
                    details={
                        "intervention": intervention_name,
                        "error_type": type(error).__name__,
                        "reason": str(error),
                    },
                ),
            ),
        )

    def run(
        self,
        samples: Iterable[BehaviorSample],
        source_cluster: Any,
        target_cluster: Any,
        intervention: Intervention,
    ) -> PerturbationResult:
        records: list[PerturbationRecord] = []

        for sample in samples:
            states = np.asarray(sample.states)
            mask = np.asarray(sample.time_mask, dtype=bool)
            if states.ndim != 2:
                raise ValueError("each sample.states must have shape [T, D]")
            if mask.ndim != 1 or mask.shape[0] != states.shape[0]:
                raise ValueError("each sample.time_mask must have shape [T]")

            valid_timesteps = np.flatnonzero(mask)
            predictions = _predict_many(
                self.cluster_adapter, states[valid_timesteps]
            )
            for timestep, original_cluster in zip(valid_timesteps, predictions):
                original_cluster = (
                    original_cluster.item()
                    if isinstance(original_cluster, np.generic)
                    else original_cluster
                )
                if original_cluster != source_cluster:
                    continue

                original_state = states[int(timestep)]
                original_evidence = _get_evidence(
                    self.cluster_adapter, original_state[None, :]
                )
                original_score = _score_from_evidence(
                    original_evidence, original_cluster
                )
                source_before = _score_from_evidence(
                    original_evidence, source_cluster
                )
                target_before = _score_from_evidence(
                    original_evidence, target_cluster
                )

                try:
                    modified = intervention.apply(sample, int(timestep))
                    validation = self.validator.validate(
                        sample, modified, int(timestep)
                    )
                except (IndexError, TypeError, ValueError) as error:
                    validation = self._application_failure(intervention.name, error)
                    records.append(
                        PerturbationRecord(
                            sample_id=sample.sample_id,
                            timestep=int(timestep),
                            patch_start_s=_patch_value(
                                sample.patch_start_s, int(timestep)
                            ),
                            patch_end_s=_patch_value(
                                sample.patch_end_s, int(timestep)
                            ),
                            source_cluster=source_cluster,
                            target_cluster=target_cluster,
                            original_cluster=original_cluster,
                            new_cluster=None,
                            intervention_name=intervention.name,
                            original_evidence=original_evidence,
                            modified_evidence=None,
                            original_cluster_score=original_score,
                            new_cluster_score=None,
                            source_score_before=source_before,
                            source_score_after=None,
                            target_score_before=target_before,
                            target_score_after=None,
                            latent_l2=None,
                            validation=validation,
                            valid=False,
                            failure_reason="intervention_application_failed",
                        )
                    )
                    continue

                if not validation.valid:
                    records.append(
                        PerturbationRecord(
                            sample_id=sample.sample_id,
                            timestep=int(timestep),
                            patch_start_s=_patch_value(
                                sample.patch_start_s, int(timestep)
                            ),
                            patch_end_s=_patch_value(
                                sample.patch_end_s, int(timestep)
                            ),
                            source_cluster=source_cluster,
                            target_cluster=target_cluster,
                            original_cluster=original_cluster,
                            new_cluster=None,
                            intervention_name=intervention.name,
                            original_evidence=original_evidence,
                            modified_evidence=None,
                            original_cluster_score=original_score,
                            new_cluster_score=None,
                            source_score_before=source_before,
                            source_score_after=None,
                            target_score_before=target_before,
                            target_score_after=None,
                            latent_l2=None,
                            validation=validation,
                            valid=False,
                            failure_reason="validation_failed",
                        )
                    )
                    continue

                modified_states = np.asarray(modified.states)
                modified_state = modified_states[int(timestep)]
                new_cluster = _predict_one(self.cluster_adapter, modified_state)
                modified_evidence = _get_evidence(
                    self.cluster_adapter, modified_state[None, :]
                )
                latent_l2 = float(np.linalg.norm(modified_state - original_state))
                records.append(
                    PerturbationRecord(
                        sample_id=sample.sample_id,
                        timestep=int(timestep),
                        patch_start_s=_patch_value(
                            sample.patch_start_s, int(timestep)
                        ),
                        patch_end_s=_patch_value(
                            sample.patch_end_s, int(timestep)
                        ),
                        source_cluster=source_cluster,
                        target_cluster=target_cluster,
                        original_cluster=original_cluster,
                        new_cluster=new_cluster,
                        intervention_name=intervention.name,
                        original_evidence=original_evidence,
                        modified_evidence=modified_evidence,
                        original_cluster_score=original_score,
                        new_cluster_score=_score_from_evidence(
                            modified_evidence, new_cluster
                        ),
                        source_score_before=source_before,
                        source_score_after=_score_from_evidence(
                            modified_evidence, source_cluster
                        ),
                        target_score_before=target_before,
                        target_score_after=_score_from_evidence(
                            modified_evidence, target_cluster
                        ),
                        latent_l2=latent_l2,
                        validation=validation,
                        valid=True,
                    )
                )

        valid_records = [record for record in records if record.valid]
        denominator = len(valid_records)
        leave_source_rate = (
            sum(record.new_cluster != source_cluster for record in valid_records)
            / denominator
            if denominator
            else 0.0
        )
        enter_target_rate = (
            sum(record.new_cluster == target_cluster for record in valid_records)
            / denominator
            if denominator
            else 0.0
        )
        shifts = [
            record.latent_l2
            for record in valid_records
            if record.latent_l2 is not None and np.isfinite(record.latent_l2)
        ]
        mean_latent_shift = float(np.mean(shifts)) if shifts else 0.0
        valid_intervention_rate = (
            denominator / len(records) if records else 0.0
        )
        return PerturbationResult(
            source_cluster=source_cluster,
            target_cluster=target_cluster,
            intervention_name=intervention.name,
            records=tuple(records),
            leave_source_rate=float(leave_source_rate),
            enter_target_rate=float(enter_target_rate),
            mean_latent_shift=mean_latent_shift,
            valid_intervention_rate=float(valid_intervention_rate),
        )


__all__ = [
    "Intervention",
    "LatentCoordinateIntervention",
    "PerturbationRecord",
    "PerturbationResult",
    "PerturbationRunner",
    "REPRESENTATION_SCOPE",
]
