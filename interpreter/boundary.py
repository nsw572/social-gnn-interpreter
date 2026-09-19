"""Bounded counterfactual search over validated representation-space paths."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

from .core import BehaviorSample, ClusterAdapter
from .perturbation import (
    REPRESENTATION_SCOPE,
    Intervention,
    LatentCoordinateIntervention,
    _get_evidence,
    _predict_one,
    _score_from_evidence,
)
from .validation import ConstraintResult, ValidationResult, Validator


class DistanceMetric(ABC):
    """Cost of a proposed change at the interpreter boundary."""

    @abstractmethod
    def compute(
        self,
        original: BehaviorSample,
        modified: BehaviorSample,
        timestep: int | None = None,
    ) -> float:
        """Return a non-negative representation cost."""


class StandardizedL2Distance(DistanceMetric):
    """L2 distance after scaling by reference-state standard deviations.

    This is a representation-space cost, not a claim of behavioral distance.
    The scale must be supplied explicitly or learned from reference states.
    """

    def __init__(
        self,
        reference_states: Any | None = None,
        *,
        scale: Any | None = None,
    ) -> None:
        if reference_states is not None and scale is not None:
            raise ValueError("provide reference_states or scale, not both")
        self.scale_: np.ndarray | None = None
        if scale is not None:
            self._set_scale(scale)
        if reference_states is not None:
            self.fit(reference_states)

    def _set_scale(self, scale: Any) -> None:
        values = np.asarray(scale, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("scale must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("scale entries must be finite and positive")
        self.scale_ = np.array(values, copy=True)
        self.scale_.setflags(write=False)

    def fit(self, reference_states: Any) -> "StandardizedL2Distance":
        states = np.asarray(reference_states, dtype=float)
        if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 1:
            raise ValueError(
                "reference_states must contain at least two rows with shape [M, D]"
            )
        if not np.all(np.isfinite(states)):
            raise ValueError("reference_states must be finite")
        scale = states.std(axis=0)
        scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
        self._set_scale(scale)
        return self

    def compute(
        self,
        original: BehaviorSample,
        modified: BehaviorSample,
        timestep: int | None = None,
    ) -> float:
        if self.scale_ is None:
            raise RuntimeError(
                "StandardizedL2Distance must be fitted or given an explicit scale"
            )
        original_states = np.asarray(original.states, dtype=float)
        modified_states = np.asarray(modified.states, dtype=float)
        if original_states.shape != modified_states.shape or original_states.ndim != 2:
            raise ValueError("original and modified states must share shape [T, D]")
        if original_states.shape[1] != self.scale_.size:
            raise ValueError("distance scale dimension does not match states")
        if timestep is None:
            mask = np.asarray(original.time_mask, dtype=bool)
            differences = original_states[mask] - modified_states[mask]
        else:
            timestep = int(timestep)
            if not 0 <= timestep < original_states.shape[0]:
                raise IndexError("timestep is out of bounds")
            differences = (
                original_states[timestep] - modified_states[timestep]
            )[None, :]
        cost = float(np.linalg.norm(differences / self.scale_[None, :]))
        if not np.isfinite(cost) or cost < 0.0:
            raise ValueError("distance metric produced an invalid cost")
        return cost


class InterventionPath(ABC):
    """A bounded one-dimensional family of coherent interventions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable path name used in boundary profiles."""

    @property
    @abstractmethod
    def lower_bound(self) -> float:
        """Smallest magnitude included in the search."""

    @property
    @abstractmethod
    def upper_bound(self) -> float:
        """Largest magnitude included in the search."""

    @abstractmethod
    def at(self, magnitude: float) -> Intervention:
        """Construct the intervention at a bounded path magnitude."""

    @abstractmethod
    def changed_variables(
        self,
        original: BehaviorSample,
        modified: BehaviorSample,
        timestep: int,
        magnitude: float,
    ) -> Mapping[str, Any]:
        """Describe the realized change without assigning semantic labels."""


@dataclass(frozen=True)
class LatentCoordinatePath(InterventionPath):
    """Search interpolation strength for one coordinate of ``s_t``."""

    index: int
    target_value: float
    minimum: float = 0.0
    maximum: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.index, (int, np.integer)) or int(self.index) < 0:
            raise ValueError("index must be a non-negative integer")
        if not np.isfinite(self.target_value):
            raise ValueError("target_value must be finite")
        if not np.isfinite(self.minimum) or not np.isfinite(self.maximum):
            raise ValueError("path bounds must be finite")
        if not 0.0 <= self.minimum < self.maximum <= 1.0:
            raise ValueError("latent interpolation bounds must satisfy 0 <= min < max <= 1")

    @property
    def name(self) -> str:
        return f"latent_coordinate[{int(self.index)}]"

    @property
    def lower_bound(self) -> float:
        return float(self.minimum)

    @property
    def upper_bound(self) -> float:
        return float(self.maximum)

    def at(self, magnitude: float) -> LatentCoordinateIntervention:
        magnitude = float(magnitude)
        tolerance = np.finfo(float).eps * 8
        if not self.lower_bound - tolerance <= magnitude <= self.upper_bound + tolerance:
            raise ValueError("magnitude lies outside the path bounds")
        magnitude = min(self.upper_bound, max(self.lower_bound, magnitude))
        return LatentCoordinateIntervention(
            index=int(self.index),
            target_value=float(self.target_value),
            alpha=magnitude,
        )

    def changed_variables(
        self,
        original: BehaviorSample,
        modified: BehaviorSample,
        timestep: int,
        magnitude: float,
    ) -> Mapping[str, Any]:
        original_value = float(original.states[timestep, int(self.index)])
        counterfactual_value = float(modified.states[timestep, int(self.index)])
        return MappingProxyType(
            {
                "latent_index": int(self.index),
                "original_value": original_value,
                "counterfactual_value": counterfactual_value,
                "requested_target_value": float(self.target_value),
                "alpha": float(magnitude),
            }
        )


@dataclass(frozen=True)
class CounterfactualResult:
    """A successful crossing or a structured account of why none was found."""

    sample_id: str
    timestep: int
    source_cluster: Any
    target_cluster: Any
    success: bool
    intervention_name: str | None
    changed_variables: Mapping[str, Any] = field(default_factory=dict)
    intervention_cost: float | None = None
    magnitude: float | None = None
    new_cluster: Any | None = None
    original_cluster_score: float | None = None
    target_cluster_score: float | None = None
    validation: ValidationResult | None = None
    counterfactual_state: np.ndarray | None = None
    failure_reason: str | None = None
    searched_candidates: int = 0
    valid_candidates: int = 0
    scope: str = REPRESENTATION_SCOPE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "changed_variables", MappingProxyType(dict(self.changed_variables))
        )
        if self.counterfactual_state is not None:
            state = np.array(self.counterfactual_state, dtype=float, copy=True)
            state.setflags(write=False)
            object.__setattr__(self, "counterfactual_state", state)
        if self.success:
            if self.intervention_cost is None or self.magnitude is None:
                raise ValueError("successful counterfactuals require cost and magnitude")
            if self.new_cluster != self.target_cluster:
                raise ValueError("successful counterfactual must enter target_cluster")
            if self.validation is None or not self.validation.valid:
                raise ValueError("successful counterfactual must pass validation")
            if self.failure_reason is not None:
                raise ValueError("successful counterfactual cannot have failure_reason")
        elif self.failure_reason is None:
            raise ValueError("failed counterfactuals require failure_reason")

    @property
    def valid(self) -> bool:
        return bool(self.success and self.validation is not None and self.validation.valid)


@dataclass(frozen=True)
class _Candidate:
    path: InterventionPath
    magnitude: float
    modified: BehaviorSample
    validation: ValidationResult
    cluster: Any
    target_score: float | None
    cost: float


class CounterfactualSolver:
    """Grid-search bounded paths, then refine validated crossing brackets."""

    def __init__(
        self,
        cluster_adapter: ClusterAdapter,
        validator: Validator,
        distance_metric: DistanceMetric,
        *,
        grid_size: int = 21,
        bisection_steps: int = 12,
        magnitude_tolerance: float = 1e-6,
    ) -> None:
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if bisection_steps < 0:
            raise ValueError("bisection_steps must be non-negative")
        if magnitude_tolerance <= 0.0 or not np.isfinite(magnitude_tolerance):
            raise ValueError("magnitude_tolerance must be finite and positive")
        self.cluster_adapter = cluster_adapter
        self.validator = validator
        self.distance_metric = distance_metric
        self.grid_size = int(grid_size)
        self.bisection_steps = int(bisection_steps)
        self.magnitude_tolerance = float(magnitude_tolerance)

    @staticmethod
    def _application_failure(
        path: InterventionPath, magnitude: float, error: Exception
    ) -> ValidationResult:
        return ValidationResult(
            valid=False,
            constraint_results=(
                ConstraintResult(
                    name="intervention_application",
                    valid=False,
                    details={
                        "path": path.name,
                        "magnitude": float(magnitude),
                        "error_type": type(error).__name__,
                        "reason": str(error),
                    },
                ),
            ),
        )

    def solve(
        self,
        sample: BehaviorSample,
        target_cluster: Any,
        allowed_interventions: InterventionPath | Iterable[InterventionPath],
        *,
        timestep: int | None = None,
        source_cluster: Any | None = None,
    ) -> CounterfactualResult:
        if isinstance(allowed_interventions, InterventionPath):
            paths = (allowed_interventions,)
        else:
            paths = tuple(allowed_interventions)
        if not paths:
            raise ValueError("allowed_interventions must contain at least one path")
        if not all(isinstance(path, InterventionPath) for path in paths):
            raise TypeError("allowed_interventions must contain InterventionPath objects")

        mask = np.asarray(sample.time_mask, dtype=bool)
        if timestep is None:
            timestep = int(np.flatnonzero(mask)[0])
        if not isinstance(timestep, (int, np.integer)):
            raise TypeError("timestep must be an integer")
        timestep = int(timestep)
        if timestep < 0 or timestep >= mask.size or not mask[timestep]:
            raise ValueError("timestep must identify a valid, non-padding state")

        original_state = np.asarray(sample.states[timestep])
        observed_source = _predict_one(self.cluster_adapter, original_state)
        original_evidence = _get_evidence(
            self.cluster_adapter, original_state[None, :]
        )
        original_score = _score_from_evidence(original_evidence, observed_source)
        if source_cluster is None:
            source_cluster = observed_source
        elif observed_source != source_cluster:
            return CounterfactualResult(
                sample_id=sample.sample_id,
                timestep=timestep,
                source_cluster=source_cluster,
                target_cluster=target_cluster,
                success=False,
                intervention_name=None,
                original_cluster_score=original_score,
                failure_reason="source_cluster_mismatch",
            )
        if observed_source == target_cluster:
            return CounterfactualResult(
                sample_id=sample.sample_id,
                timestep=timestep,
                source_cluster=source_cluster,
                target_cluster=target_cluster,
                success=False,
                intervention_name=None,
                original_cluster_score=original_score,
                failure_reason="already_in_target_cluster",
            )

        target_candidates: list[_Candidate] = []
        searched_count = 0
        valid_count = 0
        last_validation: ValidationResult | None = None

        def evaluate(path: InterventionPath, magnitude: float) -> _Candidate | None:
            nonlocal searched_count, valid_count, last_validation
            searched_count += 1
            try:
                modified = path.at(float(magnitude)).apply(sample, timestep)
                validation = self.validator.validate(sample, modified, timestep)
            except (IndexError, TypeError, ValueError) as error:
                last_validation = self._application_failure(path, magnitude, error)
                return None
            last_validation = validation
            if not validation.valid:
                return None
            valid_count += 1
            modified_state = np.asarray(modified.states[timestep])
            cluster = _predict_one(self.cluster_adapter, modified_state)
            evidence = _get_evidence(
                self.cluster_adapter, modified_state[None, :]
            )
            cost = self.distance_metric.compute(sample, modified, timestep)
            return _Candidate(
                path=path,
                magnitude=float(magnitude),
                modified=modified,
                validation=validation,
                cluster=cluster,
                target_score=_score_from_evidence(evidence, target_cluster),
                cost=cost,
            )

        for path in paths:
            magnitudes = np.linspace(
                path.lower_bound, path.upper_bound, self.grid_size
            )
            grid_candidates: list[_Candidate | None] = []
            for magnitude in magnitudes:
                candidate = evaluate(path, float(magnitude))
                grid_candidates.append(candidate)
                if candidate is not None and candidate.cluster == target_cluster:
                    target_candidates.append(candidate)

            for left, right in zip(grid_candidates[:-1], grid_candidates[1:]):
                if (
                    left is None
                    or right is None
                    or left.cluster == target_cluster
                    or right.cluster != target_cluster
                ):
                    continue
                low = left.magnitude
                high = right.magnitude
                best = right
                for _ in range(self.bisection_steps):
                    if high - low <= self.magnitude_tolerance:
                        break
                    midpoint = 0.5 * (low + high)
                    candidate = evaluate(path, midpoint)
                    if candidate is not None and candidate.cluster == target_cluster:
                        high = midpoint
                        best = candidate
                        target_candidates.append(candidate)
                    else:
                        # Invalid midpoints are not counterfactuals.  Retaining the
                        # known-valid target endpoint searches the valid upper side.
                        low = midpoint
                target_candidates.append(best)

        if not target_candidates:
            reason = (
                "no_valid_candidates"
                if valid_count == 0
                else "no_valid_target_crossing"
            )
            return CounterfactualResult(
                sample_id=sample.sample_id,
                timestep=timestep,
                source_cluster=source_cluster,
                target_cluster=target_cluster,
                success=False,
                intervention_name=None,
                original_cluster_score=original_score,
                validation=last_validation,
                failure_reason=reason,
                searched_candidates=searched_count,
                valid_candidates=valid_count,
            )

        winner = min(
            target_candidates,
            key=lambda candidate: (
                candidate.cost,
                candidate.magnitude,
                candidate.path.name,
            ),
        )
        return CounterfactualResult(
            sample_id=sample.sample_id,
            timestep=timestep,
            source_cluster=source_cluster,
            target_cluster=target_cluster,
            success=True,
            intervention_name=winner.path.name,
            changed_variables=winner.path.changed_variables(
                sample, winner.modified, timestep, winner.magnitude
            ),
            intervention_cost=winner.cost,
            magnitude=winner.magnitude,
            new_cluster=winner.cluster,
            original_cluster_score=original_score,
            target_cluster_score=winner.target_score,
            validation=winner.validation,
            counterfactual_state=np.asarray(winner.modified.states[timestep]),
            searched_candidates=searched_count,
            valid_candidates=valid_count,
        )


@dataclass(frozen=True)
class BoundaryAggregate:
    """Cluster-level profile of independently solved observations."""

    result_count: int
    success_count: int
    success_rate: float
    median_cost: float | None
    mean_cost: float | None
    intervention_counts: Mapping[str, int] = field(default_factory=dict)
    intervention_fractions: Mapping[str, float] = field(default_factory=dict)
    failure_counts: Mapping[str, int] = field(default_factory=dict)
    source_cluster: Any | None = None
    target_cluster: Any | None = None
    scope: str = REPRESENTATION_SCOPE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "intervention_counts", MappingProxyType(dict(self.intervention_counts))
        )
        object.__setattr__(
            self,
            "intervention_fractions",
            MappingProxyType(dict(self.intervention_fractions)),
        )
        object.__setattr__(
            self, "failure_counts", MappingProxyType(dict(self.failure_counts))
        )

    @classmethod
    def from_results(
        cls, results: Iterable[CounterfactualResult]
    ) -> "BoundaryAggregate":
        result_tuple = tuple(results)
        successes = tuple(result for result in result_tuple if result.valid)
        costs = np.asarray(
            [result.intervention_cost for result in successes], dtype=float
        )
        intervention_counts = Counter(
            result.intervention_name
            for result in successes
            if result.intervention_name is not None
        )
        success_count = len(successes)
        intervention_fractions = {
            name: count / success_count
            for name, count in intervention_counts.items()
        } if success_count else {}
        failure_counts = Counter(
            result.failure_reason
            for result in result_tuple
            if not result.valid and result.failure_reason is not None
        )

        source_values = {result.source_cluster for result in result_tuple}
        target_values = {result.target_cluster for result in result_tuple}
        source_cluster = next(iter(source_values)) if len(source_values) == 1 else None
        target_cluster = next(iter(target_values)) if len(target_values) == 1 else None
        return cls(
            result_count=len(result_tuple),
            success_count=success_count,
            success_rate=(success_count / len(result_tuple) if result_tuple else 0.0),
            median_cost=(float(np.median(costs)) if costs.size else None),
            mean_cost=(float(np.mean(costs)) if costs.size else None),
            intervention_counts=dict(intervention_counts),
            intervention_fractions=intervention_fractions,
            failure_counts=dict(failure_counts),
            source_cluster=source_cluster,
            target_cluster=target_cluster,
        )


__all__ = [
    "BoundaryAggregate",
    "CounterfactualResult",
    "CounterfactualSolver",
    "DistanceMetric",
    "InterventionPath",
    "LatentCoordinatePath",
    "StandardizedL2Distance",
]
