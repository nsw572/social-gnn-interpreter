"""Validation gates for representation-space interventions.

The validator deliberately contains no hand-written biological thresholds.  The
default constraints enforce structural invariants at the Social-GNN boundary;
an optional empirical-support constraint derives its threshold from reference
representations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

from .core import BehaviorSample


def _readonly_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({} if value is None else dict(value))


def _array_equal(left: Any, right: Any) -> bool:
    """Equality helper that is safe for nested mappings and NumPy arrays."""

    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(_array_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        if not isinstance(left, (tuple, list)) or not isinstance(right, (tuple, list)):
            return False
        return len(left) == len(right) and all(
            _array_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True))
        except (TypeError, ValueError):
            return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    if isinstance(left, (float, np.floating)) and isinstance(
        right, (float, np.floating)
    ):
        if np.isnan(left) and np.isnan(right):
            return True
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(equal, np.ndarray):
        return bool(np.all(equal))
    try:
        return bool(equal)
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class ConstraintResult:
    """Outcome of one independently inspectable validity constraint."""

    name: str
    valid: bool
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", _readonly_mapping(self.details))


@dataclass(frozen=True)
class ValidationResult:
    """Aggregate result returned before a modified state reaches the model."""

    valid: bool
    constraint_results: tuple[ConstraintResult, ...]

    @property
    def results(self) -> tuple[ConstraintResult, ...]:
        """Short alias useful in interactive inspection."""

        return self.constraint_results

    @property
    def details(self) -> Mapping[str, Mapping[str, Any]]:
        return MappingProxyType(
            {result.name: result.details for result in self.constraint_results}
        )

    def __bool__(self) -> bool:
        return self.valid


class Constraint(ABC):
    """A validity rule for one proposed single-timestep intervention."""

    name = "constraint"

    @abstractmethod
    def evaluate(
        self,
        original: BehaviorSample,
        modified: BehaviorSample,
        timestep: int,
    ) -> ConstraintResult:
        """Evaluate the proposed change without mutating either sample."""

    def is_valid(
        self,
        original: BehaviorSample,
        modified: BehaviorSample,
        timestep: int,
    ) -> bool:
        return self.evaluate(original, modified, timestep).valid


class Validator:
    """Run all constraints and retain every result, including failures."""

    def __init__(self, constraints: Iterable[Constraint] | None = None) -> None:
        if constraints is None:
            constraints = (
                FiniteStateConstraint(),
                ImmutableContextConstraint(),
                SingleTimestepConstraint(),
            )
        self.constraints = tuple(constraints)
        if not all(isinstance(constraint, Constraint) for constraint in self.constraints):
            raise TypeError("constraints must contain Constraint instances")

    def evaluate(
        self,
        original: BehaviorSample,
        modified: BehaviorSample,
        timestep: int,
    ) -> ValidationResult:
        results = tuple(
            constraint.evaluate(original, modified, timestep)
            for constraint in self.constraints
        )
        return ValidationResult(
            valid=all(result.valid for result in results),
            constraint_results=results,
        )

    validate = evaluate


class FiniteStateConstraint(Constraint):
    """Reject NaN and infinite values at valid (non-padding) timesteps."""

    name = "finite_state"

    def evaluate(
        self,
        original: BehaviorSample,
        modified: BehaviorSample,
        timestep: int,
    ) -> ConstraintResult:
        del original, timestep
        states = np.asarray(modified.states)
        try:
            mask = np.asarray(modified.time_mask, dtype=bool)
            if states.ndim < 1 or mask.ndim != 1 or states.shape[0] != mask.size:
                return ConstraintResult(
                    self.name,
                    False,
                    {"reason": "states and time_mask have incompatible shapes"},
                )
            finite = np.isfinite(states[mask])
            valid = bool(np.all(finite))
            invalid_count = int(finite.size - np.count_nonzero(finite))
            invalid_indices = np.argwhere(~finite)[:10].tolist()
            details: Mapping[str, Any] = {
                "invalid_value_count": invalid_count,
                "invalid_indices_within_valid_rows": invalid_indices,
            }
        except TypeError:
            valid = False
            details = {"reason": "state tensor is not numeric"}
        return ConstraintResult(self.name, valid, details)


class ImmutableContextConstraint(Constraint):
    """Protect trial identity, masks, timestamps, and readable context."""

    name = "immutable_context"

    def evaluate(
        self,
        original: BehaviorSample,
        modified: BehaviorSample,
        timestep: int,
    ) -> ConstraintResult:
        del timestep
        original_states = np.asarray(original.states)
        modified_states = np.asarray(modified.states)
        original_mask = np.asarray(original.time_mask)
        modified_mask = np.asarray(modified.time_mask)

        checks = {
            "sample_id_unchanged": original.sample_id == modified.sample_id,
            "state_shape_unchanged": original_states.shape == modified_states.shape,
            "time_mask_shape_unchanged": original_mask.shape == modified_mask.shape,
            "time_mask_unchanged": _array_equal(original_mask, modified_mask),
            "patch_start_s_unchanged": _array_equal(
                original.patch_start_s, modified.patch_start_s
            ),
            "patch_end_s_unchanged": _array_equal(
                original.patch_end_s, modified.patch_end_s
            ),
            "features_unchanged": _array_equal(original.features, modified.features),
            "metadata_unchanged": _array_equal(original.metadata, modified.metadata),
        }
        time_dimension_matches = (
            original_states.ndim >= 1
            and modified_states.ndim >= 1
            and original_mask.ndim == 1
            and modified_mask.ndim == 1
            and original_states.shape[0] == original_mask.shape[0]
            and modified_states.shape[0] == modified_mask.shape[0]
        )
        checks["state_mask_time_dimension_matches"] = time_dimension_matches
        failed = tuple(key for key, value in checks.items() if not value)
        return ConstraintResult(
            self.name,
            not failed,
            {"checks": checks, "failed_checks": failed},
        )


class SingleTimestepConstraint(Constraint):
    """Allow changes only at one selected, non-padding social timestep."""

    name = "single_timestep"

    def evaluate(
        self,
        original: BehaviorSample,
        modified: BehaviorSample,
        timestep: int,
    ) -> ConstraintResult:
        original_states = np.asarray(original.states)
        modified_states = np.asarray(modified.states)
        mask = np.asarray(original.time_mask, dtype=bool)

        if original_states.shape != modified_states.shape or original_states.ndim < 1:
            return ConstraintResult(
                self.name,
                False,
                {"reason": "state shapes differ or lack a time dimension"},
            )
        if mask.ndim != 1 or mask.shape[0] != original_states.shape[0]:
            return ConstraintResult(
                self.name,
                False,
                {"reason": "time_mask must have shape [T] matching states"},
            )
        if not isinstance(timestep, (int, np.integer)):
            return ConstraintResult(
                self.name, False, {"reason": "timestep must be an integer"}
            )
        timestep = int(timestep)
        if timestep < 0 or timestep >= mask.size:
            return ConstraintResult(
                self.name, False, {"reason": "timestep is out of bounds"}
            )
        if not mask[timestep]:
            return ConstraintResult(
                self.name, False, {"reason": "selected timestep is padding"}
            )

        equal_rows = np.asarray(
            [
                _array_equal(original_states[index], modified_states[index])
                for index in range(mask.size)
            ],
            dtype=bool,
        )
        disallowed = np.ones(mask.size, dtype=bool)
        disallowed[timestep] = False
        changed_elsewhere = np.flatnonzero(~equal_rows & disallowed)
        changed_padding = np.flatnonzero(~equal_rows & ~mask)
        valid = changed_elsewhere.size == 0 and changed_padding.size == 0
        return ConstraintResult(
            self.name,
            valid,
            {
                "selected_timestep": timestep,
                "selected_changed": bool(not equal_rows[timestep]),
                "changed_outside_selected": changed_elsewhere.tolist(),
                "changed_padding": changed_padding.tolist(),
            },
        )


class EmpiricalSupportConstraint(Constraint):
    """Reject states beyond a data-derived nearest-neighbour distance.

    Reference representations are standardized per coordinate.  The support
    threshold is the configured quantile of leave-one-out nearest-neighbour
    distances in the reference data; no domain or biological limit is baked
    into this class.
    """

    name = "empirical_support"

    def __init__(self, quantile: float = 0.99, *, chunk_size: int = 1024) -> None:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must lie in [0, 1]")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.quantile = float(quantile)
        self.chunk_size = int(chunk_size)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.reference_: np.ndarray | None = None
        self.threshold_: float | None = None

    @staticmethod
    def _as_matrix(states: Any) -> np.ndarray:
        matrix = np.asarray(states, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("reference_states must have shape [observations, dimensions]")
        if matrix.shape[0] < 2 or matrix.shape[1] < 1:
            raise ValueError("at least two non-empty reference states are required")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("reference_states must be finite")
        return matrix

    def _nearest_distances(
        self,
        query: np.ndarray,
        reference: np.ndarray,
        *,
        aligned_self: bool,
    ) -> np.ndarray:
        output = np.empty(query.shape[0], dtype=float)
        for start in range(0, query.shape[0], self.chunk_size):
            stop = min(start + self.chunk_size, query.shape[0])
            block = query[start:stop]
            differences = block[:, None, :] - reference[None, :, :]
            squared = np.einsum("ijk,ijk->ij", differences, differences)
            if aligned_self:
                rows = np.arange(stop - start)
                squared[rows, start + rows] = np.inf
            output[start:stop] = np.sqrt(np.min(squared, axis=1))
        return output

    def fit(self, reference_states: Any) -> "EmpiricalSupportConstraint":
        matrix = self._as_matrix(reference_states)
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
        standardized = (matrix - mean) / scale
        nearest = self._nearest_distances(
            standardized, standardized, aligned_self=True
        )
        self.mean_ = mean
        self.scale_ = scale
        self.reference_ = standardized
        self.threshold_ = float(np.quantile(nearest, self.quantile))
        return self

    def evaluate(
        self,
        original: BehaviorSample,
        modified: BehaviorSample,
        timestep: int,
    ) -> ConstraintResult:
        del original
        if (
            self.mean_ is None
            or self.scale_ is None
            or self.reference_ is None
            or self.threshold_ is None
        ):
            raise RuntimeError("EmpiricalSupportConstraint must be fitted before use")
        states = np.asarray(modified.states, dtype=float)
        if states.ndim != 2 or not 0 <= int(timestep) < states.shape[0]:
            return ConstraintResult(
                self.name, False, {"reason": "states/timestep shape mismatch"}
            )
        candidate = states[int(timestep)]
        if candidate.shape != self.mean_.shape or not np.all(np.isfinite(candidate)):
            return ConstraintResult(
                self.name,
                False,
                {"reason": "candidate is non-finite or has the wrong dimension"},
            )
        standardized = ((candidate - self.mean_) / self.scale_)[None, :]
        distance = float(
            self._nearest_distances(
                standardized, self.reference_, aligned_self=False
            )[0]
        )
        return ConstraintResult(
            self.name,
            distance <= self.threshold_,
            {
                "nearest_standardized_distance": distance,
                "threshold": self.threshold_,
                "quantile": self.quantile,
                "reference_count": int(self.reference_.shape[0]),
            },
        )


__all__ = [
    "Constraint",
    "ConstraintResult",
    "EmpiricalSupportConstraint",
    "FiniteStateConstraint",
    "ImmutableContextConstraint",
    "SingleTimestepConstraint",
    "ValidationResult",
    "Validator",
]
