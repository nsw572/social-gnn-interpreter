"""Descriptive, pairwise contrast analysis for behavioral clusters.

This layer reports observable associations between a source cluster and a
specific rival.  Its outputs are hypotheses for intervention experiments, not
claims about model dependence or biological causality.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import ClusterAdapter, ObservationTable


FloatArray = NDArray[np.floating[Any]]


class NearestCentroidRivalSelector:
    """Choose the non-source cluster with the nearest latent representative."""

    def select(
        self,
        cluster_id: int,
        context: Mapping[int, ArrayLike] | ClusterAdapter | ObservationTable,
    ) -> int:
        """Return a deterministic nearest-centroid rival.

        ``context`` may be an adapter, a representative mapping, or an
        observation table (whose empirical cluster means are then used).
        """

        source_id = int(cluster_id)
        representatives = self._representatives(context)
        if source_id not in representatives:
            raise KeyError(f"source cluster {source_id} has no representative")
        if len(representatives) < 2:
            raise ValueError("at least two cluster representatives are required")

        source = self._vector(representatives[source_id], source_id)
        candidates: list[tuple[float, int]] = []
        for candidate_id, raw_vector in representatives.items():
            normalized_id = int(candidate_id)
            if normalized_id == source_id:
                continue
            vector = self._vector(raw_vector, normalized_id)
            if vector.shape != source.shape:
                raise ValueError("all cluster representatives must share one dimension")
            candidates.append((float(np.linalg.norm(source - vector)), normalized_id))
        if not candidates:
            raise ValueError("no rival cluster is available")
        # The cluster id tie-break makes selection stable across mapping order.
        return min(candidates, key=lambda item: (item[0], item[1]))[1]

    @staticmethod
    def _vector(value: ArrayLike, cluster_id: int) -> FloatArray:
        vector = np.asarray(value, dtype=float)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(
                f"representative for cluster {cluster_id} must have shape [D]"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"representative for cluster {cluster_id} must be finite"
            )
        return vector

    @staticmethod
    def _representatives(
        context: Mapping[int, ArrayLike] | ClusterAdapter | ObservationTable,
    ) -> Mapping[int, ArrayLike]:
        if isinstance(context, ClusterAdapter):
            return context.representatives()
        if isinstance(context, ObservationTable):
            representatives: dict[int, FloatArray] = {}
            for cluster_id in np.unique(context.clusters):
                mask = context.clusters == cluster_id
                representatives[int(cluster_id)] = np.mean(
                    context.states[mask], axis=0
                )
            return representatives
        if isinstance(context, Mapping):
            return context
        raise TypeError(
            "context must be a representative mapping, ClusterAdapter, or "
            "ObservationTable"
        )


@dataclass(frozen=True)
class FeatureDifference:
    """Transparent descriptive statistics for one pairwise feature contrast.

    Statistics describe the analysis units named by ``aggregation``.  By
    default, each unit is a within-trial median for the selected cluster.
    ``standardized_difference`` is Cohen's d with the sign ``A - B``.
    """

    feature: str
    mean_a: float
    mean_b: float
    median_a: float
    median_b: float
    std_a: float
    std_b: float
    difference: float
    standardized_difference: float
    n_observations_a: int
    n_observations_b: int
    n_trials_a: int
    n_trials_b: int
    aggregation: str

    @property
    def absolute_effect(self) -> float:
        return abs(self.standardized_difference)

    @property
    def effect_size(self) -> float:
        """Alias naming the reported standardized difference as an effect size."""

        return self.standardized_difference

    @property
    def name(self) -> str:
        """Short alias for integrations that use ``name`` for feature keys."""

        return self.feature


@dataclass(frozen=True)
class ContrastResult:
    """Structured evidence for a descriptive A-versus-B comparison."""

    cluster_a: int
    cluster_b: int
    feature_differences: Mapping[str, FeatureDifference]
    candidate_features: tuple[str, ...]
    min_abs_standardized_difference: float
    scope: str = "descriptive_association_only"

    def __post_init__(self) -> None:
        differences = dict(self.feature_differences)
        if not differences:
            raise ValueError("feature_differences must not be empty")
        if not all(
            isinstance(name, str) and isinstance(value, FeatureDifference)
            for name, value in differences.items()
        ):
            raise TypeError(
                "feature_differences must map names to FeatureDifference objects"
            )
        candidates = tuple(self.candidate_features)
        unknown = [name for name in candidates if name not in differences]
        if unknown:
            raise ValueError(f"candidate features are missing statistics: {unknown}")
        object.__setattr__(
            self, "feature_differences", MappingProxyType(differences)
        )
        object.__setattr__(self, "candidate_features", candidates)

    @property
    def differences(self) -> Mapping[str, FeatureDifference]:
        """Alias for the full feature-statistics mapping."""

        return self.feature_differences


class PairwiseAnalyzer:
    """Compare readable features for two clusters with transparent statistics."""

    def __init__(self, min_abs_standardized_difference: float = 0.8) -> None:
        threshold = float(min_abs_standardized_difference)
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError(
                "min_abs_standardized_difference must be finite and non-negative"
            )
        self.min_abs_standardized_difference = threshold

    def compare(
        self,
        table: ObservationTable,
        cluster_a: int,
        cluster_b: int,
        *,
        aggregate_by_trial: bool = True,
    ) -> ContrastResult:
        """Compare A with B, using within-trial medians by default.

        Trial aggregation limits long trials from dominating the contrast.  It
        does not make observations independent and it does not establish model
        dependence or causality.
        """

        if not isinstance(table, ObservationTable):
            raise TypeError("table must be an ObservationTable")
        source = int(cluster_a)
        rival = int(cluster_b)
        if source == rival:
            raise ValueError("cluster_a and cluster_b must differ")
        mask_a = table.clusters == source
        mask_b = table.clusters == rival
        if not np.any(mask_a):
            raise ValueError(f"cluster {source} has no observations")
        if not np.any(mask_b):
            raise ValueError(f"cluster {rival} has no observations")
        if not table.features:
            raise ValueError("table has no features to compare")

        differences: dict[str, FeatureDifference] = {}
        for name, values in table.features.items():
            units_a = self._analysis_units(table, values, mask_a, aggregate_by_trial)
            units_b = self._analysis_units(table, values, mask_b, aggregate_by_trial)
            mean_a = float(np.mean(units_a))
            mean_b = float(np.mean(units_b))
            std_a = self._sample_std(units_a)
            std_b = self._sample_std(units_b)
            difference = mean_a - mean_b
            effect = self._cohens_d(
                difference, std_a, std_b, units_a.size, units_b.size
            )
            differences[name] = FeatureDifference(
                feature=name,
                mean_a=mean_a,
                mean_b=mean_b,
                median_a=float(np.median(units_a)),
                median_b=float(np.median(units_b)),
                std_a=std_a,
                std_b=std_b,
                difference=difference,
                standardized_difference=effect,
                n_observations_a=int(np.count_nonzero(mask_a)),
                n_observations_b=int(np.count_nonzero(mask_b)),
                n_trials_a=self._trial_count(table, mask_a),
                n_trials_b=self._trial_count(table, mask_b),
                aggregation=("per_trial_median" if aggregate_by_trial else "observation"),
            )

        candidates = [
            name
            for name, difference in differences.items()
            if difference.absolute_effect
            >= self.min_abs_standardized_difference
        ]
        candidates.sort(
            key=lambda name: (-differences[name].absolute_effect, name)
        )
        return ContrastResult(
            cluster_a=source,
            cluster_b=rival,
            feature_differences=differences,
            candidate_features=tuple(candidates),
            min_abs_standardized_difference=self.min_abs_standardized_difference,
        )

    @staticmethod
    def _analysis_units(
        table: ObservationTable,
        values: ArrayLike,
        cluster_mask: NDArray[np.bool_],
        aggregate_by_trial: bool,
    ) -> FloatArray:
        numeric = np.asarray(values, dtype=float)
        if not aggregate_by_trial:
            return numeric[cluster_mask]

        grouped: dict[str, list[float]] = {}
        for index in np.flatnonzero(cluster_mask):
            sample_id = table.refs[int(index)].sample_id
            grouped.setdefault(sample_id, []).append(float(numeric[index]))
        # Dict insertion follows observation order, preserving trial ordering.
        return np.asarray(
            [np.median(trial_values) for trial_values in grouped.values()],
            dtype=float,
        )

    @staticmethod
    def _trial_count(
        table: ObservationTable,
        cluster_mask: NDArray[np.bool_],
    ) -> int:
        return len(
            {
                table.refs[int(index)].sample_id
                for index in np.flatnonzero(cluster_mask)
            }
        )

    @staticmethod
    def _sample_std(values: FloatArray) -> float:
        return float(np.std(values, ddof=1)) if values.size > 1 else 0.0

    @staticmethod
    def _cohens_d(
        difference: float,
        std_a: float,
        std_b: float,
        count_a: int,
        count_b: int,
    ) -> float:
        degrees = count_a + count_b - 2
        if degrees > 0:
            pooled_variance = (
                (count_a - 1) * std_a**2 + (count_b - 1) * std_b**2
            ) / degrees
        else:
            pooled_variance = 0.0
        if pooled_variance <= np.finfo(float).eps:
            if abs(difference) <= np.finfo(float).eps:
                return 0.0
            return float(np.copysign(np.inf, difference))
        return float(difference / np.sqrt(pooled_variance))


__all__ = [
    "ContrastResult",
    "FeatureDifference",
    "NearestCentroidRivalSelector",
    "PairwiseAnalyzer",
]
