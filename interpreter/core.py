"""Stable data and adapter interfaces for the behavioral interpreter.

The classes in this module deliberately start at the encoded, full-trial
boundary.  They know nothing about videos, alignment, or a particular graph
encoder.  This keeps interpretation experiments reusable while ensuring that
every observation can be traced back to its trial and social timestep.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]
FeatureExtractor = Callable[["BehaviorSample"], ArrayLike]


def _readonly_array(
    value: ArrayLike,
    *,
    dtype: Any | None = None,
) -> NDArray[Any]:
    """Return an owning, read-only array detached from caller-owned memory."""

    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _freeze_metadata(value: Any) -> Any:
    """Detach and recursively freeze common mutable metadata containers."""

    if isinstance(value, np.ndarray):
        return _readonly_array(value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_metadata(item) for item in value)
    return value


def _validate_matrix(states: ArrayLike, *, name: str = "states") -> FloatArray:
    array = np.asarray(states)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [M, D]; got {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty in both dimensions")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    numeric = np.array(array, dtype=float, copy=True)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain only finite values")
    return numeric


@dataclass(frozen=True)
class BehaviorSample:
    """An encoded full trial at the interpreter boundary.

    ``states`` and every contextual array retain the full padded trial length.
    ``time_mask`` is the only authority for deciding which timesteps are valid;
    consumers must never concatenate time across trials before applying it.
    """

    sample_id: str
    states: ArrayLike
    time_mask: ArrayLike
    features: Mapping[str, ArrayLike] = field(default_factory=dict)
    patch_start_s: ArrayLike | None = None
    patch_end_s: ArrayLike | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")

        raw_states = np.asarray(self.states)
        if raw_states.ndim != 2:
            raise ValueError(
                f"states must have shape [T, D]; got {raw_states.shape}"
            )
        if raw_states.shape[0] == 0 or raw_states.shape[1] == 0:
            raise ValueError("states must be non-empty in both dimensions")
        if not np.issubdtype(raw_states.dtype, np.number):
            raise TypeError("states must contain numeric values")
        states = _readonly_array(raw_states, dtype=float)

        raw_mask = np.asarray(self.time_mask)
        if raw_mask.ndim != 1 or raw_mask.shape[0] != states.shape[0]:
            raise ValueError(
                "time_mask must have shape [T] matching the states time axis"
            )
        if raw_mask.dtype != np.bool_:
            if not np.issubdtype(raw_mask.dtype, np.number):
                raise TypeError("time_mask must be boolean or contain only 0/1")
            if not np.all(np.isfinite(raw_mask)) or not np.all(
                np.isin(raw_mask, (0, 1))
            ):
                raise ValueError("time_mask must contain only boolean or 0/1 values")
        mask = _readonly_array(raw_mask, dtype=bool)
        if not np.any(mask):
            raise ValueError("time_mask must mark at least one valid timestep")
        if not np.all(np.isfinite(states[mask])):
            raise ValueError("states must be finite at valid timesteps")

        if not isinstance(self.features, Mapping):
            raise TypeError("features must be a mapping of name to [T] arrays")
        frozen_features: dict[str, NDArray[Any]] = {}
        for name, values in self.features.items():
            if not isinstance(name, str) or not name:
                raise ValueError("feature names must be non-empty strings")
            array = np.asarray(values)
            if array.ndim != 1 or array.shape[0] != states.shape[0]:
                raise ValueError(
                    f"feature {name!r} must have shape [T] matching states"
                )
            if not np.issubdtype(array.dtype, np.number):
                raise TypeError(f"feature {name!r} must contain numeric values")
            frozen_features[name] = _readonly_array(array, dtype=float)

        start = self._validate_clock(self.patch_start_s, "patch_start_s", states.shape[0])
        end = self._validate_clock(self.patch_end_s, "patch_end_s", states.shape[0])
        if start is not None and not np.all(np.isfinite(start[mask])):
            raise ValueError("patch_start_s must be finite at valid timesteps")
        if end is not None and not np.all(np.isfinite(end[mask])):
            raise ValueError("patch_end_s must be finite at valid timesteps")
        if start is not None and end is not None:
            if not np.all(end[mask] >= start[mask]):
                raise ValueError(
                    "patch_end_s must be greater than or equal to patch_start_s "
                    "at valid timesteps"
                )

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        object.__setattr__(self, "states", states)
        object.__setattr__(self, "time_mask", mask)
        object.__setattr__(self, "features", MappingProxyType(frozen_features))
        object.__setattr__(self, "patch_start_s", start)
        object.__setattr__(self, "patch_end_s", end)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @staticmethod
    def _validate_clock(
        values: ArrayLike | None,
        name: str,
        time_length: int,
    ) -> FloatArray | None:
        if values is None:
            return None
        array = np.asarray(values)
        if array.ndim != 1 or array.shape[0] != time_length:
            raise ValueError(f"{name} must have shape [T] matching states")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} must contain numeric values")
        return _readonly_array(array, dtype=float)

    def with_states(self, new_states: ArrayLike) -> "BehaviorSample":
        """Create a sample with new encoded states and unchanged trial context."""

        return BehaviorSample(
            sample_id=self.sample_id,
            states=new_states,
            time_mask=self.time_mask,
            features=self.features,
            patch_start_s=self.patch_start_s,
            patch_end_s=self.patch_end_s,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class ObservationRef:
    """Provenance for one valid social timestep."""

    sample_id: str
    timestep: int
    start_s: float | None
    end_s: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")
        if isinstance(self.timestep, bool) or not isinstance(
            self.timestep, (int, np.integer)
        ):
            raise TypeError("timestep must be an integer")
        if self.timestep < 0:
            raise ValueError("timestep must be non-negative")
        for name, value in (("start_s", self.start_s), ("end_s", self.end_s)):
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite when provided")
        if (
            self.start_s is not None
            and self.end_s is not None
            and self.end_s < self.start_s
        ):
            raise ValueError("end_s must be greater than or equal to start_s")


@dataclass(frozen=True)
class ClusterEvidence:
    """Continuous out-of-sample evidence for a set of cluster assignments."""

    values: ArrayLike
    cluster_ids: tuple[int, ...]
    kind: str
    larger_is_better: bool = True

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 2:
            raise ValueError(f"values must have shape [M, K]; got {values.shape}")
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("values must be non-empty in both dimensions")
        if len(self.cluster_ids) != values.shape[1]:
            raise ValueError("cluster_ids length must match the evidence columns")
        if any(
            isinstance(cluster_id, (bool, np.bool_))
            or not isinstance(cluster_id, (int, np.integer))
            for cluster_id in self.cluster_ids
        ):
            raise TypeError("cluster_ids must contain integer identifiers")
        normalized_ids = tuple(int(cluster_id) for cluster_id in self.cluster_ids)
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("cluster_ids must be unique")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("kind must be a non-empty string")
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("evidence values must be numeric")
        frozen_values = _readonly_array(values, dtype=float)
        if not np.all(np.isfinite(frozen_values)):
            raise ValueError("evidence values must be finite")
        object.__setattr__(self, "values", frozen_values)
        object.__setattr__(self, "cluster_ids", normalized_ids)

    def score_for(self, cluster_id: int) -> FloatArray:
        """Return the ``[M]`` evidence column for ``cluster_id``."""

        try:
            index = self.cluster_ids.index(int(cluster_id))
        except ValueError as exc:
            raise KeyError(f"no evidence is available for cluster {cluster_id}") from exc
        scores = self.values[:, index].view()
        scores.setflags(write=False)
        return scores

    @property
    def scores(self) -> FloatArray:
        """Alias for integrations that refer to evidence as scores."""

        return self.values

    @property
    def labels(self) -> tuple[int, ...]:
        """Alias for integrations that refer to cluster identifiers as labels."""

        return self.cluster_ids


class ModelAdapter(ABC):
    """Interface from an upstream batch to full-trial encoded states."""

    @abstractmethod
    def encode(self, batch: Any) -> tuple[BehaviorSample, ...]:
        """Encode a batch without dropping trial boundaries or ``time_mask``."""


class ClusterAdapter(ABC):
    """Out-of-sample assignment interface, independent of clusterer type."""

    @abstractmethod
    def predict(self, states: ArrayLike) -> IntArray:
        """Assign each row in ``states`` to a cluster."""

    def evidence(self, states: ArrayLike) -> ClusterEvidence | None:
        """Return optional continuous assignment evidence for each row."""

        return None

    @abstractmethod
    def representatives(self) -> Mapping[int, FloatArray]:
        """Return representative latent vectors keyed by cluster identifier."""


def _estimator_dimension(estimator: Any, representatives: NDArray[Any]) -> int:
    dimension = getattr(estimator, "n_features_in_", representatives.shape[1])
    try:
        normalized = int(dimension)
    except (TypeError, ValueError) as exc:
        raise ValueError("estimator has an invalid fitted feature dimension") from exc
    if normalized <= 0 or representatives.shape[1] != normalized:
        raise ValueError("estimator fitted dimension conflicts with representatives")
    return normalized


class SklearnKMeansAdapter(ClusterAdapter):
    """Adapter for a fitted scikit-learn KMeans-compatible estimator."""

    def __init__(self, estimator: Any) -> None:
        if not hasattr(estimator, "cluster_centers_"):
            raise ValueError("KMeans estimator must be fitted before adaptation")
        if not callable(getattr(estimator, "predict", None)) or not callable(
            getattr(estimator, "transform", None)
        ):
            raise TypeError("estimator must provide predict() and transform()")
        centers = np.asarray(estimator.cluster_centers_)
        if centers.ndim != 2 or centers.shape[0] == 0 or centers.shape[1] == 0:
            raise ValueError("cluster_centers_ must have shape [K, D]")
        if not np.issubdtype(centers.dtype, np.number) or not np.all(
            np.isfinite(centers)
        ):
            raise ValueError("cluster_centers_ must contain finite numeric values")
        self._estimator = estimator
        self._centers = _readonly_array(centers, dtype=float)
        self._dimension = _estimator_dimension(estimator, self._centers)

    def _inputs(self, states: ArrayLike) -> FloatArray:
        values = _validate_matrix(states)
        if values.shape[1] != self._dimension:
            raise ValueError(
                f"states feature dimension {values.shape[1]} does not match fitted "
                f"dimension {self._dimension}"
            )
        return values

    def predict(self, states: ArrayLike) -> IntArray:
        values = self._inputs(states)
        predictions = np.asarray(self._estimator.predict(values))
        if predictions.shape != (values.shape[0],):
            raise ValueError("estimator.predict() must return shape [M]")
        return _readonly_array(predictions, dtype=int)

    def evidence(self, states: ArrayLike) -> ClusterEvidence:
        values = self._inputs(states)
        distances = np.asarray(self._estimator.transform(values))
        expected = (values.shape[0], self._centers.shape[0])
        if distances.shape != expected:
            raise ValueError(f"estimator.transform() must return shape {expected}")
        return ClusterEvidence(
            values=-distances,
            cluster_ids=tuple(range(self._centers.shape[0])),
            kind="negative_euclidean_distance",
            larger_is_better=True,
        )

    def representatives(self) -> Mapping[int, FloatArray]:
        representatives = {
            cluster_id: _readonly_array(center, dtype=float)
            for cluster_id, center in enumerate(self._centers)
        }
        return MappingProxyType(representatives)


class SklearnGaussianMixtureAdapter(ClusterAdapter):
    """Adapter for a fitted scikit-learn GaussianMixture estimator."""

    def __init__(self, estimator: Any) -> None:
        if not hasattr(estimator, "means_"):
            raise ValueError("GaussianMixture estimator must be fitted before adaptation")
        if not callable(getattr(estimator, "predict", None)) or not callable(
            getattr(estimator, "predict_proba", None)
        ):
            raise TypeError("estimator must provide predict() and predict_proba()")
        means = np.asarray(estimator.means_)
        if means.ndim != 2 or means.shape[0] == 0 or means.shape[1] == 0:
            raise ValueError("means_ must have shape [K, D]")
        if not np.issubdtype(means.dtype, np.number) or not np.all(
            np.isfinite(means)
        ):
            raise ValueError("means_ must contain finite numeric values")
        self._estimator = estimator
        self._means = _readonly_array(means, dtype=float)
        self._dimension = _estimator_dimension(estimator, self._means)

    def _inputs(self, states: ArrayLike) -> FloatArray:
        values = _validate_matrix(states)
        if values.shape[1] != self._dimension:
            raise ValueError(
                f"states feature dimension {values.shape[1]} does not match fitted "
                f"dimension {self._dimension}"
            )
        return values

    def predict(self, states: ArrayLike) -> IntArray:
        values = self._inputs(states)
        predictions = np.asarray(self._estimator.predict(values))
        if predictions.shape != (values.shape[0],):
            raise ValueError("estimator.predict() must return shape [M]")
        return _readonly_array(predictions, dtype=int)

    def evidence(self, states: ArrayLike) -> ClusterEvidence:
        values = self._inputs(states)
        probabilities = np.asarray(self._estimator.predict_proba(values))
        expected = (values.shape[0], self._means.shape[0])
        if probabilities.shape != expected:
            raise ValueError(f"estimator.predict_proba() must return shape {expected}")
        return ClusterEvidence(
            values=probabilities,
            cluster_ids=tuple(range(self._means.shape[0])),
            kind="membership_probability",
            larger_is_better=True,
        )

    def representatives(self) -> Mapping[int, FloatArray]:
        representatives = {
            cluster_id: _readonly_array(mean, dtype=float)
            for cluster_id, mean in enumerate(self._means)
        }
        return MappingProxyType(representatives)


class FeatureRegistry:
    """Build scalar, timestep-aligned features for contrast analysis.

    Sample-provided features, registered extractors, and optional latent
    dimensions are merged without silently overwriting duplicate names.
    """

    def __init__(
        self,
        extractors: Mapping[str, FeatureExtractor] | None = None,
        *,
        include_latent_dimensions: bool = True,
    ) -> None:
        self.include_latent_dimensions = bool(include_latent_dimensions)
        self._extractors: dict[str, FeatureExtractor] = {}
        if extractors is not None:
            for name, extractor in extractors.items():
                self.register(name, extractor)

    def register(self, name: str, extractor: FeatureExtractor) -> "FeatureRegistry":
        """Register a scalar ``[T]`` extractor and return this registry."""

        if not isinstance(name, str) or not name:
            raise ValueError("feature name must be a non-empty string")
        if not callable(extractor):
            raise TypeError("extractor must be callable")
        if name in self._extractors:
            raise ValueError(f"an extractor named {name!r} is already registered")
        self._extractors[name] = extractor
        return self

    def extract(self, sample: BehaviorSample) -> Mapping[str, FloatArray]:
        """Return validated scalar features aligned to the sample's time axis."""

        if not isinstance(sample, BehaviorSample):
            raise TypeError("sample must be a BehaviorSample")
        combined: dict[str, ArrayLike] = dict(sample.features)
        for name, extractor in self._extractors.items():
            if name in combined:
                raise ValueError(f"duplicate feature name {name!r}")
            combined[name] = extractor(sample)
        if self.include_latent_dimensions:
            for dimension in range(sample.states.shape[1]):
                name = f"latent_{dimension}"
                if name in combined:
                    raise ValueError(f"duplicate feature name {name!r}")
                combined[name] = sample.states[:, dimension]

        validated: dict[str, FloatArray] = {}
        time_length = sample.states.shape[0]
        for name, values in combined.items():
            array = np.asarray(values)
            if array.ndim != 1 or array.shape[0] != time_length:
                raise ValueError(
                    f"feature {name!r} must be scalar with shape [T]; got "
                    f"{array.shape}"
                )
            if not np.issubdtype(array.dtype, np.number):
                raise TypeError(f"feature {name!r} must contain numeric values")
            frozen = _readonly_array(array, dtype=float)
            if not np.all(np.isfinite(frozen[sample.time_mask])):
                raise ValueError(
                    f"feature {name!r} must be finite at valid timesteps"
                )
            validated[name] = frozen
        return MappingProxyType(validated)

    # ``build`` is a readable alias for callers that treat this as a builder.
    build = extract


@dataclass(frozen=True)
class ObservationTable:
    """Valid observations, their clusters, readable features, and provenance."""

    refs: tuple[ObservationRef, ...]
    states: ArrayLike
    clusters: ArrayLike
    features: Mapping[str, ArrayLike]
    evidence: ClusterEvidence | None = None

    def __post_init__(self) -> None:
        states = _validate_matrix(self.states)
        observation_count = states.shape[0]
        refs = tuple(self.refs)
        if len(refs) != observation_count or not all(
            isinstance(ref, ObservationRef) for ref in refs
        ):
            raise ValueError(
                "refs must contain one ObservationRef per state observation"
            )

        raw_clusters = np.asarray(self.clusters)
        if raw_clusters.ndim != 1 or raw_clusters.shape[0] != observation_count:
            raise ValueError("clusters must have shape [M] matching states")
        if not np.issubdtype(raw_clusters.dtype, np.integer):
            raise TypeError("clusters must contain integer identifiers")
        clusters = _readonly_array(raw_clusters, dtype=int)

        if not isinstance(self.features, Mapping):
            raise TypeError("features must be a mapping")
        frozen_features: dict[str, FloatArray] = {}
        for name, values in self.features.items():
            if not isinstance(name, str) or not name:
                raise ValueError("feature names must be non-empty strings")
            array = np.asarray(values)
            if array.ndim != 1 or array.shape[0] != observation_count:
                raise ValueError(f"feature {name!r} must have shape [M]")
            if not np.issubdtype(array.dtype, np.number):
                raise TypeError(f"feature {name!r} must contain numeric values")
            frozen = _readonly_array(array, dtype=float)
            if not np.all(np.isfinite(frozen)):
                raise ValueError(f"feature {name!r} must contain finite values")
            frozen_features[name] = frozen

        if self.evidence is not None:
            if not isinstance(self.evidence, ClusterEvidence):
                raise TypeError("evidence must be ClusterEvidence or None")
            if self.evidence.values.shape[0] != observation_count:
                raise ValueError("evidence rows must match states observations")

        object.__setattr__(self, "refs", refs)
        states.setflags(write=False)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "clusters", clusters)
        object.__setattr__(self, "features", MappingProxyType(frozen_features))

    def __len__(self) -> int:
        return self.states.shape[0]

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[BehaviorSample],
        cluster_adapter: ClusterAdapter,
        feature_registry: FeatureRegistry,
    ) -> "ObservationTable":
        """Flatten valid timesteps only, retaining deterministic trial provenance."""

        sample_tuple = tuple(samples)
        if not sample_tuple:
            raise ValueError("samples must contain at least one full trial")
        if not all(isinstance(sample, BehaviorSample) for sample in sample_tuple):
            raise TypeError("every sample must be a BehaviorSample")
        sample_ids = [sample.sample_id for sample in sample_tuple]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample_id values must be unique within an observation table")
        if not isinstance(cluster_adapter, ClusterAdapter):
            raise TypeError("cluster_adapter must implement ClusterAdapter")
        if not isinstance(feature_registry, FeatureRegistry):
            raise TypeError("feature_registry must be a FeatureRegistry")

        state_dimension = sample_tuple[0].states.shape[1]
        refs: list[ObservationRef] = []
        state_parts: list[FloatArray] = []
        feature_parts: dict[str, list[FloatArray]] = {}
        expected_feature_names: tuple[str, ...] | None = None

        for sample in sample_tuple:
            if sample.states.shape[1] != state_dimension:
                raise ValueError("all samples must have the same state dimension")
            valid_indices = np.flatnonzero(sample.time_mask)
            state_parts.append(np.asarray(sample.states[valid_indices], dtype=float))

            extracted = feature_registry.extract(sample)
            names = tuple(extracted)
            if expected_feature_names is None:
                expected_feature_names = names
                feature_parts = {name: [] for name in names}
            elif set(names) != set(expected_feature_names):
                missing = sorted(set(expected_feature_names) - set(names))
                extra = sorted(set(names) - set(expected_feature_names))
                raise ValueError(
                    "all samples must expose the same feature names; "
                    f"missing={missing}, extra={extra}"
                )
            for name in expected_feature_names:
                feature_parts[name].append(
                    np.asarray(extracted[name][valid_indices], dtype=float)
                )

            for timestep in valid_indices:
                start_s = (
                    None
                    if sample.patch_start_s is None
                    else float(sample.patch_start_s[timestep])
                )
                end_s = (
                    None
                    if sample.patch_end_s is None
                    else float(sample.patch_end_s[timestep])
                )
                refs.append(
                    ObservationRef(
                        sample_id=sample.sample_id,
                        timestep=int(timestep),
                        start_s=start_s,
                        end_s=end_s,
                    )
                )

        states = np.concatenate(state_parts, axis=0)
        clusters = np.asarray(cluster_adapter.predict(states))
        if clusters.ndim != 1 or clusters.shape[0] != states.shape[0]:
            raise ValueError("cluster_adapter.predict() must return shape [M]")
        if not np.issubdtype(clusters.dtype, np.integer):
            raise TypeError("cluster_adapter.predict() must return integer identifiers")
        evidence = cluster_adapter.evidence(states)
        if evidence is not None and evidence.values.shape[0] != states.shape[0]:
            raise ValueError("cluster_adapter.evidence() must contain M rows")
        flattened_features = {
            name: np.concatenate(parts, axis=0)
            for name, parts in feature_parts.items()
        }
        return cls(
            refs=tuple(refs),
            states=states,
            clusters=clusters,
            features=flattened_features,
            evidence=evidence,
        )


__all__ = [
    "BehaviorSample",
    "ClusterAdapter",
    "ClusterEvidence",
    "FeatureRegistry",
    "ModelAdapter",
    "ObservationRef",
    "ObservationTable",
    "SklearnGaussianMixtureAdapter",
    "SklearnKMeansAdapter",
]
