"""End-to-end orchestration for the minimum three-layer interpretation loop."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .boundary import (
    BoundaryAggregate,
    CounterfactualResult,
    CounterfactualSolver,
    DistanceMetric,
    LatentCoordinatePath,
    StandardizedL2Distance,
)
from .contrast import NearestCentroidRivalSelector, PairwiseAnalyzer
from .core import (
    BehaviorSample,
    ClusterAdapter,
    FeatureRegistry,
    ModelAdapter,
    ObservationTable,
)
from .perturbation import LatentCoordinateIntervention, PerturbationRunner
from .report import InterpreterReport
from .validation import Validator


_DEFAULT_LIMITATIONS = (
    "Pairwise differences are descriptive associations, not model dependence.",
    "The MVP intervention probes the fitted boundary in s_t representation space; "
    "it does not establish dependence of the upstream graph/TCN encoder.",
    "A change to s_t need not correspond to a physically realizable behavior; "
    "validity is limited to the configured representation constraints.",
    "No biological or semantic cluster label is generated automatically.",
)


def _latent_index(feature_name: str) -> int | None:
    prefix = "latent_"
    if not feature_name.startswith(prefix):
        return None
    suffix = feature_name[len(prefix) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


class InterpreterPipeline:
    """Run contrast, perturbation, and bounded boundary search in order.

    The default action mechanism is intentionally narrow: only a candidate
    named ``latent_<index>`` can be changed.  Readable decoder outputs may be
    compared in Layer 1, but they are not edited unless a future, explicitly
    invertible and validated intervention is registered for them.
    """

    def __init__(
        self,
        cluster_adapter: ClusterAdapter,
        *,
        feature_registry: FeatureRegistry | None = None,
        rival_selector: NearestCentroidRivalSelector | None = None,
        pairwise_analyzer: PairwiseAnalyzer | None = None,
        validator: Validator | None = None,
        distance_metric: DistanceMetric | None = None,
        perturbation_alpha: float = 1.0,
        counterfactual_grid_size: int = 21,
        counterfactual_bisection_steps: int = 12,
        max_counterfactuals: int | None = None,
    ) -> None:
        if not isinstance(cluster_adapter, ClusterAdapter):
            raise TypeError("cluster_adapter must implement ClusterAdapter")
        if not 0.0 <= float(perturbation_alpha) <= 1.0:
            raise ValueError("perturbation_alpha must lie in [0, 1]")
        if max_counterfactuals is not None and max_counterfactuals < 1:
            raise ValueError("max_counterfactuals must be positive when provided")
        self.cluster_adapter = cluster_adapter
        self.feature_registry = feature_registry or FeatureRegistry()
        self.rival_selector = rival_selector or NearestCentroidRivalSelector()
        self.pairwise_analyzer = pairwise_analyzer or PairwiseAnalyzer()
        self.validator = validator or Validator()
        self.distance_metric = distance_metric
        self.perturbation_alpha = float(perturbation_alpha)
        self.counterfactual_grid_size = int(counterfactual_grid_size)
        self.counterfactual_bisection_steps = int(counterfactual_bisection_steps)
        self.max_counterfactuals = max_counterfactuals

    def explain_batch(
        self,
        batch: Any,
        model_adapter: ModelAdapter,
        cluster_id: int,
        *,
        rival_cluster: int | None = None,
    ) -> InterpreterReport:
        """Encode one already-loaded batch, then explain ``cluster_id``."""

        if not isinstance(model_adapter, ModelAdapter):
            raise TypeError("model_adapter must implement ModelAdapter")
        return self.explain_cluster(
            model_adapter.encode(batch),
            cluster_id,
            rival_cluster=rival_cluster,
        )

    def explain_cluster(
        self,
        samples: Sequence[BehaviorSample],
        cluster_id: int,
        *,
        rival_cluster: int | None = None,
    ) -> InterpreterReport:
        """Explain one cluster while retaining full-trial provenance."""

        sample_tuple = tuple(samples)
        table = ObservationTable.from_samples(
            sample_tuple,
            self.cluster_adapter,
            self.feature_registry,
        )
        source_cluster = int(cluster_id)
        if not np.any(table.clusters == source_cluster):
            raise ValueError(f"cluster {source_cluster} has no valid observations")
        rival = (
            int(rival_cluster)
            if rival_cluster is not None
            else self.rival_selector.select(source_cluster, self.cluster_adapter)
        )
        if rival == source_cluster:
            raise ValueError("rival_cluster must differ from cluster_id")
        if not np.any(table.clusters == rival):
            raise ValueError(
                f"rival cluster {rival} has no observations in the supplied trials"
            )

        contrast = self.pairwise_analyzer.compare(table, source_cluster, rival)
        sample_by_id = {sample.sample_id: sample for sample in sample_tuple}
        ineligible = tuple(
            sample.sample_id
            for sample in sample_tuple
            if sample.metadata.get("training_eligible") is False
        )
        eligible_ids = set(sample_by_id) - set(ineligible)
        candidate_feature = next(
            (
                name
                for name in contrast.candidate_features
                if _latent_index(name) is not None
            ),
            None,
        )

        perturbation = None
        counterfactuals: tuple[CounterfactualResult, ...] = ()
        boundary = BoundaryAggregate.from_results(counterfactuals)
        limitations = list(_DEFAULT_LIMITATIONS)
        distance_reference_count = 0
        distance_scale_source: str | None = None

        if candidate_feature is None:
            limitations.append(
                "No latent coordinate exceeded the configured contrast threshold, so "
                "Layers 2 and 3 were not run."
            )
        else:
            coordinate = _latent_index(candidate_feature)
            assert coordinate is not None
            target_values = table.states[table.clusters == rival, coordinate]
            target_value = float(np.median(target_values))
            intervention = LatentCoordinateIntervention(
                index=coordinate,
                target_value=target_value,
                alpha=self.perturbation_alpha,
            )
            perturbation = PerturbationRunner(
                self.cluster_adapter,
                self.validator,
            ).run(
                sample_tuple,
                source_cluster=source_cluster,
                target_cluster=rival,
                intervention=intervention,
            )

            distance = self.distance_metric
            if distance is None:
                eligible_rows = np.asarray(
                    [ref.sample_id in eligible_ids for ref in table.refs], dtype=bool
                )
                eligible_states = table.states[eligible_rows]
                distance_reference_count = int(eligible_states.shape[0])
                if distance_reference_count >= 2:
                    distance = StandardizedL2Distance(
                        reference_states=eligible_states
                    )
                    distance_scale_source = "training_eligible_observations"
                else:
                    # Identity scaling keeps interface-only trials runnable without
                    # fitting any statistic to training-ineligible data.  The result
                    # remains an explicitly uncalibrated representation distance.
                    distance = StandardizedL2Distance(
                        scale=np.ones(table.states.shape[1], dtype=float)
                    )
                    distance_scale_source = "identity_scale_interface_only"
                    limitations.append(
                        "Fewer than two training-eligible observations were available; "
                        "boundary cost uses unstandardized L2 solely for interface "
                        "validation and is not comparable across representations."
                    )
            else:
                distance_scale_source = "caller_supplied_metric"
            solver = CounterfactualSolver(
                self.cluster_adapter,
                self.validator,
                distance,
                grid_size=self.counterfactual_grid_size,
                bisection_steps=self.counterfactual_bisection_steps,
            )
            path = LatentCoordinatePath(coordinate, target_value)
            source_refs = [
                ref
                for ref, assignment in zip(table.refs, table.clusters)
                if int(assignment) == source_cluster
            ]
            total_source_refs = len(source_refs)
            if self.max_counterfactuals is not None:
                source_refs = source_refs[: self.max_counterfactuals]
                if len(source_refs) < total_source_refs:
                    limitations.append(
                        "Boundary aggregation was deterministically truncated to "
                        f"the first {len(source_refs)} of {total_source_refs} source "
                        "observations."
                    )
            counterfactuals = tuple(
                solver.solve(
                    sample_by_id[ref.sample_id],
                    target_cluster=rival,
                    allowed_interventions=path,
                    timestep=ref.timestep,
                    source_cluster=source_cluster,
                )
                for ref in source_refs
            )
            boundary = BoundaryAggregate.from_results(counterfactuals)

        if ineligible:
            limitations.append(
                "Training-ineligible upstream trials are present; this report is "
                "appropriate for interface checks only and must not validate a trained "
                "behavioral interpretation."
            )

        return InterpreterReport(
            source_cluster=source_cluster,
            rival_cluster=rival,
            candidate_feature=candidate_feature,
            contrast=contrast,
            perturbation=perturbation,
            counterfactuals=counterfactuals,
            boundary=boundary,
            limitations=tuple(limitations),
            metadata={
                "trial_count": len(sample_tuple),
                "valid_observation_count": len(table),
                "source_observation_count": int(
                    np.count_nonzero(table.clusters == source_cluster)
                ),
                "rival_observation_count": int(
                    np.count_nonzero(table.clusters == rival)
                ),
                "rival_selection": (
                    "explicit"
                    if rival_cluster is not None
                    else "nearest_representative"
                ),
                "training_ineligible_trials": ineligible,
                "distance_reference_observation_count": distance_reference_count,
                "distance_scale_source": distance_scale_source,
            },
        )


__all__ = ["InterpreterPipeline"]
