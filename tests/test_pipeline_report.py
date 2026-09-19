from __future__ import annotations

import json
from types import MappingProxyType
import unittest

import numpy as np

from interpreter.boundary import BoundaryAggregate, CounterfactualResult
from interpreter.contrast import PairwiseAnalyzer
from interpreter.core import BehaviorSample, ClusterAdapter, ClusterEvidence
from interpreter.pipeline import InterpreterPipeline
from interpreter.report import InterpreterReport, TextReportRenderer


class ThresholdClusterAdapter(ClusterAdapter):
    """Deterministic test-only boundary: latent_0 < 0 versus >= 0."""

    def predict(self, states):
        values = np.asarray(states, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("test adapter expects states with shape [M,2]")
        result = (values[:, 0] >= 0.0).astype(int)
        result.setflags(write=False)
        return result

    def evidence(self, states):
        values = np.asarray(states, dtype=float)
        centers = np.asarray([[-1.0, 0.0], [1.0, 0.0]])
        scores = -np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2)
        return ClusterEvidence(
            values=scores,
            cluster_ids=(0, 1),
            kind="negative_euclidean_distance",
            larger_is_better=True,
        )

    def representatives(self):
        return MappingProxyType(
            {
                0: np.asarray([-1.0, 0.0]),
                1: np.asarray([1.0, 0.0]),
            }
        )


def _trial(sample_id: str, latent_0: tuple[float, float]) -> BehaviorSample:
    # The last row intentionally belongs to the opposite cluster.  Its false
    # mask makes padding exclusion observable throughout the end-to-end test.
    padded_value = 100.0 if max(latent_0) < 0.0 else -100.0
    return BehaviorSample(
        sample_id=sample_id,
        states=np.asarray(
            [
                [latent_0[0], 0.10],
                [latent_0[1], -0.10],
                [padded_value, 99.0],
            ],
            dtype=float,
        ),
        time_mask=np.asarray([True, True, False]),
        patch_start_s=np.asarray([0.0, 1.0, 0.0]),
        patch_end_s=np.asarray([1.0, 2.0, 0.0]),
        metadata={"trial_group": sample_id.split("-")[0]},
    )


def _samples() -> tuple[BehaviorSample, ...]:
    # Two independent full trials per cluster.  Slightly different trial
    # medians keep the standardized contrast finite and JSON-representable.
    return (
        _trial("source-1", (-1.30, -1.10)),
        _trial("source-2", (-0.90, -0.80)),
        _trial("rival-1", (0.75, 0.90)),
        _trial("rival-2", (1.05, 1.20)),
    )


class PipelineReportTests(unittest.TestCase):
    def test_three_layer_pipeline_boundary_and_reports(self):
        pipeline = InterpreterPipeline(
            ThresholdClusterAdapter(),
            counterfactual_grid_size=21,
            counterfactual_bisection_steps=14,
        )
        report = pipeline.explain_cluster(_samples(), cluster_id=0)

        self.assertIsInstance(report, InterpreterReport)
        self.assertEqual(report.source_cluster, 0)
        self.assertEqual(report.rival_cluster, 1)
        self.assertEqual(report.candidate_feature, "latent_0")
        self.assertEqual(report.contrast.candidate_features[0], "latent_0")
        self.assertEqual(
            report.contrast.feature_differences["latent_0"].n_trials_a, 2
        )
        self.assertEqual(
            report.contrast.feature_differences["latent_0"].n_trials_b, 2
        )

        self.assertIsNotNone(report.perturbation)
        self.assertEqual(report.perturbation.valid_intervention_rate, 1.0)
        self.assertEqual(report.perturbation.leave_source_rate, 1.0)
        self.assertEqual(report.perturbation.enter_target_rate, 1.0)
        self.assertEqual(len(report.perturbation.records), 4)

        self.assertIsInstance(report.boundary, BoundaryAggregate)
        self.assertEqual(report.boundary.result_count, 4)
        self.assertEqual(report.boundary.success_count, 4)
        self.assertEqual(report.boundary.success_rate, 1.0)
        self.assertIsNotNone(report.boundary.median_cost)
        self.assertEqual(
            report.boundary.intervention_counts["latent_coordinate[0]"], 4
        )
        self.assertEqual(len(report.counterfactuals), 4)
        for result in report.counterfactuals:
            self.assertIsInstance(result, CounterfactualResult)
            self.assertTrue(result.success)
            self.assertTrue(result.valid)
            self.assertEqual(result.new_cluster, 1)
            self.assertEqual(result.intervention_name, "latent_coordinate[0]")
            self.assertGreaterEqual(result.counterfactual_state[0], 0.0)
            self.assertLess(result.counterfactual_state[0], 1e-3)

            original = result.changed_variables["original_value"]
            target = result.changed_variables["requested_target_value"]
            analytic_crossing = -original / (target - original)
            self.assertAlmostEqual(result.magnitude, analytic_crossing, places=3)

        # Four trials x two valid timesteps; the four padded rows are excluded
        # even though their latent_0 values lie in the opposite cluster.
        self.assertEqual(report.metadata["trial_count"], 4)
        self.assertEqual(report.metadata["valid_observation_count"], 8)
        self.assertEqual(report.metadata["source_observation_count"], 4)
        self.assertEqual(report.metadata["rival_observation_count"], 4)

        json_text = report.to_json()
        self.assertNotIn("NaN", json_text)
        self.assertNotIn("Infinity", json_text)
        payload = json.loads(json_text)
        self.assertEqual(payload["candidate_feature"], "latent_0")
        self.assertEqual(payload["boundary"]["success_count"], 4)
        self.assertEqual(len(payload["counterfactuals"]), 4)

        rendered = TextReportRenderer().render(report)
        self.assertIn("CLUSTER 0 INTERPRETATION", rendered)
        self.assertIn("Nearest rival: Cluster 1", rendered)
        self.assertIn("Candidate factor: latent_0", rendered)
        self.assertIn("PERTURBATION", rendered)
        self.assertIn("BOUNDARY", rendered)
        self.assertNotIn("approach", rendered.lower())

    def test_no_actionable_candidate_is_a_structured_report(self):
        pipeline = InterpreterPipeline(
            ThresholdClusterAdapter(),
            pairwise_analyzer=PairwiseAnalyzer(
                min_abs_standardized_difference=1_000_000.0
            ),
        )
        report = pipeline.explain_cluster(_samples(), cluster_id=0)

        self.assertIsInstance(report, InterpreterReport)
        self.assertEqual(report.rival_cluster, 1)
        self.assertIsNone(report.candidate_feature)
        self.assertEqual(report.contrast.candidate_features, ())
        self.assertIsNone(report.perturbation)
        self.assertEqual(report.counterfactuals, ())
        self.assertIsInstance(report.boundary, BoundaryAggregate)
        self.assertEqual(report.boundary.result_count, 0)
        self.assertEqual(report.boundary.success_count, 0)
        self.assertEqual(report.boundary.success_rate, 0.0)
        self.assertTrue(
            any("Layers 2 and 3 were not run" in item for item in report.limitations)
        )

        json_text = report.to_json()
        self.assertNotIn("NaN", json_text)
        self.assertNotIn("Infinity", json_text)
        self.assertIsNone(json.loads(json_text)["candidate_feature"])
        self.assertIn(
            "Candidate factor: none above threshold",
            TextReportRenderer().render(report),
        )

    def test_training_ineligible_trials_do_not_fit_default_distance_scale(self):
        ineligible = tuple(
            BehaviorSample(
                sample_id=sample.sample_id,
                states=sample.states,
                time_mask=sample.time_mask,
                features=sample.features,
                patch_start_s=sample.patch_start_s,
                patch_end_s=sample.patch_end_s,
                metadata={"training_eligible": False},
            )
            for sample in _samples()
        )

        report = InterpreterPipeline(ThresholdClusterAdapter()).explain_cluster(
            ineligible, cluster_id=0
        )

        self.assertEqual(
            report.metadata["distance_scale_source"],
            "identity_scale_interface_only",
        )
        self.assertEqual(report.metadata["distance_reference_observation_count"], 0)
        self.assertEqual(
            set(report.metadata["training_ineligible_trials"]),
            {sample.sample_id for sample in ineligible},
        )
        self.assertTrue(
            any("unstandardized L2" in item for item in report.limitations)
        )
        self.assertTrue(
            any("interface checks only" in item for item in report.limitations)
        )


if __name__ == "__main__":
    unittest.main()
