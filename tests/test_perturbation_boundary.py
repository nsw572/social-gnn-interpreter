"""Deterministic tests for validated perturbation and boundary search."""

from __future__ import annotations

import unittest
from types import MappingProxyType, SimpleNamespace

import numpy as np

from interpreter.boundary import (
    BoundaryAggregate,
    CounterfactualSolver,
    InterventionPath,
    LatentCoordinatePath,
    StandardizedL2Distance,
)
from interpreter.core import BehaviorSample, ClusterAdapter, ClusterEvidence
from interpreter.perturbation import (
    Intervention,
    LatentCoordinateIntervention,
    PerturbationRunner,
)
from interpreter.validation import (
    EmpiricalSupportConstraint,
    FiniteStateConstraint,
    Validator,
)


class ThresholdClusterAdapter(ClusterAdapter):
    """Test-only out-of-sample clusterer with a boundary at latent_0 == 0.5."""

    def predict(self, states):
        values = np.asarray(states, dtype=float)
        return (values[:, 0] >= 0.5).astype(int)

    def evidence(self, states):
        values = np.asarray(states, dtype=float)
        target_score = values[:, 0]
        return ClusterEvidence(
            values=np.column_stack((1.0 - target_score, target_score)),
            cluster_ids=(0, 1),
            kind="toy_linear_score",
        )

    def representatives(self):
        return MappingProxyType(
            {0: np.asarray([0.0, 0.0]), 1: np.asarray([1.0, 0.0])}
        )


class NaNIntervention(Intervention):
    @property
    def name(self):
        return "test_nan"

    def apply(self, sample, timestep):
        states = np.array(sample.states, copy=True)
        states[timestep, 0] = np.nan
        # BehaviorSample rejects the invalid proposal before model evaluation;
        # PerturbationRunner turns that rejection into a structured record.
        return sample.with_states(states)


class TwoTimestepIntervention(Intervention):
    def __init__(self, magnitude):
        self.magnitude = float(magnitude)

    @property
    def name(self):
        return "test_two_timestep"

    def apply(self, sample, timestep):
        states = np.array(sample.states, copy=True)
        states[timestep, 0] = self.magnitude
        other = 1 if timestep == 0 else 0
        states[other, 1] += 1.0
        return sample.with_states(states)


class TwoTimestepPath(InterventionPath):
    @property
    def name(self):
        return "test_two_timestep"

    @property
    def lower_bound(self):
        return 0.0

    @property
    def upper_bound(self):
        return 1.0

    def at(self, magnitude):
        return TwoTimestepIntervention(magnitude)

    def changed_variables(self, original, modified, timestep, magnitude):
        return {"magnitude": float(magnitude)}


def make_sample(sample_id="trial-1", first=0.0, second=0.2):
    return BehaviorSample(
        sample_id=sample_id,
        states=np.asarray([[first, 2.0], [second, 3.0], [99.0, 99.0]]),
        time_mask=np.asarray([True, True, False]),
        features={"readable_context": np.asarray([4.0, 5.0, np.nan])},
        patch_start_s=np.asarray([0.0, 1.0, np.nan]),
        patch_end_s=np.asarray([1.0, 2.0, np.nan]),
        metadata={"training_eligible": True},
    )


class ValidationAndPerturbationTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ThresholdClusterAdapter()
        self.validator = Validator()

    def test_latent_intervention_does_not_mutate_source(self):
        sample = make_sample()
        before_states = np.array(sample.states, copy=True)
        before_mask = np.array(sample.time_mask, copy=True)
        intervention = LatentCoordinateIntervention(
            index=0, target_value=1.0, alpha=0.25
        )

        modified = intervention.apply(sample, timestep=0)

        np.testing.assert_array_equal(sample.states, before_states)
        np.testing.assert_array_equal(sample.time_mask, before_mask)
        self.assertIsNot(modified, sample)
        self.assertEqual(modified.states.shape, sample.states.shape)
        self.assertAlmostEqual(modified.states[0, 0], 0.25)
        np.testing.assert_array_equal(modified.states[1:], sample.states[1:])

    def test_nonfinite_state_is_rejected(self):
        # Directly exercise the validation rule because BehaviorSample itself
        # also rejects non-finite values at valid timesteps.
        unsafe = SimpleNamespace(
            states=np.asarray([[np.nan, 0.0]]),
            time_mask=np.asarray([True]),
        )
        result = FiniteStateConstraint().evaluate(unsafe, unsafe, 0)
        self.assertFalse(result.valid)
        self.assertEqual(result.details["invalid_value_count"], 1)

        runner_result = PerturbationRunner(
            self.adapter, self.validator
        ).run([make_sample()], 0, 1, NaNIntervention())
        self.assertEqual(len(runner_result.records), 2)
        self.assertEqual(runner_result.valid_intervention_rate, 0.0)
        self.assertTrue(
            all(record.failure_reason == "intervention_application_failed"
                for record in runner_result.records)
        )

    def test_perturbation_metrics_use_valid_source_observations(self):
        sample = make_sample()
        result = PerturbationRunner(self.adapter, self.validator).run(
            samples=[sample],
            source_cluster=0,
            target_cluster=1,
            intervention=LatentCoordinateIntervention(
                index=0, target_value=1.0, alpha=1.0
            ),
        )

        self.assertEqual(result.attempted_count, 2)
        self.assertEqual(result.valid_count, 2)
        self.assertAlmostEqual(result.valid_intervention_rate, 1.0)
        self.assertAlmostEqual(result.leave_source_rate, 1.0)
        self.assertAlmostEqual(result.enter_target_rate, 1.0)
        self.assertAlmostEqual(result.mean_latent_shift, 0.9)
        self.assertEqual(result.scope, "representation_space")
        self.assertTrue(all(record.new_cluster == 1 for record in result.records))
        self.assertTrue(
            all(record.target_score_after == 1.0 for record in result.records)
        )

    def test_empirical_support_threshold_is_fitted_from_reference_states(self):
        reference = np.asarray([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
        constraint = EmpiricalSupportConstraint(quantile=1.0).fit(reference)
        original = make_sample(first=0.1)
        near = original.with_states(
            np.asarray([[0.15, 2.0], [0.2, 3.0], [99.0, 99.0]])
        )
        far = original.with_states(
            np.asarray([[10.0, 2.0], [0.2, 3.0], [99.0, 99.0]])
        )

        # Coordinate 1 is far from the fitted support in both samples because
        # the toy reference fixes it at zero; compare direct candidate states.
        near_states = np.array(near.states, copy=True)
        near_states[0, 1] = 0.0
        near = near.with_states(near_states)
        far_states = np.array(far.states, copy=True)
        far_states[0, 1] = 0.0
        far = far.with_states(far_states)

        self.assertTrue(constraint.evaluate(original, near, 0).valid)
        self.assertFalse(constraint.evaluate(original, far, 0).valid)


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ThresholdClusterAdapter()
        self.validator = Validator()
        self.metric = StandardizedL2Distance(scale=np.asarray([1.0, 1.0]))

    def test_solver_finds_toy_threshold_near_half_alpha(self):
        sample = make_sample(first=0.0)
        solver = CounterfactualSolver(
            self.adapter,
            self.validator,
            self.metric,
            grid_size=5,
            bisection_steps=14,
            magnitude_tolerance=1e-8,
        )

        result = solver.solve(
            sample,
            target_cluster=1,
            allowed_interventions=LatentCoordinatePath(
                index=0, target_value=1.0
            ),
            timestep=0,
            source_cluster=0,
        )

        self.assertTrue(result.success)
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.magnitude, 0.5, places=6)
        self.assertAlmostEqual(result.intervention_cost, 0.5, places=6)
        self.assertEqual(result.new_cluster, 1)
        self.assertEqual(result.changed_variables["latent_index"], 0)
        self.assertEqual(result.scope, "representation_space")

        aggregate = BoundaryAggregate.from_results([result])
        self.assertEqual(aggregate.success_count, 1)
        self.assertAlmostEqual(aggregate.success_rate, 1.0)
        self.assertAlmostEqual(aggregate.median_cost, 0.5, places=6)

    def test_solver_never_sends_invalid_modified_trial_to_clusterer(self):
        sample = make_sample(first=0.0)
        solver = CounterfactualSolver(
            self.adapter,
            self.validator,
            self.metric,
            grid_size=5,
            bisection_steps=4,
        )

        result = solver.solve(
            sample,
            target_cluster=1,
            allowed_interventions=TwoTimestepPath(),
            timestep=0,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "no_valid_candidates")
        self.assertEqual(result.valid_candidates, 0)
        self.assertIsNotNone(result.validation)
        self.assertFalse(result.validation.valid)


if __name__ == "__main__":
    unittest.main()
