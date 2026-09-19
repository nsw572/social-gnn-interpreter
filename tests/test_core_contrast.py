from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from interpreter.contrast import NearestCentroidRivalSelector, PairwiseAnalyzer
from interpreter.core import (
    BehaviorSample,
    ClusterAdapter,
    FeatureRegistry,
    ModelAdapter,
    ObservationTable,
    SklearnGaussianMixtureAdapter,
    SklearnKMeansAdapter,
)

try:
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
except ImportError:  # pragma: no cover - exercised only in minimal environments
    KMeans = None
    GaussianMixture = None


class _IdentityModel(ModelAdapter):
    def encode(self, batch: object) -> tuple[BehaviorSample, ...]:
        return tuple(batch)  # type: ignore[arg-type]


class _SignClusterer(ClusterAdapter):
    def predict(self, states: object) -> np.ndarray:
        values = np.asarray(states)
        return (values[:, 0] >= 0).astype(int)

    def representatives(self) -> dict[int, np.ndarray]:
        return {0: np.array([-1.0]), 1: np.array([1.0])}


class BehaviorSampleTests(unittest.TestCase):
    def test_is_defensive_readonly_and_context_preserving(self) -> None:
        states = np.array([[1.0, 2.0], [99.0, 99.0], [3.0, 4.0]])
        feature = np.array([0.1, np.nan, 0.3])
        metadata_values = ["a", "b"]
        sample = BehaviorSample(
            sample_id="trial-1",
            states=states,
            time_mask=[True, False, True],
            features={"speed": feature},
            patch_start_s=[0.0, np.nan, 2.0],
            patch_end_s=[1.0, np.nan, 3.0],
            metadata={"labels": metadata_values},
        )

        states[0, 0] = -100.0
        feature[0] = -100.0
        metadata_values.append("c")
        self.assertEqual(sample.states[0, 0], 1.0)
        self.assertEqual(sample.features["speed"][0], 0.1)
        self.assertEqual(sample.metadata["labels"], ("a", "b"))
        with self.assertRaises(ValueError):
            sample.states[0, 0] = 5.0
        with self.assertRaises(TypeError):
            sample.features["new"] = np.zeros(3)  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            sample.sample_id = "other"  # type: ignore[misc]

        replacement = sample.with_states(np.zeros((3, 4)))
        self.assertEqual(replacement.states.shape, (3, 4))
        self.assertEqual(replacement.sample_id, sample.sample_id)
        np.testing.assert_array_equal(replacement.time_mask, sample.time_mask)
        np.testing.assert_array_equal(replacement.patch_start_s, sample.patch_start_s)

    def test_validation_rejects_malformed_or_empty_trials(self) -> None:
        with self.assertRaises(ValueError):
            BehaviorSample("", np.ones((2, 1)), [True, True])
        with self.assertRaises(ValueError):
            BehaviorSample("x", np.empty((0, 2)), [])
        with self.assertRaises(ValueError):
            BehaviorSample("x", np.ones((2, 1)), [False, False])
        with self.assertRaises(ValueError):
            BehaviorSample(
                "x",
                np.ones((2, 1)),
                [True, True],
                features={"not_scalar": np.ones((2, 1))},
            )

    def test_model_adapter_can_preserve_full_trial_samples(self) -> None:
        sample = BehaviorSample("x", np.ones((2, 1)), [True, False])
        encoded = _IdentityModel().encode([sample])
        self.assertIsInstance(encoded, tuple)
        self.assertIs(encoded[0], sample)


class ObservationTableTests(unittest.TestCase):
    def test_padding_is_excluded_and_provenance_is_retained(self) -> None:
        first = BehaviorSample(
            "first",
            states=np.array([[-2.0], [100.0], [2.0]]),
            time_mask=[True, False, True],
            features={"distance": [1.0, 999.0, 3.0]},
            patch_start_s=[0.0, 1.0, 2.0],
            patch_end_s=[1.0, 2.0, 3.0],
        )
        second = BehaviorSample(
            "second",
            states=np.array([[-3.0], [-4.0]]),
            time_mask=[True, True],
            features={"distance": [4.0, 5.0]},
            patch_start_s=[10.0, 11.0],
            patch_end_s=[11.0, 12.0],
        )
        table = ObservationTable.from_samples(
            [first, second],
            _SignClusterer(),
            FeatureRegistry(include_latent_dimensions=False),
        )

        self.assertEqual(len(table), 4)
        self.assertEqual(
            [(ref.sample_id, ref.timestep) for ref in table.refs],
            [("first", 0), ("first", 2), ("second", 0), ("second", 1)],
        )
        self.assertEqual(table.refs[1].start_s, 2.0)
        np.testing.assert_array_equal(table.features["distance"], [1, 3, 4, 5])
        np.testing.assert_array_equal(table.clusters, [0, 1, 0, 0])
        self.assertNotIn(999.0, table.features["distance"])

    def test_registry_validates_extractors_on_valid_rows(self) -> None:
        sample = BehaviorSample("x", [[1.0], [2.0]], [True, False])
        registry = FeatureRegistry(include_latent_dimensions=False).register(
            "derived", lambda _: np.array([4.0, np.nan])
        )
        extracted = registry.extract(sample)
        np.testing.assert_array_equal(extracted["derived"][0], 4.0)
        registry_bad = FeatureRegistry(include_latent_dimensions=False).register(
            "bad", lambda _: np.array([np.nan, 4.0])
        )
        with self.assertRaises(ValueError):
            registry_bad.extract(sample)


@unittest.skipIf(KMeans is None, "scikit-learn is not installed")
class SklearnAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.training = np.array(
            [[-2.0, -1.0], [-1.5, -1.0], [2.0, 1.0], [1.5, 1.0]]
        )

    def test_kmeans_supports_out_of_sample_assignment_and_evidence(self) -> None:
        estimator = KMeans(n_clusters=2, random_state=7, n_init=10).fit(self.training)
        adapter = SklearnKMeansAdapter(estimator)
        query = np.array([[-1.8, -1.0], [1.8, 1.0]])
        predictions = adapter.predict(query)
        evidence = adapter.evidence(query)

        self.assertEqual(predictions.shape, (2,))
        self.assertEqual(evidence.values.shape, (2, 2))
        self.assertEqual(evidence.kind, "negative_euclidean_distance")
        for row, label in enumerate(predictions):
            self.assertEqual(np.argmax(evidence.values[row]), label)
        np.testing.assert_array_equal(
            evidence.score_for(int(predictions[0])),
            evidence.values[:, int(predictions[0])],
        )
        self.assertEqual(set(adapter.representatives()), {0, 1})
        with self.assertRaises(ValueError):
            adapter.predict(np.ones((1, 3)))
        with self.assertRaises(ValueError):
            SklearnKMeansAdapter(KMeans(n_clusters=2))

    def test_gaussian_mixture_supports_probabilistic_evidence(self) -> None:
        estimator = GaussianMixture(n_components=2, random_state=7).fit(self.training)
        adapter = SklearnGaussianMixtureAdapter(estimator)
        query = np.array([[-1.8, -1.0], [1.8, 1.0]])
        predictions = adapter.predict(query)
        evidence = adapter.evidence(query)

        self.assertEqual(evidence.kind, "membership_probability")
        np.testing.assert_allclose(evidence.values.sum(axis=1), 1.0)
        np.testing.assert_array_equal(np.argmax(evidence.values, axis=1), predictions)
        self.assertEqual(set(adapter.representatives()), {0, 1})
        with self.assertRaises(ValueError):
            adapter.evidence(np.ones((1, 3)))
        with self.assertRaises(ValueError):
            SklearnGaussianMixtureAdapter(GaussianMixture(n_components=2))


class ContrastTests(unittest.TestCase):
    def test_nearest_centroid_selector_is_pairwise_and_deterministic(self) -> None:
        selector = NearestCentroidRivalSelector()
        rival = selector.select(
            4,
            {
                4: np.array([0.0, 0.0]),
                9: np.array([2.0, 0.0]),
                1: np.array([0.0, 2.0]),
                8: np.array([10.0, 10.0]),
            },
        )
        self.assertEqual(rival, 1)

    def test_pairwise_analysis_recovers_known_feature(self) -> None:
        samples = []
        # Two source and two rival trials ensure the default analysis unit is
        # demonstrably a trial median, not an individual timestep.
        for sample_id, state, speed, distance in (
            ("a1", 2.0, [8.0, 9.0, 10.0], [4.9, 5.0, 5.1]),
            ("a2", 3.0, [10.0, 11.0, 12.0], [5.0, 5.1, 5.2]),
            ("b1", -2.0, [0.0, 1.0, 2.0], [4.8, 5.0, 5.2]),
            ("b2", -3.0, [2.0, 3.0, 4.0], [4.9, 5.1, 5.3]),
        ):
            samples.append(
                BehaviorSample(
                    sample_id,
                    states=np.full((3, 1), state),
                    time_mask=np.ones(3, dtype=bool),
                    features={"speed": speed, "distance": distance},
                )
            )
        table = ObservationTable.from_samples(
            samples,
            _SignClusterer(),
            FeatureRegistry(include_latent_dimensions=False),
        )
        result = PairwiseAnalyzer(0.8).compare(table, 1, 0)

        self.assertEqual(result.candidate_features[0], "speed")
        self.assertNotIn("distance", result.candidate_features)
        speed = result.feature_differences["speed"]
        self.assertEqual(speed.aggregation, "per_trial_median")
        self.assertEqual(speed.n_observations_a, 6)
        self.assertEqual(speed.n_observations_b, 6)
        self.assertEqual(speed.n_trials_a, 2)
        self.assertEqual(speed.n_trials_b, 2)
        self.assertGreater(speed.standardized_difference, 0)
        self.assertEqual(result.scope, "descriptive_association_only")


if __name__ == "__main__":
    unittest.main()
