from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency environment
    torch = None
    nn = None

from interpreter.core import ClusterAdapter, ClusterEvidence
from interpreter.social_gnn import (
    CANONICAL_REENCODED_SCOPE,
    CanonicalReencodingRunner,
    SocialGNNEncoderAdapter,
)
from interpreter.ssl import (
    MaskedMSELoss,
    PredictionBatch,
    PredictionTargetBuilder,
    SSLObjective,
    TargetBatch,
    TemporalDecoder,
)


def _formal_batch():
    assert torch is not None
    node_features = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
    node_mask = torch.tensor(
        [
            [[1, 1], [1, 1], [1, 1]],
            [[1, 1], [1, 0], [0, 0]],
        ],
        dtype=torch.bool,
    )
    edge_values = torch.full((2, 3, 2, 2, 8), 2.0, dtype=torch.float32)
    edge_confidence = torch.full((2, 3, 2, 2, 8), 0.5, dtype=torch.float32)
    time_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    return {
        "node_features": node_features,
        "node_mask": node_mask,
        "edge_values": edge_values,
        "edge_confidence": edge_confidence,
        "time_mask": time_mask,
        "patch_start_s": torch.tensor(
            [[0.0, 1.0, 2.0], [10.0, 11.0, 0.0]], dtype=torch.float64
        ),
        "patch_end_s": torch.tensor(
            [[1.0, 2.0, 3.0], [11.0, 12.0, 0.0]], dtype=torch.float64
        ),
        "trial_id": ["trial-a", "trial-b"],
        "identity": [("A", "B"), ("C", "D")],
        "sequence_length": torch.tensor([3, 2], dtype=torch.long),
        "training_eligible": torch.tensor([True, False]),
        "metadata": [{"cohort": "first"}, {"cohort": "second"}],
    }


def _snapshot(mapping):
    assert torch is not None
    result = {}
    for key, value in mapping.items():
        result[key] = value.clone() if isinstance(value, torch.Tensor) else deepcopy(value)
    return result


def _assert_mapping_unchanged(testcase, before, after):
    assert torch is not None
    testcase.assertEqual(before.keys(), after.keys())
    for key in before:
        if isinstance(before[key], torch.Tensor):
            testcase.assertTrue(torch.equal(before[key], after[key]), key)
        else:
            testcase.assertEqual(before[key], after[key], key)


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class SocialGNNAdapterTests(unittest.TestCase):
    class TinySocialModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.tensor(0.0))
            self.forward_training = None
            self.grad_enabled = None
            self.seen_time_mask = None
            self.edge_channels = None

        def forward(
            self,
            *,
            node_features,
            edge_features,
            node_mask,
            time_mask,
        ):
            self.forward_training = self.training
            self.grad_enabled = torch.is_grad_enabled()
            self.seen_time_mask = time_mask.detach().clone()
            self.edge_channels = edge_features.shape[-1]

            # Deliberately mutate the supplied tensor.  The adapter must have
            # cloned it, so the formal source batch remains unchanged.
            node_features.add_(1.0)
            node_summary = node_features.mean(dim=(2, 3))
            edge_summary = edge_features[..., 0].mean(dim=(2, 3))
            states = torch.stack((node_summary, edge_summary), dim=-1)
            sentinel = torch.full_like(states, 777.0)
            return torch.where(time_mask.unsqueeze(-1), states, sentinel)

    class NodeThresholdCluster(ClusterAdapter):
        def predict(self, states):
            values = np.asarray(states, dtype=float)
            return (values[:, 0] > 2.0).astype(int)

        def evidence(self, states):
            values = np.asarray(states, dtype=float)
            representatives = np.asarray([[1.0, 2.0], [10.0, 2.0]])
            return ClusterEvidence(
                values=-np.linalg.norm(
                    values[:, None, :] - representatives[None, :, :], axis=2
                ),
                cluster_ids=(0, 1),
                kind="negative_euclidean_distance",
            )

        def representatives(self):
            return {0: np.asarray([1.0, 2.0]), 1: np.asarray([10.0, 2.0])}

    def test_adapter_preserves_trial_mask_provenance_and_source(self):
        batch = _formal_batch()
        before = _snapshot(batch)
        model = self.TinySocialModel()
        model.train()
        composer_observation = {}

        def compose(values, confidence):
            composer_observation["input_shape"] = tuple(values.shape)
            composer_observation["value_mean"] = float(values.mean().item())
            return torch.cat((values, confidence), dim=-1)

        adapter = SocialGNNEncoderAdapter(model, edge_composer=compose)
        samples = adapter.encode(batch)

        self.assertIsInstance(samples, tuple)
        self.assertEqual(len(samples), 2)
        self.assertEqual(composer_observation["input_shape"], (2, 3, 2, 2, 8))
        self.assertEqual(composer_observation["value_mean"], 2.0)
        self.assertEqual(model.edge_channels, 16)
        self.assertFalse(model.forward_training)
        self.assertFalse(model.grad_enabled)
        self.assertTrue(model.training, "adapter must restore the caller's training state")
        self.assertTrue(torch.equal(model.seen_time_mask.cpu(), batch["time_mask"]))

        first, second = samples
        self.assertEqual(first.sample_id, "trial-a")
        self.assertEqual(second.sample_id, "trial-b")
        self.assertEqual(first.states.shape, (3, 2))
        np.testing.assert_array_equal(second.time_mask, [True, True, False])
        np.testing.assert_allclose(second.patch_start_s, [10.0, 11.0, 0.0])
        np.testing.assert_allclose(second.patch_end_s, [11.0, 12.0, 0.0])
        self.assertEqual(second.metadata["identity"], ("C", "D"))
        self.assertEqual(second.metadata["cohort"], "second")
        self.assertFalse(second.metadata["training_eligible"])
        self.assertEqual(second.metadata["sequence_length"], 2)

        # Padding is retained in the full trial tensor and is authoritative
        # only through time_mask; consumers must not interpret its value.
        np.testing.assert_allclose(second.states[2], [777.0, 777.0])
        self.assertFalse(second.time_mask[2])
        _assert_mapping_unchanged(self, before, batch)

    def test_adapter_rejects_non_16_channel_composition(self):
        adapter = SocialGNNEncoderAdapter(
            self.TinySocialModel(), edge_composer=lambda values, _confidence: values
        )
        with self.assertRaisesRegex(ValueError, "16"):
            adapter.encode(_formal_batch())

    def test_canonical_transform_receives_a_defensive_copy(self):
        batch = _formal_batch()
        before = _snapshot(batch)
        adapter = SocialGNNEncoderAdapter(
            self.TinySocialModel(),
            edge_composer=lambda values, confidence: torch.cat((values, confidence), dim=-1),
        )

        def transform(copied):
            copied["node_features"].zero_()
            copied["metadata"][0]["cohort"] = "modified"
            return copied

        samples = adapter.encode_transformed(batch, transform)
        self.assertEqual(samples[0].metadata["cohort"], "modified")
        _assert_mapping_unchanged(self, before, batch)

    def test_canonical_reencoding_runner_reruns_encoder_and_clusterer(self):
        batch = _formal_batch()
        before = _snapshot(batch)
        encoder = SocialGNNEncoderAdapter(
            self.TinySocialModel(),
            edge_composer=lambda values, confidence: torch.cat(
                (values, confidence), dim=-1
            ),
        )

        def remove_node_signal(copied):
            copied["node_features"].zero_()
            return copied

        result = CanonicalReencodingRunner(
            encoder, self.NodeThresholdCluster()
        ).run(
            batch,
            remove_node_signal,
            source_cluster=1,
            target_cluster=0,
            intervention_name="remove_node_signal",
        )

        self.assertEqual(result.scope, CANONICAL_REENCODED_SCOPE)
        self.assertEqual(result.attempted_count, 5)
        self.assertEqual(result.valid_intervention_rate, 1.0)
        self.assertEqual(result.leave_source_rate, 1.0)
        self.assertEqual(result.enter_target_rate, 1.0)
        self.assertTrue(all(record.valid for record in result.records))
        self.assertTrue(
            all(record.scope == CANONICAL_REENCODED_SCOPE for record in result.records)
        )
        self.assertTrue(all(record.latent_l2 > 0.0 for record in result.records))
        _assert_mapping_unchanged(self, before, batch)


@unittest.skipIf(torch is None, "PyTorch optional dependency is not installed")
class SSLInterfaceTests(unittest.TestCase):
    def test_masked_mse_combines_time_and_element_masks(self):
        predictions = torch.tensor(
            [[[1.0, 3.0], [2.0, 4.0], [100.0, 100.0]]], requires_grad=True
        )
        raw_targets = torch.zeros_like(predictions, requires_grad=True)
        time_mask = torch.tensor([[True, True, False]])
        element_mask = torch.tensor(
            [[[True, True], [True, False], [True, True]]]
        )
        prediction = PredictionBatch(predictions=predictions, time_mask=time_mask)
        target = TargetBatch(
            targets=raw_targets,
            time_mask=time_mask,
            target_mask=element_mask,
        )

        with torch.no_grad():
            raw_targets.fill_(9.0)
        self.assertFalse(target.targets.requires_grad)
        self.assertTrue(torch.equal(target.targets, torch.zeros_like(target.targets)))

        loss = MaskedMSELoss()(prediction, target)
        self.assertAlmostEqual(
            float(loss.item()), (1.0 + 9.0 + 4.0) / 3.0, places=6
        )
        loss.backward()
        self.assertTrue(torch.equal(predictions.grad[0, 2], torch.zeros(2)))
        self.assertEqual(float(predictions.grad[0, 1, 1]), 0.0)

    class DoubleDecoder(TemporalDecoder):
        def forward(self, states, time_mask):
            predictions = states * 2.0
            predictions = torch.where(
                time_mask.unsqueeze(-1),
                predictions,
                predictions + 1000.0,
            )
            return PredictionBatch(predictions=predictions, time_mask=time_mask)

    class DoubleTarget(PredictionTargetBuilder):
        def __init__(self):
            self.context_seen = None

        def build(self, states, time_mask, *, context=None):
            self.context_seen = context
            return TargetBatch(targets=states * 2.0, time_mask=time_mask)

    def test_ssl_objective_end_to_end_uses_only_valid_timesteps(self):
        states = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], requires_grad=True
        )
        original = states.detach().clone()
        time_mask = torch.tensor([[True, True, False]])
        target_builder = self.DoubleTarget()
        objective = SSLObjective(
            decoder=self.DoubleDecoder(),
            target_builder=target_builder,
            loss=MaskedMSELoss(),
        )

        loss, prediction, target = objective.compute(
            states, time_mask, context={"objective": "test-only"}
        )
        self.assertEqual(float(loss.item()), 0.0)
        self.assertEqual(target_builder.context_seen, {"objective": "test-only"})
        self.assertFalse(target.targets.requires_grad)
        self.assertTrue(torch.equal(states.detach(), original))
        self.assertTrue(torch.equal(prediction.time_mask, time_mask))
        self.assertTrue(torch.equal(target.time_mask, time_mask))
        self.assertGreater(
            float(prediction.predictions[0, 2, 0].detach().item()), 1000.0
        )


if __name__ == "__main__":
    unittest.main()
