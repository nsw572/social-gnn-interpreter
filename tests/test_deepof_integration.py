"""Optional smoke test against an actual Social-GNN checkout.

Set ``SOCIAL_GNN_REPO`` to the repository root before discovery.  The test is
skipped in a standalone install and performs no file I/O beyond importing the
checkout.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np


UPSTREAM_ROOT = os.environ.get("SOCIAL_GNN_REPO")
if UPSTREAM_ROOT:
    sys.path.insert(0, str(Path(UPSTREAM_ROOT).resolve()))

try:
    from social_gnn.data import SocialTrialPackage, collate_social_trials
    from social_gnn.models import SocialGNNWithTCN
except ImportError:
    SocialTrialPackage = None
    collate_social_trials = None
    SocialGNNWithTCN = None

from interpreter.social_gnn import SocialGNNEncoderAdapter


@unittest.skipUnless(
    UPSTREAM_ROOT and SocialTrialPackage is not None,
    "set SOCIAL_GNN_REPO to run the real upstream integration smoke test",
)
class DeepOFIntegrationTest(unittest.TestCase):
    @staticmethod
    def _trial(trial_id: str, timesteps: int):
        starts = np.arange(timesteps, dtype=np.float64) * 0.16
        return SocialTrialPackage(
            trial_id=trial_id,
            node_features=np.arange(
                timesteps * 2 * 4, dtype=np.float32
            ).reshape(timesteps, 2, 4),
            node_mask=np.ones((timesteps, 2), dtype=bool),
            edge_values=np.zeros((timesteps, 2, 2, 8), dtype=np.float32),
            edge_confidence=np.ones((timesteps, 2, 2, 8), dtype=np.float32),
            patch_start_s=starts,
            patch_end_s=starts + 0.32,
            identity=np.asarray(["A", "B"]),
        )

    def test_real_social_gnn_batch_reaches_state_boundary(self):
        batch = collate_social_trials(
            [self._trial("short", 3).to_torch_sample(), self._trial("long", 5).to_torch_sample()]
        )
        model = SocialGNNWithTCN(
            node_dim=4,
            edge_dim=16,
            graph_hidden_dim=8,
            temporal_hidden_dim=6,
            tcn_levels=2,
            tcn_dropout=0.0,
        )
        samples = SocialGNNEncoderAdapter(model).encode(batch)

        self.assertEqual(tuple(sample.sample_id for sample in samples), ("short", "long"))
        self.assertEqual(tuple(samples[0].states.shape), (5, 6))
        self.assertEqual(samples[0].time_mask.tolist(), [True, True, True, False, False])
        self.assertTrue(np.allclose(samples[0].states[3:], 0.0))
        self.assertEqual(samples[1].metadata["identity"], ("A", "B"))


if __name__ == "__main__":
    unittest.main()

