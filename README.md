# Social behavior Interpreter

This repository contains a small, independent downstream interpreter for the
`s_t` representation emitted by `SocialGNNWithTCN`.  It deliberately does not
read H ZIPs, node/edge NPZs, CSVs, or videos, and it never reconstructs the
social clock.  The integration boundary is an already loaded Social-GNN batch:

```text
complete aligned trial batch
  -> existing SocialGNNWithTCN adapter
  -> s_t + time_mask + trial/timestamp provenance
  -> frozen cluster adapter
  -> contrast -> perturbation -> boundary report
```

The storage and batching unit remains one complete trial.  Cluster assignments
and explanations are attached only to valid social timesteps, identified by
`(trial_id, timestep, patch_start_s, patch_end_s)`.  Padded timesteps are never
sent to the clusterer.

## Scientific scope

The first working intervention and counterfactual are intentionally performed
in representation space.  They answer whether a fitted downstream cluster
boundary depends on a coordinate of `s_t`; they do **not** establish biological
causality, and an edit to `s_t` need not correspond to a realizable pose or
trajectory.  For stronger model-dependence experiments,
`CanonicalReencodingRunner` accepts an upstream-owned transform of an in-memory
aligned batch, reruns the frozen encoder and clusterer, and compares the new
`s_t + time_mask` output.  The Interpreter itself still does not need file-level
or video-level dependencies.

Readable behavioral claims require a trained and validated decoder/readout.
Until one exists, the default feature registry names factors `latent_0`,
`latent_1`, and so on.  The package exposes decoder, target-builder, and masked
loss interfaces without choosing an arbitrary SSL objective.

## Upstream contract

The Social-GNN adapter expects the existing formal fields:

```text
node_features     [B,T,N,D_node]
node_mask         [B,T,N]
edge_values       [B,T,N,N,8]
edge_confidence   [B,T,N,N,8]
time_mask         [B,T]
patch_start_s     [B,T]
patch_end_s       [B,T]
trial_id          list[str]
```

It reuses `social_gnn.graph_builder.compose_edge_inputs`, so the encoder sees
the established 16-channel `[edge_values, edge_confidence]` input.  It always
passes the authoritative `time_mask` explicitly.

```python
from interpreter import InterpreterPipeline, SklearnKMeansAdapter
from interpreter.social_gnn import SocialGNNEncoderAdapter

# `batch` comes from the existing SocialTrialDataLoader. `social_model` and
# `fitted_kmeans` are frozen artifacts supplied by the upstream project.
samples = SocialGNNEncoderAdapter(social_model).encode(batch)
report = InterpreterPipeline(
    SklearnKMeansAdapter(fitted_kmeans)
).explain_cluster(samples, cluster_id=14)

machine_readable_json = report.to_json()
```

The duplicate A/B latent packages currently marked
`training_eligible=false` by the H converter are suitable only for interface
checks.  They must not be used to fit an encoder, decoder, scaler, or clusterer.

## Current upstream gaps

- There is not yet a frozen trained encoder/decoder plus persisted cluster
  artifact, so the repository's complete numerical example remains a
  deterministic test fixture rather than a biological result.
- The current Social-GNN KMeans/GMM fit helpers flatten embeddings without
  passing `time_mask`; training code must exclude padded rows and retain the
  `(trial_id, timestep)` mapping before its artifact is accepted here.
- `SocialTrialPackage.from_npz` does not currently propagate the converter's
  `training_eligible` flag.  Training manifests must reject duplicate-latent
  trials explicitly; the adapter preserves the flag when it is present in an
  in-memory batch.
- The upstream directed-edge aggregation and invalid-node message masking need
  semantic tests before an Interpreter result is treated as scientific
  evidence.  This project does not silently change those encoder semantics.

## Tests

The test suite uses only deterministic test-local models for the unfinished
training artifacts:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

To exercise the optional Social-GNN integration against the adjacent upstream
checkout, put that checkout on `PYTHONPATH`; no source files need to be copied.
