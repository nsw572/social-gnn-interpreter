"""Optional, in-memory integration with the upstream Social-GNN encoder.

This module deliberately starts *after* the upstream alignment/data-loading
boundary.  It never opens a ZIP, NPZ, CSV, or video and it never reconstructs
the social clock.  Its only input is the formal batch mapping emitted by the
upstream ``collate_social_trials`` function.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .core import BehaviorSample, ClusterAdapter, ClusterEvidence, ModelAdapter
from .perturbation import PerturbationRecord, PerturbationResult
from .validation import ConstraintResult, ValidationResult

try:  # PyTorch is an optional dependency of the standalone interpreter.
    import torch
except ImportError as exc:  # pragma: no cover - exercised only without extra
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


EdgeComposer = Callable[["torch.Tensor", "torch.Tensor"], "torch.Tensor"]


@runtime_checkable
class CanonicalBatchTransform(Protocol):
    """Upstream-owned, in-memory transform of an already aligned batch.

    This is a hook for a future canonical-input intervention.  Implementations
    receive a defensive copy and must return a formal Social-GNN batch.  The
    stable downstream result remains a tuple of :class:`BehaviorSample`
    objects containing ``s_t``; this protocol is not a file-loading API.
    """

    def __call__(self, batch: Mapping[str, Any]) -> Mapping[str, Any]: ...


def _require_torch() -> Any:
    if torch is None:  # pragma: no cover - depends on installation
        raise ImportError(
            "SocialGNNEncoderAdapter requires the optional dependency "
            "'torch'. Install this project with the 'social-gnn' extra."
        ) from _TORCH_IMPORT_ERROR
    return torch


def _default_edge_composer() -> EdgeComposer:
    """Import the authoritative upstream composition rule only when needed."""

    try:
        from social_gnn.graph_builder import compose_edge_inputs
    except ImportError as exc:
        raise ImportError(
            "Could not import social_gnn.graph_builder.compose_edge_inputs. "
            "Put the existing Social-GNN checkout on PYTHONPATH, install it, "
            "or inject edge_composer= when constructing the adapter."
        ) from exc
    return compose_edge_inputs


def _as_tensor(value: Any, name: str) -> "torch.Tensor":
    runtime = _require_torch()
    if isinstance(value, runtime.Tensor):
        return value
    try:
        return runtime.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"batch field {name!r} must be tensor-like") from exc


def _binary_mask(value: Any, name: str) -> "torch.Tensor":
    runtime = _require_torch()
    tensor = _as_tensor(value, name)
    if tensor.dtype == runtime.bool:
        return tensor
    if tensor.is_floating_point() and not bool(runtime.isfinite(tensor).all().item()):
        raise ValueError(f"batch field {name!r} contains a non-finite value")
    if not bool(((tensor == 0) | (tensor == 1)).all().item()):
        raise ValueError(f"batch field {name!r} must contain only boolean/0/1 values")
    return tensor.to(dtype=runtime.bool)


def _normalise_trial_ids(value: Any, batch_size: int) -> tuple[str, ...]:
    runtime = _require_torch()
    if isinstance(value, str):
        raw_ids: Sequence[Any] = (value,)
    elif isinstance(value, runtime.Tensor):
        if value.ndim == 0:
            raw_ids = (value.item(),)
        else:
            raw_ids = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        raw_ids = value.reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_ids = value
    else:
        raise TypeError("batch field 'trial_id' must be a string or a sequence of IDs")
    if len(raw_ids) != batch_size:
        raise ValueError(
            f"batch field 'trial_id' must contain {batch_size} IDs, got {len(raw_ids)}"
        )
    trial_ids = tuple(str(item).strip() for item in raw_ids)
    if any(not item for item in trial_ids):
        raise ValueError("batch field 'trial_id' contains an empty ID")
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("batch field 'trial_id' must be unique within a batch")
    return trial_ids


def _python_value(value: Any) -> Any:
    """Detach metadata from mutable tensors/arrays and convert scalar values."""

    runtime = _require_torch()
    if isinstance(value, runtime.Tensor):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else deepcopy(value.tolist())
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else deepcopy(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return deepcopy(value)


def _batched_item(value: Any, index: int, batch_size: int, name: str) -> Any:
    """Select one item from a batch-level metadata container."""

    runtime = _require_torch()
    if isinstance(value, runtime.Tensor):
        if value.ndim == 0:
            return _python_value(value)
        if value.shape[0] != batch_size:
            raise ValueError(
                f"batch metadata field {name!r} must have leading dimension {batch_size}"
            )
        return _python_value(value[index])
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _python_value(value)
        if value.shape[0] != batch_size:
            raise ValueError(
                f"batch metadata field {name!r} must have leading dimension {batch_size}"
            )
        return _python_value(value[index])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != batch_size:
            raise ValueError(
                f"batch metadata field {name!r} must contain {batch_size} items"
            )
        return _python_value(value[index])
    return _python_value(value)


def _metadata_for_trial(batch: Mapping[str, Any], index: int, batch_size: int) -> dict[str, Any]:
    raw_metadata = batch.get("metadata")
    if raw_metadata is None:
        metadata: dict[str, Any] = {}
    elif isinstance(raw_metadata, Mapping):
        metadata = {str(key): _python_value(value) for key, value in raw_metadata.items()}
    elif isinstance(raw_metadata, Sequence) and not isinstance(
        raw_metadata, (str, bytes, bytearray)
    ):
        if len(raw_metadata) != batch_size or not isinstance(raw_metadata[index], Mapping):
            raise ValueError(
                "batch field 'metadata' must be a mapping or one mapping per trial"
            )
        metadata = {
            str(key): _python_value(value)
            for key, value in raw_metadata[index].items()
        }
    else:
        raise TypeError("batch field 'metadata' must be a mapping or sequence of mappings")

    if "identity" in batch:
        identity = _batched_item(batch["identity"], index, batch_size, "identity")
        if isinstance(identity, list):
            identity = tuple(identity)
        metadata["identity"] = identity
    if "training_eligible" in batch:
        metadata["training_eligible"] = _batched_item(
            batch["training_eligible"], index, batch_size, "training_eligible"
        )
    metadata["source_adapter"] = "SocialGNNEncoderAdapter"
    return metadata


def _clone_in_memory_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Clone a mapping before handing it to an external canonical transform."""

    runtime = _require_torch()
    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, runtime.Tensor):
            cloned[key] = value.detach().clone()
        elif isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        else:
            cloned[key] = deepcopy(value)
    return cloned


class SocialGNNEncoderAdapter(ModelAdapter):
    """Adapt an injected ``SocialGNNWithTCN``-like model to ``s_t`` samples.

    Parameters
    ----------
    model:
        An already constructed PyTorch model with the upstream forward
        signature.  Model weights and device placement remain owned by the
        caller.
    edge_composer:
        Optional composition callback, primarily useful for isolated tests.
        By default the adapter lazily imports the authoritative upstream
        ``compose_edge_inputs`` implementation.
    device:
        Optional inference device.  If omitted, the first model parameter or
        buffer determines it; a parameterless model uses the input device.
    """

    _REQUIRED_FIELDS = (
        "node_features",
        "node_mask",
        "edge_values",
        "edge_confidence",
        "time_mask",
        "patch_start_s",
        "patch_end_s",
        "trial_id",
    )

    def __init__(
        self,
        model: Any,
        *,
        edge_composer: EdgeComposer | None = None,
        device: str | "torch.device" | None = None,
    ) -> None:
        runtime = _require_torch()
        if not callable(model):
            raise TypeError("model must be a callable SocialGNNWithTCN-like object")
        self.model = model
        self._edge_composer = edge_composer
        self.device = None if device is None else runtime.device(device)

    def _model_device(self, fallback: "torch.device") -> "torch.device":
        if self.device is not None:
            return self.device
        for accessor in ("parameters", "buffers"):
            values = getattr(self.model, accessor, None)
            if callable(values):
                try:
                    first = next(iter(values()))
                except StopIteration:
                    continue
                return first.device
        return fallback

    def _validated_inputs(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        runtime = _require_torch()
        if not isinstance(batch, Mapping):
            raise TypeError(
                "SocialGNNEncoderAdapter.encode expects an already-loaded batch mapping; "
                "file paths are not accepted"
            )
        missing = [field for field in self._REQUIRED_FIELDS if field not in batch]
        if missing:
            raise KeyError(f"Social-GNN batch is missing required fields: {missing}")

        nodes = _as_tensor(batch["node_features"], "node_features")
        node_mask = _binary_mask(batch["node_mask"], "node_mask")
        edge_values = _as_tensor(batch["edge_values"], "edge_values")
        edge_confidence = _as_tensor(batch["edge_confidence"], "edge_confidence")
        time_mask = _binary_mask(batch["time_mask"], "time_mask")
        patch_start = _as_tensor(batch["patch_start_s"], "patch_start_s")
        patch_end = _as_tensor(batch["patch_end_s"], "patch_end_s")

        if nodes.ndim != 4:
            raise ValueError("node_features must have shape [B,T,N,D_node]")
        batch_size, timesteps, animals, node_dim = nodes.shape
        if min(batch_size, timesteps, animals, node_dim) < 1:
            raise ValueError("node_features dimensions B,T,N,D_node must be positive")
        if not nodes.is_floating_point():
            raise TypeError("node_features must have a floating-point dtype")
        if tuple(node_mask.shape) != (batch_size, timesteps, animals):
            raise ValueError("node_mask must have shape [B,T,N]")
        expected_edges = (batch_size, timesteps, animals, animals, 8)
        if tuple(edge_values.shape) != expected_edges:
            raise ValueError(
                f"edge_values must have shape {expected_edges}, got {tuple(edge_values.shape)}"
            )
        if tuple(edge_confidence.shape) != expected_edges:
            raise ValueError(
                "edge_confidence must align with edge_values and contain exactly 8 traits"
            )
        if not edge_values.is_floating_point() or not edge_confidence.is_floating_point():
            raise TypeError("edge_values and edge_confidence must have floating-point dtypes")
        if tuple(time_mask.shape) != (batch_size, timesteps):
            raise ValueError("time_mask must have shape [B,T]")
        for name, clock in (("patch_start_s", patch_start), ("patch_end_s", patch_end)):
            if tuple(clock.shape) != (batch_size, timesteps):
                raise ValueError(f"{name} must have shape [B,T]")
            if not clock.is_floating_point():
                raise TypeError(f"{name} must have a floating-point dtype")

        for name, tensor in (
            ("node_features", nodes),
            ("edge_values", edge_values),
            ("edge_confidence", edge_confidence),
            ("patch_start_s", patch_start),
            ("patch_end_s", patch_end),
        ):
            if not bool(runtime.isfinite(tensor).all().item()):
                raise ValueError(f"batch field {name!r} contains a non-finite value")
        if not bool(((edge_confidence >= 0) & (edge_confidence <= 1)).all().item()):
            raise ValueError("edge_confidence values must lie in [0,1]")

        # Upstream collation pads suffixes only.  Rejecting holes makes the
        # one-complete-trial invariant explicit at this boundary.
        if bool((time_mask[:, 1:] & ~time_mask[:, :-1]).any().item()):
            raise ValueError("time_mask must contain one valid prefix followed by padding")
        lengths = time_mask.sum(dim=1)
        if bool((lengths < 1).any().item()):
            raise ValueError("every batch item must contain at least one valid timestep")
        if bool((node_mask & ~time_mask.unsqueeze(-1)).any().item()):
            raise ValueError("node_mask cannot mark animals present at padded timesteps")

        valid_clock = time_mask
        if bool((patch_end[valid_clock] <= patch_start[valid_clock]).any().item()):
            raise ValueError("valid patches must have patch_end_s > patch_start_s")
        for row in range(batch_size):
            length = int(lengths[row].item())
            if length > 1:
                if bool((patch_start[row, 1:length] <= patch_start[row, : length - 1]).any().item()):
                    raise ValueError("patch_start_s must be strictly increasing within each trial")
                if bool((patch_end[row, 1:length] <= patch_end[row, : length - 1]).any().item()):
                    raise ValueError("patch_end_s must be strictly increasing within each trial")

        if "sequence_length" in batch:
            declared = _as_tensor(batch["sequence_length"], "sequence_length")
            if tuple(declared.shape) != (batch_size,):
                raise ValueError("sequence_length must have shape [B]")
            if not bool((declared.to(device=lengths.device) == lengths).all().item()):
                raise ValueError("sequence_length disagrees with time_mask")

        trial_ids = _normalise_trial_ids(batch["trial_id"], batch_size)
        target_device = self._model_device(nodes.device)
        return {
            # Cloning protects the source batch even if an injected model uses
            # an in-place operation during inference.
            "node_features": nodes.detach().clone().to(target_device),
            "node_mask": node_mask.detach().clone().to(target_device),
            "edge_values": edge_values.detach().clone().to(target_device),
            "edge_confidence": edge_confidence.detach().clone().to(target_device),
            "time_mask": time_mask.detach().clone().to(target_device),
            "patch_start_s": patch_start.detach().clone(),
            "patch_end_s": patch_end.detach().clone(),
            "trial_ids": trial_ids,
            "batch_size": batch_size,
            "timesteps": timesteps,
            "lengths": lengths.detach().cpu(),
        }

    def encode(self, batch: Mapping[str, Any]) -> tuple[BehaviorSample, ...]:
        """Encode one formal batch and return one sample per complete trial."""

        runtime = _require_torch()
        inputs = self._validated_inputs(batch)
        composer = self._edge_composer or _default_edge_composer()
        edge_features = composer(inputs["edge_values"], inputs["edge_confidence"])
        if not isinstance(edge_features, runtime.Tensor):
            raise TypeError("edge_composer must return a torch.Tensor")
        expected = (
            inputs["batch_size"],
            inputs["timesteps"],
            inputs["node_features"].shape[2],
            inputs["node_features"].shape[2],
            16,
        )
        if tuple(edge_features.shape) != expected:
            raise ValueError(
                "edge_composer must concatenate 8 value and 8 confidence channels; "
                f"expected {expected}, got {tuple(edge_features.shape)}"
            )

        had_training_state = hasattr(self.model, "training")
        was_training = bool(self.model.training) if had_training_state else False
        eval_method = getattr(self.model, "eval", None)
        train_method = getattr(self.model, "train", None)
        if callable(eval_method):
            eval_method()
        try:
            with runtime.no_grad():
                output = self.model(
                    node_features=inputs["node_features"],
                    edge_features=edge_features,
                    node_mask=inputs["node_mask"],
                    time_mask=inputs["time_mask"],
                )
        finally:
            if had_training_state and callable(train_method):
                train_method(was_training)

        states = output[0] if isinstance(output, tuple) else output
        if not isinstance(states, runtime.Tensor):
            raise TypeError("Social-GNN model must return s_t as a torch.Tensor")
        if states.ndim != 3 or tuple(states.shape[:2]) != (
            inputs["batch_size"],
            inputs["timesteps"],
        ):
            raise ValueError("Social-GNN model output s_t must have shape [B,T,D_s]")
        if states.shape[2] < 1:
            raise ValueError("Social-GNN model output D_s must be positive")
        if not bool(runtime.isfinite(states).all().item()):
            raise ValueError("Social-GNN model output s_t contains a non-finite value")

        state_array = states.detach().cpu().numpy().copy()
        mask_array = inputs["time_mask"].detach().cpu().numpy().astype(bool, copy=True)
        start_array = inputs["patch_start_s"].detach().cpu().numpy().copy()
        end_array = inputs["patch_end_s"].detach().cpu().numpy().copy()
        samples: list[BehaviorSample] = []
        for index, trial_id in enumerate(inputs["trial_ids"]):
            metadata = _metadata_for_trial(batch, index, inputs["batch_size"])
            metadata["sequence_length"] = int(inputs["lengths"][index].item())
            samples.append(
                BehaviorSample(
                    sample_id=trial_id,
                    states=state_array[index],
                    time_mask=mask_array[index],
                    features={},
                    patch_start_s=start_array[index],
                    patch_end_s=end_array[index],
                    metadata=metadata,
                )
            )
        return tuple(samples)

    def encode_transformed(
        self,
        batch: Mapping[str, Any],
        transform: CanonicalBatchTransform,
    ) -> tuple[BehaviorSample, ...]:
        """Apply an upstream-owned in-memory transform and rerun the encoder.

        The source mapping is cloned before the callback, preventing a future
        canonical intervention from corrupting the unmodified target batch.
        """

        if not callable(transform):
            raise TypeError("transform must be a callable CanonicalBatchTransform")
        modified = transform(_clone_in_memory_batch(batch))
        if not isinstance(modified, Mapping):
            raise TypeError("canonical batch transform must return a batch mapping")
        return self.encode(modified)


CANONICAL_REENCODED_SCOPE = "canonical_input_reencoded"


def _evidence_row(
    evidence: ClusterEvidence | None, index: int
) -> ClusterEvidence | None:
    if evidence is None:
        return None
    return ClusterEvidence(
        values=evidence.values[index : index + 1],
        cluster_ids=evidence.cluster_ids,
        kind=evidence.kind,
        larger_is_better=evidence.larger_is_better,
    )


def _evidence_score(
    evidence: ClusterEvidence | None, cluster_id: int
) -> float | None:
    if evidence is None:
        return None
    try:
        return float(evidence.score_for(int(cluster_id))[0])
    except KeyError:
        return None


def _same_optional_array(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    return bool(np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True))


class CanonicalReencodingRunner:
    """Run an in-memory canonical intervention through encoder and clusterer.

    This is the stronger Layer-2 path.  The transform receives only a defensive
    copy of an already aligned Social-GNN batch; it cannot read upstream files or
    rebuild the social clock.  Both original and modified batches are encoded as
    complete trials before timestep-level cluster responses are compared.

    The runner validates alignment/provenance but deliberately does not apply
    the representation-space ``SingleTimestepConstraint``: a canonical change
    can legitimately alter the selected state and later causal TCN states.
    Behavioral/physical plausibility constraints remain the responsibility of
    the upstream transform; this runner establishes model sensitivity to the
    supplied in-memory experiment and not biological validity.
    """

    def __init__(
        self,
        encoder: SocialGNNEncoderAdapter,
        cluster_adapter: ClusterAdapter,
    ) -> None:
        if not isinstance(encoder, SocialGNNEncoderAdapter):
            raise TypeError("encoder must be a SocialGNNEncoderAdapter")
        if not isinstance(cluster_adapter, ClusterAdapter):
            raise TypeError("cluster_adapter must implement ClusterAdapter")
        self.encoder = encoder
        self.cluster_adapter = cluster_adapter

    @staticmethod
    def _alignment_validation(
        original: BehaviorSample, modified: BehaviorSample
    ) -> ValidationResult:
        checks = {
            "sample_id_unchanged": original.sample_id == modified.sample_id,
            "state_shape_unchanged": original.states.shape == modified.states.shape,
            "time_mask_unchanged": np.array_equal(
                original.time_mask, modified.time_mask
            ),
            "patch_start_s_unchanged": _same_optional_array(
                original.patch_start_s, modified.patch_start_s
            ),
            "patch_end_s_unchanged": _same_optional_array(
                original.patch_end_s, modified.patch_end_s
            ),
            "identity_unchanged": original.metadata.get("identity")
            == modified.metadata.get("identity"),
            "training_eligibility_unchanged": original.metadata.get(
                "training_eligible"
            )
            == modified.metadata.get("training_eligible"),
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        result = ConstraintResult(
            name="reencoded_trial_alignment",
            valid=not failed,
            details={"checks": checks, "failed_checks": failed},
        )
        return ValidationResult(valid=result.valid, constraint_results=(result,))

    def run(
        self,
        batch: Mapping[str, Any],
        transform: CanonicalBatchTransform,
        *,
        source_cluster: int,
        target_cluster: int,
        intervention_name: str,
    ) -> PerturbationResult:
        if not isinstance(intervention_name, str) or not intervention_name.strip():
            raise ValueError("intervention_name must be a non-empty string")
        originals = self.encoder.encode(batch)
        modified = self.encoder.encode_transformed(batch, transform)
        if len(originals) != len(modified):
            raise ValueError("canonical transform changed the number of trials")

        records: list[PerturbationRecord] = []
        for original, changed in zip(originals, modified):
            validation = self._alignment_validation(original, changed)
            if not validation.valid:
                failed = ", ".join(
                    validation.constraint_results[0].details["failed_checks"]
                )
                raise ValueError(
                    f"canonical transform broke trial alignment for "
                    f"{original.sample_id!r}: {failed}"
                )

            valid_indices = np.flatnonzero(original.time_mask)
            original_states = np.asarray(original.states[valid_indices], dtype=float)
            modified_states = np.asarray(changed.states[valid_indices], dtype=float)
            original_clusters = np.asarray(
                self.cluster_adapter.predict(original_states)
            )
            new_clusters = np.asarray(self.cluster_adapter.predict(modified_states))
            if original_clusters.shape != (len(valid_indices),) or new_clusters.shape != (
                len(valid_indices),
            ):
                raise ValueError("cluster_adapter.predict() must return shape [M]")
            if not np.issubdtype(original_clusters.dtype, np.integer) or not np.issubdtype(
                new_clusters.dtype, np.integer
            ):
                raise TypeError(
                    "cluster_adapter.predict() must return integer identifiers"
                )
            original_evidence = self.cluster_adapter.evidence(original_states)
            modified_evidence = self.cluster_adapter.evidence(modified_states)
            for evidence in (original_evidence, modified_evidence):
                if evidence is not None and evidence.values.shape[0] != len(valid_indices):
                    raise ValueError(
                        "cluster_adapter.evidence() must contain one row per valid state"
                    )

            for row, timestep in enumerate(valid_indices):
                observed_source = int(original_clusters[row])
                if observed_source != int(source_cluster):
                    continue
                observed_new = int(new_clusters[row])
                before = _evidence_row(original_evidence, row)
                after = _evidence_row(modified_evidence, row)
                start_s = (
                    None
                    if original.patch_start_s is None
                    else float(original.patch_start_s[timestep])
                )
                end_s = (
                    None
                    if original.patch_end_s is None
                    else float(original.patch_end_s[timestep])
                )
                records.append(
                    PerturbationRecord(
                        sample_id=original.sample_id,
                        timestep=int(timestep),
                        patch_start_s=start_s,
                        patch_end_s=end_s,
                        source_cluster=int(source_cluster),
                        target_cluster=int(target_cluster),
                        original_cluster=observed_source,
                        new_cluster=observed_new,
                        intervention_name=intervention_name,
                        original_evidence=before,
                        modified_evidence=after,
                        original_cluster_score=_evidence_score(before, observed_source),
                        new_cluster_score=_evidence_score(after, observed_new),
                        source_score_before=_evidence_score(before, source_cluster),
                        source_score_after=_evidence_score(after, source_cluster),
                        target_score_before=_evidence_score(before, target_cluster),
                        target_score_after=_evidence_score(after, target_cluster),
                        latent_l2=float(
                            np.linalg.norm(modified_states[row] - original_states[row])
                        ),
                        validation=validation,
                        valid=True,
                        scope=CANONICAL_REENCODED_SCOPE,
                    )
                )

        valid_count = len(records)
        shifts = [
            float(record.latent_l2)
            for record in records
            if record.latent_l2 is not None
        ]
        return PerturbationResult(
            source_cluster=int(source_cluster),
            target_cluster=int(target_cluster),
            intervention_name=intervention_name,
            records=tuple(records),
            leave_source_rate=(
                sum(record.new_cluster != int(source_cluster) for record in records)
                / valid_count
                if valid_count
                else 0.0
            ),
            enter_target_rate=(
                sum(record.new_cluster == int(target_cluster) for record in records)
                / valid_count
                if valid_count
                else 0.0
            ),
            mean_latent_shift=float(np.mean(shifts)) if shifts else 0.0,
            valid_intervention_rate=1.0 if records else 0.0,
            scope=CANONICAL_REENCODED_SCOPE,
        )


__all__ = [
    "CANONICAL_REENCODED_SCOPE",
    "CanonicalBatchTransform",
    "CanonicalReencodingRunner",
    "SocialGNNEncoderAdapter",
]
