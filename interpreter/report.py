"""Machine-readable and deliberately conservative Interpreter reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


def _json_value(value: Any) -> Any:
    """Convert nested result objects to JSON-compatible Python values."""

    if is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class InterpreterReport:
    """Evidence from the three layers for one source cluster.

    The report intentionally keeps the statistical contrast, model experiment,
    and boundary search separate.  Downstream renderers should not collapse
    those evidence levels into a causal or semantic label.
    """

    source_cluster: int
    rival_cluster: int
    candidate_feature: str | None
    contrast: Any
    perturbation: Any
    counterfactuals: tuple[Any, ...]
    boundary: Any
    interpretation_scope: str = "representation_space"
    limitations: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        metadata = {} if self.metadata is None else dict(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)


class TextReportRenderer:
    """Render a compact evidence summary without inventing behavior names."""

    def render(self, report: InterpreterReport) -> str:
        contrast = report.contrast
        perturbation = report.perturbation
        boundary = report.boundary
        rival_label = (
            "Nearest rival"
            if report.metadata.get("rival_selection") == "nearest_representative"
            else "Rival"
        )
        lines = [
            f"CLUSTER {report.source_cluster} INTERPRETATION",
            "",
            f"{rival_label}: Cluster {report.rival_cluster}",
            f"Scope: {report.interpretation_scope}",
            "",
            "CONTRAST",
            f"Candidate factor: {report.candidate_feature or 'none above threshold'}",
        ]

        if perturbation is not None:
            lines.extend(
                [
                    "",
                    "PERTURBATION",
                    f"Valid intervention rate: {getattr(perturbation, 'valid_intervention_rate', 0.0):.3f}",
                    f"Leave-source rate: {getattr(perturbation, 'leave_source_rate', 0.0):.3f}",
                    f"Enter-target rate: {getattr(perturbation, 'enter_target_rate', 0.0):.3f}",
                ]
            )

        if boundary is not None:
            lines.extend(
                [
                    "",
                    "BOUNDARY",
                    f"Valid crossing rate: {getattr(boundary, 'success_rate', 0.0):.3f}",
                    f"Median representation cost: {getattr(boundary, 'median_cost', None)}",
                ]
            )

        if report.limitations:
            lines.extend(["", "LIMITATIONS"])
            lines.extend(f"- {item}" for item in report.limitations)
        return "\n".join(lines)


__all__ = ["InterpreterReport", "TextReportRenderer"]
