"""Trial-aware three-layer interpretation of temporal social states."""

from .boundary import (
    BoundaryAggregate,
    CounterfactualResult,
    CounterfactualSolver,
    DistanceMetric,
    InterventionPath,
    LatentCoordinatePath,
    StandardizedL2Distance,
)
from .contrast import (
    ContrastResult,
    FeatureDifference,
    NearestCentroidRivalSelector,
    PairwiseAnalyzer,
)
from .core import (
    BehaviorSample,
    ClusterAdapter,
    ClusterEvidence,
    FeatureRegistry,
    ModelAdapter,
    ObservationRef,
    ObservationTable,
    SklearnGaussianMixtureAdapter,
    SklearnKMeansAdapter,
)
from .perturbation import (
    Intervention,
    LatentCoordinateIntervention,
    PerturbationRecord,
    PerturbationResult,
    PerturbationRunner,
)
from .pipeline import InterpreterPipeline
from .report import InterpreterReport, TextReportRenderer
from .validation import (
    Constraint,
    ConstraintResult,
    EmpiricalSupportConstraint,
    FiniteStateConstraint,
    ImmutableContextConstraint,
    SingleTimestepConstraint,
    ValidationResult,
    Validator,
)

__version__ = "0.1.0"

__all__ = [
    "BehaviorSample",
    "BoundaryAggregate",
    "ClusterAdapter",
    "ClusterEvidence",
    "Constraint",
    "ConstraintResult",
    "ContrastResult",
    "CounterfactualResult",
    "CounterfactualSolver",
    "DistanceMetric",
    "EmpiricalSupportConstraint",
    "FeatureDifference",
    "FeatureRegistry",
    "FiniteStateConstraint",
    "ImmutableContextConstraint",
    "Intervention",
    "InterventionPath",
    "InterpreterPipeline",
    "InterpreterReport",
    "LatentCoordinateIntervention",
    "LatentCoordinatePath",
    "ModelAdapter",
    "NearestCentroidRivalSelector",
    "ObservationRef",
    "ObservationTable",
    "PairwiseAnalyzer",
    "PerturbationRecord",
    "PerturbationResult",
    "PerturbationRunner",
    "SingleTimestepConstraint",
    "SklearnGaussianMixtureAdapter",
    "SklearnKMeansAdapter",
    "StandardizedL2Distance",
    "TextReportRenderer",
    "ValidationResult",
    "Validator",
]

