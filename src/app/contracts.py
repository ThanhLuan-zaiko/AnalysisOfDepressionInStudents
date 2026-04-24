from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RunProfile(str, Enum):
    SAFE = "safe"
    FULL = "full"


class RunPreset(str, Enum):
    QUICK = "quick"
    RESEARCH = "research"


class ArtifactPolicy(str, Enum):
    CONSOLE_ONLY = "console-only"
    JSON = "json"
    FULL_EXPORT = "full-export"


@dataclass(slots=True)
class DatasetBundle:
    path: Path
    frame: Any
    cache_path: Path | None
    loaded_from_cache: bool
    load_seconds: float


@dataclass(slots=True)
class DatasetSplit:
    train_idx: list[int]
    test_idx: list[int]
    random_state: int
    test_size: float


@dataclass(slots=True)
class PredictionBundle:
    y_true: list[int]
    y_score: list[float]
    y_pred: list[int]
    split_name: str
    source: str


@dataclass(slots=True)
class ProfileReport:
    summary: dict[str, Any]
    warnings: list[str]
    profile: dict[str, Any]
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ModelResult:
    name: str
    oof: dict[str, Any]
    holdout: dict[str, Any]
    thresholds: dict[str, Any]
    fairness: list[dict[str, Any]]
    feature_importance: list[dict[str, Any]]
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunConfig:
    profile: RunProfile = RunProfile.SAFE
    preset: RunPreset = RunPreset.QUICK
    artifact_policy: ArtifactPolicy = ArtifactPolicy.JSON
    output_dir: Path = Path("results/app")
    export_html: bool = False
    training_budget_mode: str = "default"
    resolved_training_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    random_state: int = 42
    test_size: float = 0.2
    cv_splits: int = 5
    models: tuple[str, ...] | None = None


@dataclass(slots=True)
class RunReport:
    config: dict[str, Any]
    dataset: dict[str, Any]
    split: dict[str, Any]
    models: dict[str, ModelResult]
    timings: dict[str, float]
    artifacts: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ComparisonReport:
    preset: str
    dataset: dict[str, Any]
    split: dict[str, Any]
    profiles: dict[str, RunReport]
    summary: dict[str, Any]
    artifacts: dict[str, str] = field(default_factory=dict)
