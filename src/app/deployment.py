from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl

from src.training_budget import resolve_training_budget

from .contracts import ArtifactPolicy, ComparisonReport, DatasetBundle, RunPreset, RunProfile
from .services import (
    MIN_RUST_GAM_ROWS,
    NUMERIC_COLUMNS,
    ORDINAL_COLUMNS,
    TARGET_COLUMN,
    _build_logistic_pipeline,
    _rust_engine_status,
    _selected_columns,
    compare_profiles,
    load_dataset,
)


DEFAULT_MODEL_PATH = Path("models/best_depression_model.joblib")
DEFAULT_SELECTION_PATH = Path("results/best_model_selection.json")
DISCLAIMER = (
    "This model is a screening support tool for research/education. "
    "It is not a clinical diagnosis and must not replace professional assessment."
)
DEPLOYABLE_MODELS = ("logistic", "gam", "catboost")
MODEL_TIE_BREAK = {"logistic": 1, "gam": 2, "catboost": 3}
PROFILE_TIE_BREAK = {"full": 1, "safe": 2}


@dataclass(slots=True)
class DeploymentBuildResult:
    selection: dict[str, Any]
    model_path: str
    metadata_path: str
    selection_path: str | None
    comparison_path: str | None
    metadata: dict[str, Any]


@dataclass(slots=True)
class CatBoostFrameTransformer:
    categorical_cols: list[str]
    numeric_cols: list[str]
    fill_values: dict[str, dict[str, Any]]

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "CatBoostFrameTransformer":
        categorical_cols = [
            col
            for col in df.columns
            if df[col].dtype == "object" or str(df[col].dtype).startswith("string")
        ]
        numeric_cols = [col for col in df.columns if col not in categorical_cols]
        fill_values: dict[str, dict[str, Any]] = {"numeric": {}, "categorical": {}}

        for col in numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            median = series.median()
            fill_values["numeric"][col] = float(median) if not pd.isna(median) else 0.0

        for col in categorical_cols:
            series = df[col].astype("string")
            mode = series.mode(dropna=True)
            fill_values["categorical"][col] = str(mode.iloc[0]) if not mode.empty else "Missing"

        return cls(
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            fill_values=fill_values,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        for col in self.numeric_cols:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce").fillna(
                self.fill_values["numeric"][col]
            )
        for col in self.categorical_cols:
            prepared[col] = (
                prepared[col]
                .astype("string")
                .fillna(self.fill_values["categorical"][col])
                .astype(str)
            )
        return prepared


@dataclass(slots=True)
class GAMDesignTransformer:
    feature_names: list[str]
    feature_types: dict[str, str]
    numeric_state: dict[str, dict[str, float]]
    ordinal_state: dict[str, float]
    nominal_state: dict[str, dict[str, Any]]

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "GAMDesignTransformer":
        feature_names: list[str] = []
        feature_types: dict[str, str] = {}
        numeric_state: dict[str, dict[str, float]] = {}
        ordinal_state: dict[str, float] = {}
        nominal_state: dict[str, dict[str, Any]] = {}

        numeric_cols = [col for col in NUMERIC_COLUMNS if col in df.columns]
        ordinal_cols = [col for col in ORDINAL_COLUMNS if col in df.columns]
        nominal_cols = [col for col in df.columns if col not in numeric_cols + ordinal_cols]

        for col in numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            fill_value = float(series.median()) if not pd.isna(series.median()) else 0.0
            filled = series.fillna(fill_value)
            mean = float(filled.mean()) if not pd.isna(filled.mean()) else 0.0
            std = float(filled.std(ddof=0)) if not pd.isna(filled.std(ddof=0)) else 1.0
            numeric_state[col] = {"fill": fill_value, "mean": mean, "std": std or 1.0}
            feature_names.append(col)
            feature_types[col] = "numeric"

        for col in ordinal_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            mode = series.mode(dropna=True)
            fill_value = float(mode.iloc[0]) if not mode.empty else 0.0
            ordinal_state[col] = fill_value
            feature_names.append(col)
            feature_types[col] = "ordinal"

        for col in nominal_cols:
            series = df[col].astype("string")
            mode = series.mode(dropna=True)
            fill_value = str(mode.iloc[0]) if not mode.empty else "Missing"
            filled = series.fillna(fill_value).astype(str)
            categories = sorted(pd.unique(filled))
            mapping = {value: idx for idx, value in enumerate(categories)}
            nominal_state[col] = {
                "fill": fill_value,
                "mapping": mapping,
                "fallback": mapping.get(fill_value, 0),
            }
            feature_names.append(col)
            feature_types[col] = "nominal"

        return cls(
            feature_names=feature_names,
            feature_types=feature_types,
            numeric_state=numeric_state,
            ordinal_state=ordinal_state,
            nominal_state=nominal_state,
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        columns: list[np.ndarray] = []

        for col, state in self.numeric_state.items():
            series = pd.to_numeric(df[col], errors="coerce").fillna(state["fill"])
            columns.append(((series - state["mean"]) / state["std"]).to_numpy(dtype=float))

        for col, fill_value in self.ordinal_state.items():
            series = pd.to_numeric(df[col], errors="coerce").fillna(fill_value)
            columns.append(series.to_numpy(dtype=float))

        for col, state in self.nominal_state.items():
            mapping = state["mapping"]
            fallback = state["fallback"]
            series = df[col].astype("string").fillna(state["fill"]).astype(str)
            columns.append(series.map(lambda value: mapping.get(str(value), fallback)).to_numpy(dtype=float))

        return np.column_stack(columns) if columns else np.empty((len(df), 0))


@dataclass(slots=True)
class DeployedDepressionModel:
    model_name: str
    profile: str
    estimator: Any
    selected_columns: list[str]
    threshold: float
    threshold_policy: str
    metadata: dict[str, Any]
    transformer: Any | None = None

    def predict_frame(self, records_or_frame: Any) -> pd.DataFrame:
        features = _coerce_prediction_frame(records_or_frame, self.selected_columns)
        scores = self._predict_scores(features)
        predictions = (scores >= self.threshold).astype(int)
        return pd.DataFrame(
            {
                "row_index": list(range(len(scores))),
                "probability": [round(float(value), 6) for value in scores],
                "prediction": predictions.astype(int),
                "risk_label": [
                    "screening_flag" if int(value) == 1 else "not_flagged"
                    for value in predictions
                ],
                "threshold": round(float(self.threshold), 6),
                "threshold_policy": self.threshold_policy,
                "model": self.model_name,
                "profile": self.profile,
                "disclaimer": DISCLAIMER,
            }
        )

    def predict(self, records_or_frame: Any) -> list[dict[str, Any]]:
        return self.predict_frame(records_or_frame).to_dict(orient="records")

    def _predict_scores(self, features: pd.DataFrame) -> np.ndarray:
        if self.model_name == "catboost":
            from catboost import Pool

            if not isinstance(self.transformer, CatBoostFrameTransformer):
                raise ValueError("CatBoost deployment artifact is missing its transformer.")
            prepared = self.transformer.transform(features)
            pool = Pool(prepared, cat_features=self.transformer.categorical_cols)
            scores = self.estimator.predict_proba(pool)[:, 1]
            return np.asarray(scores, dtype=float)

        if self.model_name == "gam":
            if not isinstance(self.transformer, GAMDesignTransformer):
                raise ValueError("GAM deployment artifact is missing its transformer.")
            matrix = self.transformer.transform(features)
            scores = self.estimator.predict_proba(matrix)
            return np.asarray(scores, dtype=float).reshape(-1)

        scores = self.estimator.predict_proba(features)[:, 1]
        return np.asarray(scores, dtype=float)


def train_best_deployment(
    dataset_path: str | Path = "Student_Depression_Dataset.csv",
    *,
    preset: RunPreset | str = RunPreset.RESEARCH,
    training_budget_mode: str = "auto",
    output_dir: str | Path = "results/app",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path | None = None,
    selection_path: str | Path | None = DEFAULT_SELECTION_PATH,
    threshold_policy: str = "screening",
) -> DeploymentBuildResult:
    preset = RunPreset(preset)
    bundle = load_dataset(dataset_path)
    comparison = compare_profiles(
        bundle=bundle,
        preset=preset,
        artifact_policy=ArtifactPolicy.JSON,
        output_dir=output_dir,
        training_budget_mode=training_budget_mode,
    )
    selection = select_best_deployable_model(comparison, threshold_policy=threshold_policy)
    deployment = fit_final_deployment_model(
        bundle=bundle,
        selection=selection,
        preset=preset,
        training_budget_mode=training_budget_mode,
    )
    model_path, metadata_path = save_deployment(
        deployment,
        model_path=model_path,
        metadata_path=metadata_path,
    )

    selection_artifact = None
    if selection_path is not None:
        payload = {
            **deployment.metadata["selection"],
            "model_artifact": str(model_path),
            "metadata_artifact": str(metadata_path),
            "disclaimer": DISCLAIMER,
        }
        selection_artifact = _write_json(payload, Path(selection_path))

    comparison_path = None
    if isinstance(comparison.artifacts, dict):
        comparison_path = comparison.artifacts.get("comparison_json")

    return DeploymentBuildResult(
        selection=deployment.metadata["selection"],
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        selection_path=str(selection_artifact) if selection_artifact is not None else None,
        comparison_path=comparison_path,
        metadata=deployment.metadata,
    )


def select_best_deployable_model(
    comparison: ComparisonReport | dict[str, Any],
    *,
    threshold_policy: str = "screening",
) -> dict[str, Any]:
    profiles = _get_value(comparison, "profiles", {})
    candidates: list[dict[str, Any]] = []

    for profile_name in ("safe", "full"):
        report = profiles.get(profile_name) if isinstance(profiles, dict) else None
        if report is None:
            continue
        models = _get_value(report, "models", {})
        selected_cols = _get_value(_get_value(report, "config", {}), "selected_columns", [])
        for model_name in DEPLOYABLE_MODELS:
            result = models.get(model_name) if isinstance(models, dict) else None
            if result is None:
                continue
            holdout = _get_value(result, "holdout", {})
            roc_auc = _to_float(holdout.get("roc_auc"))
            if roc_auc is None:
                continue
            pr_auc = _to_float(holdout.get("pr_auc")) or 0.0
            f1 = _to_float(holdout.get("f1")) or 0.0
            threshold, threshold_source, threshold_warning = _select_threshold(
                _get_value(result, "thresholds", {}),
                threshold_policy,
            )
            candidates.append(
                {
                    "profile": profile_name,
                    "model": model_name,
                    "selected_columns": list(selected_cols),
                    "threshold_policy": threshold_policy,
                    "threshold": threshold,
                    "threshold_source": threshold_source,
                    "threshold_warning": threshold_warning,
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                    "f1": f1,
                    "brier_score": holdout.get("brier_score"),
                    "holdout": dict(holdout),
                    "rank_score": {
                        "roc_auc": roc_auc,
                        "pr_auc": pr_auc,
                        "f1": f1,
                        "model_tie_break": MODEL_TIE_BREAK[model_name],
                        "profile_tie_break": PROFILE_TIE_BREAK.get(profile_name, 0),
                    },
                }
            )

    if not candidates:
        raise ValueError("No deployable model result found. Run at least logistic, GAM, or CatBoost first.")

    candidates.sort(
        key=lambda item: (
            item["rank_score"]["roc_auc"],
            item["rank_score"]["pr_auc"],
            item["rank_score"]["f1"],
            item["rank_score"]["model_tie_break"],
            item["rank_score"]["profile_tie_break"],
        ),
        reverse=True,
    )
    best = candidates[0]
    best["reason"] = (
        "Selected by holdout ROC-AUC across Safe A and Full B; "
        "PR-AUC, F1, model preference, and profile preference are tie-breakers."
    )
    if best["profile"] == "full":
        best["profile_warning"] = (
            "Full B includes the suicidal-thoughts feature and may carry label-leakage risk."
        )
    return best


def fit_final_deployment_model(
    bundle: DatasetBundle,
    selection: dict[str, Any],
    *,
    preset: RunPreset | str = RunPreset.RESEARCH,
    training_budget_mode: str = "auto",
    random_state: int = 42,
) -> DeployedDepressionModel:
    preset = RunPreset(preset)
    profile = RunProfile(selection["profile"])
    selected_cols = selection.get("selected_columns") or _selected_columns(bundle.frame, profile)
    missing = [col for col in selected_cols if col not in bundle.frame.columns]
    if missing:
        raise ValueError(f"Training data is missing selected columns: {', '.join(missing)}")

    full_df = bundle.frame.to_pandas()
    X = _normalize_feature_frame(full_df[selected_cols].copy())
    y = bundle.frame[TARGET_COLUMN].to_numpy().astype(int)
    training_params = resolve_training_budget(
        mode=training_budget_mode,
        family="modern",
        preset=preset.value,
        train_rows=len(X),
    )

    model_name = str(selection["model"])
    transformer: Any | None = None
    model_metadata: dict[str, Any] = {}

    if model_name == "logistic":
        estimator = _fit_logistic(X, y, training_params, random_state)
    elif model_name == "catboost":
        estimator, transformer, model_metadata = _fit_catboost(X, y, training_params, random_state)
    elif model_name == "gam":
        estimator, transformer, model_metadata = _fit_gam(X, y, training_params, random_state)
    else:
        raise ValueError(f"Unsupported deployment model: {model_name}")

    metadata = {
        "artifact_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model": model_name,
        "profile": profile.value,
        "selected_columns": list(selected_cols),
        "target_column": TARGET_COLUMN,
        "threshold": float(selection["threshold"]),
        "threshold_policy": selection["threshold_policy"],
        "threshold_source": selection["threshold_source"],
        "selection": _json_ready(selection),
        "dataset": {
            "path": str(bundle.path),
            "rows": int(bundle.frame.height),
            "cols": int(bundle.frame.width),
            "positive_rate": round(float(y.mean()), 6),
        },
        "final_training_params": training_params,
        "final_model_metadata": model_metadata,
        "disclaimer": DISCLAIMER,
    }

    return DeployedDepressionModel(
        model_name=model_name,
        profile=profile.value,
        estimator=estimator,
        selected_columns=list(selected_cols),
        threshold=float(selection["threshold"]),
        threshold_policy=str(selection["threshold_policy"]),
        metadata=metadata,
        transformer=transformer,
    )


def save_deployment(
    deployment: DeployedDepressionModel,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path | None = None,
) -> tuple[Path, Path]:
    model_path = Path(model_path)
    metadata_path = Path(metadata_path) if metadata_path is not None else model_path.with_suffix(".json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    deployment.metadata["model_artifact"] = str(model_path)
    deployment.metadata["metadata_artifact"] = str(metadata_path)
    joblib.dump(deployment, model_path)
    metadata_path.write_text(
        json.dumps(_json_ready(deployment.metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return model_path, metadata_path


def load_deployment(path: str | Path = DEFAULT_MODEL_PATH) -> DeployedDepressionModel:
    deployment = joblib.load(Path(path))
    if not isinstance(deployment, DeployedDepressionModel):
        raise TypeError(f"Unsupported deployment artifact: {path}")
    return deployment


def predict_records(
    records_or_frame: Any,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> list[dict[str, Any]]:
    return load_deployment(model_path).predict(records_or_frame)


def feature_columns_for_profile(df: pl.DataFrame, profile: RunProfile | str) -> list[str]:
    return _selected_columns(df, RunProfile(profile))


def _fit_logistic(
    X: pd.DataFrame,
    y: np.ndarray,
    training_params: dict[str, dict[str, Any]],
    random_state: int,
) -> Any:
    pipeline = _build_logistic_pipeline(X, random_state)
    pipeline.named_steps["model"].set_params(
        max_iter=training_params.get("logistic", {}).get("max_iter", 1500)
    )
    pipeline.fit(X, y)
    return pipeline


def _fit_catboost(
    X: pd.DataFrame,
    y: np.ndarray,
    training_params: dict[str, dict[str, Any]],
    random_state: int,
) -> tuple[Any, CatBoostFrameTransformer, dict[str, Any]]:
    from catboost import CatBoostClassifier, Pool
    import torch

    transformer = CatBoostFrameTransformer.fit(X)
    prepared = transformer.transform(X)
    budget = training_params.get("catboost", {})
    params: dict[str, Any] = {
        "iterations": budget.get("iterations", 300),
        "depth": budget.get("depth", 6),
        "learning_rate": budget.get("learning_rate", 0.05),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "allow_writing_files": False,
        "verbose": False,
        "random_seed": random_state,
    }
    used_gpu = bool(torch.cuda.is_available())
    if used_gpu:
        params["task_type"] = "GPU"
        params["devices"] = "0"

    estimator = CatBoostClassifier(**params, class_weights=_class_weights(y))
    pool = Pool(prepared, y, cat_features=transformer.categorical_cols)
    estimator.fit(pool)
    metadata = {
        "cat_features": transformer.categorical_cols,
        "numeric_features": transformer.numeric_cols,
        "used_gpu": used_gpu,
    }
    return estimator, transformer, metadata


def _fit_gam(
    X: pd.DataFrame,
    y: np.ndarray,
    training_params: dict[str, dict[str, Any]],
    random_state: int,
) -> tuple[Any, GAMDesignTransformer, dict[str, Any]]:
    from src.ml_models.gam_model import GAMClassifier

    transformer = GAMDesignTransformer.fit(X)
    matrix = transformer.transform(X)
    budget = training_params.get("gam", {})
    rust_status = _rust_engine_status()
    use_rust = bool(rust_status["available"] and len(X) >= MIN_RUST_GAM_ROWS)
    gam = GAMClassifier(random_state=random_state)
    metrics = gam.train(
        matrix,
        y,
        feature_types=transformer.feature_types,
        feature_names=transformer.feature_names,
        n_splines=budget.get("n_splines", 12),
        optimize_splines=bool(budget.get("optimize_splines", False)),
        use_rust=use_rust,
    )
    metadata = {
        "engine": metrics.get("_engine", "pygam"),
        "rust_available": rust_status["available"],
        "rust_used": metrics.get("_engine") == "rust",
        "rust_version": rust_status["version"],
    }
    return gam, transformer, metadata


def _select_threshold(
    thresholds: dict[str, Any],
    threshold_policy: str,
) -> tuple[float, str, str | None]:
    key_by_policy = {
        "screening": "screening",
        "best_f1": "best_f1",
        "f1": "best_f1",
        "youden": "best_youden",
        "youden_j": "best_youden",
    }
    preferred_key = key_by_policy.get(threshold_policy, "screening")
    fallback_keys = [preferred_key, "best_f1", "best_youden"]

    for key in dict.fromkeys(fallback_keys):
        row = thresholds.get(key) if isinstance(thresholds, dict) else None
        if isinstance(row, dict) and row.get("threshold") is not None:
            return float(row["threshold"]), key, None if key == preferred_key else f"Missing {preferred_key}; used {key}."

    return 0.5, "default_0.5", f"Missing {preferred_key}; used default threshold 0.5."


def _coerce_prediction_frame(records_or_frame: Any, selected_columns: list[str]) -> pd.DataFrame:
    if isinstance(records_or_frame, pl.DataFrame):
        frame = records_or_frame.to_pandas()
    elif isinstance(records_or_frame, pd.DataFrame):
        frame = records_or_frame.copy()
    elif isinstance(records_or_frame, dict):
        if any(isinstance(value, (list, tuple, np.ndarray, pd.Series)) for value in records_or_frame.values()):
            frame = pd.DataFrame(records_or_frame)
        else:
            frame = pd.DataFrame([records_or_frame])
    elif isinstance(records_or_frame, list):
        frame = pd.DataFrame(records_or_frame)
    else:
        raise TypeError("Prediction input must be a dict, list of dicts, pandas DataFrame, or polars DataFrame.")

    missing = [col for col in selected_columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Prediction input is missing required columns: {', '.join(missing)}")

    return _normalize_feature_frame(frame[selected_columns].copy())


def _normalize_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for col in [*NUMERIC_COLUMNS, *ORDINAL_COLUMNS]:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    for col in normalized.columns:
        if col not in NUMERIC_COLUMNS and col not in ORDINAL_COLUMNS:
            normalized[col] = normalized[col].astype("string")
    return normalized


def _class_weights(y: np.ndarray) -> list[float]:
    n_samples = len(y)
    negatives = max(int((y == 0).sum()), 1)
    positives = max(int((y == 1).sum()), 1)
    return [n_samples / (2 * negatives), n_samples / (2 * positives)]


def _get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if hasattr(value, "value"):
        return value.value
    return value


def _write_json(data: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(data), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
