from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ml_models.leakage_check import LabelLeakageInvestigator
from src.visualization import ExploratoryAnalyzer

from .contracts import (
    ArtifactPolicy,
    ComparisonReport,
    DatasetBundle,
    DatasetSplit,
    ModelResult,
    ProfileReport,
    RunConfig,
    RunPreset,
    RunProfile,
    RunReport,
)

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Depression"
SENSITIVE_COLUMN = "Have you ever had suicidal thoughts ?"
EXCLUDED_COLUMNS = [
    "id",
    "Profession",
    "Work Pressure",
    "Job Satisfaction",
]
NUMERIC_COLUMNS = [
    "Age",
    "CGPA",
    "Work/Study Hours",
]
ORDINAL_COLUMNS = [
    "Academic Pressure",
    "Study Satisfaction",
    "Financial Stress",
]
NOMINAL_COLUMNS = [
    "Gender",
    "City",
    "Degree",
    "Sleep Duration",
    "Dietary Habits",
    "Family History of Mental Illness",
]
CACHE_DIR = Path("results/app_cache/datasets")
MIN_RUST_GAM_ROWS = 200


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _json_ready(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
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


def _write_json(data: Any, path: Path) -> Path | None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_json_ready(data), handle, indent=2, ensure_ascii=False)
        return path
    except OSError as exc:
        logger.warning("Could not write JSON artifact %s: %s", path, exc)
        return None


def _positive_class_pct(profile: dict[str, Any]) -> float | None:
    for label, values in profile.get("class_distribution", {}).items():
        if "(1)" in label:
            return values.get("pct")
    return None


def _selected_columns(df: pl.DataFrame, profile: RunProfile) -> list[str]:
    columns = NUMERIC_COLUMNS + ORDINAL_COLUMNS + NOMINAL_COLUMNS
    if profile is RunProfile.FULL:
        columns.append(SENSITIVE_COLUMN)
    return [col for col in columns if col in df.columns]


def _clean_feature_name(name: str) -> str:
    cleaned = name
    for prefix in ("numeric__", "ordinal__", "nominal__"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    return cleaned


def _compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = np.asarray(y_pred).astype(int)

    metrics: dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "positive_rate": round(float(y_true.mean()), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["specificity"] = round(float(tn / (tn + fp)), 4) if (tn + fp) else None
    metrics["fpr"] = round(float(fp / (fp + tn)), 4) if (fp + tn) else None
    metrics["fnr"] = round(float(fn / (fn + tp)), 4) if (fn + tp) else None
    metrics["confusion_matrix"] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        metrics["pr_auc"] = round(float(average_precision_score(y_true, y_score)), 4)
        metrics["brier_score"] = round(float(brier_score_loss(y_true, y_score)), 4)
        prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="uniform")
        metrics["calibration"] = {
            "prob_true": [round(float(value), 4) for value in prob_true],
            "prob_pred": [round(float(value), 4) for value in prob_pred],
        }
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["brier_score"] = None
        metrics["calibration"] = {"prob_true": [], "prob_pred": []}

    return metrics


def _compute_threshold_report(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for threshold in np.arange(0.1, 0.91, 0.05):
        y_pred = (y_score >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "recall": round(float(recall), 4),
                "specificity": round(float(specificity), 4),
                "precision": round(float(precision), 4),
                "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
                "youden_j": round(float(recall + specificity - 1), 4),
                "flagged_pct": round(float(y_pred.mean() * 100), 2),
            }
        )

    return {
        "best_f1": max(rows, key=lambda row: row["f1"]),
        "best_youden": max(rows, key=lambda row: row["youden_j"]),
        "screening": max(rows, key=lambda row: (row["recall"], -row["threshold"])),
        "rows": rows,
    }


def _compute_subgroup_metrics(
    holdout_df: pl.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    y_pred = (y_score >= threshold).astype(int)
    rows: list[dict[str, Any]] = []
    subgroup_masks: list[tuple[str, str, np.ndarray]] = []

    if "Gender" in holdout_df.columns:
        for value in holdout_df["Gender"].unique():
            subgroup_masks.append(
                ("Gender", str(value), holdout_df["Gender"].to_numpy() == value)
            )

    if "Age" in holdout_df.columns:
        ages = holdout_df["Age"].to_numpy()
        bins = [("18-22", 18, 22), ("23-26", 23, 26), ("27-30", 27, 30), ("31+", 31, 120)]
        for label, low, high in bins:
            subgroup_masks.append(("Age", label, (ages >= low) & (ages <= high)))

    if "Family History of Mental Illness" in holdout_df.columns:
        for value in holdout_df["Family History of Mental Illness"].unique():
            subgroup_masks.append(
                (
                    "Family History",
                    str(value),
                    holdout_df["Family History of Mental Illness"].to_numpy() == value,
                )
            )

    for family, label, mask in subgroup_masks:
        if int(mask.sum()) < 20:
            continue
        metrics = _compute_binary_metrics(y_true[mask], y_score[mask], y_pred[mask])
        rows.append(
            {
                "family": family,
                "label": label,
                "n_samples": int(mask.sum()),
                "positive_rate": metrics["positive_rate"],
                "roc_auc": metrics["roc_auc"],
                "f1": metrics["f1"],
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "fpr": metrics["fpr"],
                "fnr": metrics["fnr"],
                "brier_score": metrics["brier_score"],
            }
        )

    return rows


def _build_logistic_pipeline(df: pd.DataFrame, random_state: int) -> Pipeline:
    numeric_cols = [col for col in NUMERIC_COLUMNS if col in df.columns]
    ordinal_cols = [col for col in ORDINAL_COLUMNS if col in df.columns]
    nominal_cols = [col for col in df.columns if col not in numeric_cols + ordinal_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "ordinal",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ordinal_cols,
            ),
            (
                "nominal",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                nominal_cols,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )


def _prepare_catboost_frame(
    df: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
    fill_values: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prepared = df.copy()
    learned_fill_values = fill_values or {"numeric": {}, "categorical": {}}

    for col in numeric_cols:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
        if fill_values is None:
            learned_fill_values["numeric"][col] = float(prepared[col].median())
        prepared[col] = prepared[col].fillna(learned_fill_values["numeric"][col])

    for col in categorical_cols:
        prepared[col] = prepared[col].astype("string")
        if fill_values is None:
            mode = prepared[col].mode(dropna=True)
            learned_fill_values["categorical"][col] = str(mode.iloc[0]) if not mode.empty else "Missing"
        prepared[col] = prepared[col].fillna(learned_fill_values["categorical"][col]).astype(str)

    return prepared, learned_fill_values


def _rust_engine_status() -> dict[str, Any]:
    try:
        import rust_engine
    except Exception as exc:
        return {
            "available": False,
            "version": None,
            "error": str(exc),
        }

    return {
        "available": True,
        "version": getattr(rust_engine, "__version__", "unknown"),
        "error": None,
    }


def _build_gam_design(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[str], dict[str, str]]:
    train_columns: list[np.ndarray] = []
    eval_columns: list[np.ndarray] = []
    feature_names: list[str] = []
    feature_types: dict[str, str] = {}

    numeric_cols = [col for col in NUMERIC_COLUMNS if col in train_df.columns]
    ordinal_cols = [col for col in ORDINAL_COLUMNS if col in train_df.columns]
    nominal_cols = [col for col in train_df.columns if col not in numeric_cols + ordinal_cols]

    for col in numeric_cols:
        train_series = pd.to_numeric(train_df[col], errors="coerce")
        fill_value = float(train_series.median())
        train_filled = train_series.fillna(fill_value)
        mean = float(train_filled.mean())
        std = float(train_filled.std(ddof=0)) or 1.0
        train_columns.append(((train_filled - mean) / std).to_numpy(dtype=float))

        if eval_df is not None:
            eval_series = pd.to_numeric(eval_df[col], errors="coerce").fillna(fill_value)
            eval_columns.append(((eval_series - mean) / std).to_numpy(dtype=float))

        feature_names.append(col)
        feature_types[col] = "numeric"

    for col in ordinal_cols:
        train_series = pd.to_numeric(train_df[col], errors="coerce")
        mode = train_series.mode(dropna=True)
        fill_value = float(mode.iloc[0]) if not mode.empty else 0.0
        train_filled = train_series.fillna(fill_value)
        train_columns.append(train_filled.to_numpy(dtype=float))

        if eval_df is not None:
            eval_series = pd.to_numeric(eval_df[col], errors="coerce").fillna(fill_value)
            eval_columns.append(eval_series.to_numpy(dtype=float))

        feature_names.append(col)
        feature_types[col] = "ordinal"

    for col in nominal_cols:
        train_series = train_df[col].astype("string")
        mode = train_series.mode(dropna=True)
        fill_value = str(mode.iloc[0]) if not mode.empty else "Missing"
        train_filled = train_series.fillna(fill_value).astype(str)
        categories = sorted(pd.unique(train_filled))
        mapping = {value: idx for idx, value in enumerate(categories)}
        fallback_idx = mapping.get(fill_value, 0)

        train_columns.append(
            train_filled.map(lambda value: mapping.get(str(value), fallback_idx)).to_numpy(dtype=float)
        )

        if eval_df is not None:
            eval_series = eval_df[col].astype("string").fillna(fill_value).astype(str)
            eval_columns.append(
                eval_series.map(lambda value: mapping.get(str(value), fallback_idx)).to_numpy(dtype=float)
            )

        feature_names.append(col)
        feature_types[col] = "nominal"

    train_matrix = np.column_stack(train_columns) if train_columns else np.empty((len(train_df), 0))
    eval_matrix = None
    if eval_df is not None:
        eval_matrix = np.column_stack(eval_columns) if eval_columns else np.empty((len(eval_df), 0))

    return train_matrix, eval_matrix, feature_names, feature_types


class DepressionAnalysisService:
    def load_dataset(self, dataset_path: str | Path) -> DatasetBundle:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        stat = path.stat()
        cache_path = CACHE_DIR / f"{path.stem}_{stat.st_size}_{int(stat.st_mtime)}.parquet"

        start = time.perf_counter()
        loaded_from_cache = False
        if cache_path.exists():
            frame = pl.read_parquet(cache_path)
            loaded_from_cache = True
        else:
            frame = pl.scan_csv(path).collect()
            try:
                frame.write_parquet(cache_path)
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not write parquet cache %s: %s", cache_path, exc)
                cache_path = None

        return DatasetBundle(
            path=path,
            frame=frame,
            cache_path=cache_path,
            loaded_from_cache=loaded_from_cache,
            load_seconds=time.perf_counter() - start,
        )

    def profile_dataset(
        self,
        bundle: DatasetBundle,
        artifact_policy: ArtifactPolicy = ArtifactPolicy.JSON,
        export_html: bool = False,
        output_dir: str | Path = "results/app",
    ) -> ProfileReport:
        analyzer = ExploratoryAnalyzer()
        profile = analyzer.generate_data_profile(bundle.frame)
        warnings = self._build_profile_warnings(bundle.frame, profile)
        summary = {
            "rows": profile["shape"]["rows"],
            "cols": profile["shape"]["cols"],
            "target_positive_rate": _positive_class_pct(profile),
            "loaded_from_cache": bundle.loaded_from_cache,
            "cache_path": str(bundle.cache_path) if bundle.cache_path else None,
            "low_variance_columns": [item["column"] for item in profile["low_variance_columns"]],
            "selected_columns_safe": _selected_columns(bundle.frame, RunProfile.SAFE),
            "selected_columns_full": _selected_columns(bundle.frame, RunProfile.FULL),
        }

        artifacts: dict[str, str] = {}
        base_dir = Path(output_dir)
        if artifact_policy in {ArtifactPolicy.JSON, ArtifactPolicy.FULL_EXPORT}:
            profile_path = _write_json(profile, base_dir / "dataset_profile.json")
            summary_path = _write_json({"summary": summary, "warnings": warnings}, base_dir / "dataset_summary.json")
            if profile_path is not None:
                artifacts["profile_json"] = str(profile_path)
            if summary_path is not None:
                artifacts["summary_json"] = str(summary_path)

        if artifact_policy is ArtifactPolicy.FULL_EXPORT and export_html:
            eda = analyzer.run_full_eda(
                bundle.frame,
                output_dir=str(base_dir / "visualizations"),
                save_html=True,
                save_report=True,
            )
            artifacts["eda_dir"] = str(base_dir / "visualizations")
            for key, value in eda["figures"].items():
                artifacts[f"eda_{key}"] = value

        return ProfileReport(summary=summary, warnings=warnings, profile=profile, artifacts=artifacts)

    def make_split(self, df: pl.DataFrame, random_state: int, test_size: float) -> DatasetSplit:
        indices = np.arange(df.height)
        y = df[TARGET_COLUMN].to_numpy()
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        return DatasetSplit(
            train_idx=train_idx.tolist(),
            test_idx=test_idx.tolist(),
            random_state=random_state,
            test_size=test_size,
        )

    def run_pipeline(
        self,
        bundle: DatasetBundle,
        profile: RunProfile = RunProfile.SAFE,
        preset: RunPreset = RunPreset.QUICK,
        artifact_policy: ArtifactPolicy = ArtifactPolicy.JSON,
        output_dir: str | Path = "results/app",
        split: DatasetSplit | None = None,
        config: RunConfig | None = None,
    ) -> RunReport:
        cfg = config or RunConfig(
            profile=profile,
            preset=preset,
            artifact_policy=artifact_policy,
            output_dir=Path(output_dir),
        )
        cfg.profile = profile
        cfg.preset = preset
        cfg.artifact_policy = artifact_policy
        cfg.output_dir = Path(output_dir)

        timings: dict[str, float] = {"load_dataset_seconds": round(bundle.load_seconds, 4)}
        split = split or self.make_split(bundle.frame, cfg.random_state, cfg.test_size)

        start = time.perf_counter()
        profile_report = self.profile_dataset(bundle, ArtifactPolicy.CONSOLE_ONLY)
        timings["profile_seconds"] = round(time.perf_counter() - start, 4)

        selected_cols = _selected_columns(bundle.frame, cfg.profile)
        full_df = bundle.frame.to_pandas()
        feature_df = full_df[selected_cols].copy()
        y = bundle.frame[TARGET_COLUMN].to_numpy().astype(int)

        train_idx = np.asarray(split.train_idx, dtype=int)
        test_idx = np.asarray(split.test_idx, dtype=int)
        X_train = feature_df.iloc[train_idx].reset_index(drop=True)
        X_test = feature_df.iloc[test_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_test = y[test_idx]
        holdout_df = bundle.frame[test_idx.tolist()]

        model_names = list(cfg.models) if cfg.models else self._default_models(cfg.preset)
        model_results: dict[str, ModelResult] = {}
        warnings = list(profile_report.warnings)
        rust_status = _rust_engine_status() if "gam" in model_names or cfg.preset is RunPreset.RESEARCH else None

        if "logistic" in model_names:
            start = time.perf_counter()
            model_results["logistic"] = self._run_logistic(X_train, X_test, y_train, y_test, holdout_df, cfg)
            timings["logistic_seconds"] = round(time.perf_counter() - start, 4)

        if "gam" in model_names:
            start = time.perf_counter()
            model_results["gam"] = self._run_gam(
                X_train,
                X_test,
                y_train,
                y_test,
                holdout_df,
                cfg,
                rust_status=rust_status,
            )
            timings["gam_seconds"] = round(time.perf_counter() - start, 4)
            if model_results["gam"].metadata.get("engine") != "rust":
                fallback_reason = model_results["gam"].metadata.get("rust_error")
                warnings.append(
                    "Rust GAM khong san sang, dang dung pyGAM fallback."
                    + (f" Ly do: {fallback_reason}" if fallback_reason else "")
                )

        if "catboost" in model_names:
            start = time.perf_counter()
            model_results["catboost"] = self._run_catboost(X_train, X_test, y_train, y_test, holdout_df, cfg)
            timings["catboost_seconds"] = round(time.perf_counter() - start, 4)

        research_payload: dict[str, Any] | None = None
        if cfg.preset is RunPreset.RESEARCH:
            start = time.perf_counter()
            research_payload = self._research_summary(bundle.frame, rust_status=rust_status)
            timings["research_seconds"] = round(time.perf_counter() - start, 4)
            warnings.extend(research_payload["warnings"])

        report = RunReport(
            config={
                "profile": cfg.profile.value,
                "preset": cfg.preset.value,
                "artifact_policy": cfg.artifact_policy.value,
                "selected_columns": selected_cols,
                "models": model_names,
                "rust_engine": rust_status,
            },
            dataset={
                "path": str(bundle.path),
                "rows": int(bundle.frame.height),
                "cols": int(bundle.frame.width),
                "loaded_from_cache": bundle.loaded_from_cache,
            },
            split={
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "test_positive_rate": round(float(y_test.mean()), 4),
                "random_state": cfg.random_state,
            },
            models=model_results,
            timings=timings,
            warnings=list(dict.fromkeys(warnings)),
        )

        if cfg.artifact_policy in {ArtifactPolicy.JSON, ArtifactPolicy.FULL_EXPORT}:
            run_path = _write_json(
                _json_ready(report),
                cfg.output_dir / f"run_{cfg.profile.value}_{cfg.preset.value}.json",
            )
            if run_path is not None:
                report.artifacts["run_json"] = str(run_path)
            if research_payload is not None:
                research_path = _write_json(research_payload, cfg.output_dir / f"research_{cfg.profile.value}.json")
                if research_path is not None:
                    report.artifacts["research_json"] = str(research_path)

        return report

    def compare_profiles(
        self,
        bundle: DatasetBundle,
        preset: RunPreset = RunPreset.QUICK,
        artifact_policy: ArtifactPolicy = ArtifactPolicy.JSON,
        output_dir: str | Path = "results/app",
    ) -> ComparisonReport:
        split = self.make_split(bundle.frame, random_state=42, test_size=0.2)
        safe_report = self.run_pipeline(
            bundle=bundle,
            profile=RunProfile.SAFE,
            preset=preset,
            artifact_policy=artifact_policy,
            output_dir=output_dir,
            split=split,
        )
        full_report = self.run_pipeline(
            bundle=bundle,
            profile=RunProfile.FULL,
            preset=preset,
            artifact_policy=artifact_policy,
            output_dir=output_dir,
            split=split,
        )

        summary: dict[str, Any] = {}
        for model_name in sorted(set(safe_report.models) & set(full_report.models)):
            safe_holdout = safe_report.models[model_name].holdout
            full_holdout = full_report.models[model_name].holdout
            summary[model_name] = {
                "safe_roc_auc": safe_holdout.get("roc_auc"),
                "full_roc_auc": full_holdout.get("roc_auc"),
                "roc_auc_delta_full_minus_safe": self._metric_delta(
                    full_holdout.get("roc_auc"), safe_holdout.get("roc_auc")
                ),
                "safe_f1": safe_holdout.get("f1"),
                "full_f1": full_holdout.get("f1"),
                "f1_delta_full_minus_safe": self._metric_delta(
                    full_holdout.get("f1"), safe_holdout.get("f1")
                ),
            }

        comparison = ComparisonReport(
            preset=preset.value,
            dataset=safe_report.dataset,
            split=safe_report.split,
            profiles={"safe": safe_report, "full": full_report},
            summary=summary,
        )

        if artifact_policy in {ArtifactPolicy.JSON, ArtifactPolicy.FULL_EXPORT}:
            comparison_path = _write_json(
                _json_ready(comparison),
                Path(output_dir) / f"compare_profiles_{preset.value}.json",
            )
            if comparison_path is not None:
                comparison.artifacts["comparison_json"] = str(comparison_path)

        return comparison

    def _default_models(self, preset: RunPreset) -> list[str]:
        if preset is RunPreset.RESEARCH:
            return ["logistic", "gam", "catboost"]
        if preset is RunPreset.QUICK:
            return ["logistic", "catboost"]
        return ["logistic"]

    def _metric_delta(self, newer: float | None, older: float | None) -> float | None:
        if newer is None or older is None:
            return None
        return round(float(newer - older), 4)

    def _build_profile_warnings(self, df: pl.DataFrame, profile: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        positive_rate = _positive_class_pct(profile)
        if positive_rate is not None and positive_rate > 55:
            warnings.append(
                f"Positive class chiem {positive_rate}% - can giu stratified split va khong dung accuracy don le."
            )

        if SENSITIVE_COLUMN in df.columns:
            sensitive_yes = df.filter(pl.col(SENSITIVE_COLUMN) == "Yes").height
            sensitive_rate = round(sensitive_yes / df.height * 100, 2)
            if sensitive_rate > 50:
                warnings.append(
                    f"'{SENSITIVE_COLUMN}' co ty le Yes = {sensitive_rate}% - nguy co leakage cao trong profile full."
                )

        missing_columns = [col for col, count in profile["missing_values"].items() if count > 0]
        if missing_columns:
            warnings.append(
                f"Dataset con {len(missing_columns)} cot co missing values: {', '.join(missing_columns)}."
            )

        rare_categories: list[str] = []
        for col in df.columns:
            if df[col].dtype != pl.String:
                continue
            counts = df[col].value_counts()
            for row in counts.iter_rows(named=True):
                pct = row["count"] / df.height * 100
                if pct < 1:
                    rare_categories.append(f"{col}={row[col]} ({pct:.2f}%)")
                    break
        if rare_categories:
            warnings.append(f"Co category hiem can can nhac gop nhom: {', '.join(rare_categories[:4])}.")

        return warnings

    def _run_logistic(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        holdout_df: pl.DataFrame,
        cfg: RunConfig,
    ) -> ModelResult:
        pipeline = _build_logistic_pipeline(X_train, cfg.random_state)
        cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.random_state)
        oof_scores = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        oof_pred = (oof_scores >= 0.5).astype(int)

        pipeline.fit(X_train, y_train)
        holdout_scores = pipeline.predict_proba(X_test)[:, 1]
        holdout_pred = pipeline.predict(X_test)

        feature_names = [
            _clean_feature_name(name)
            for name in pipeline.named_steps["preprocessor"].get_feature_names_out()
        ]
        coefficients = pipeline.named_steps["model"].coef_[0]
        coef_rows = [
            {
                "feature": name,
                "coefficient": round(float(coef), 6),
                "abs_coefficient": round(float(abs(coef)), 6),
                "odds_ratio": round(float(np.exp(coef)), 6),
            }
            for name, coef in zip(feature_names, coefficients)
        ]
        coef_rows.sort(key=lambda row: row["abs_coefficient"], reverse=True)

        thresholds = _compute_threshold_report(y_test, holdout_scores)
        fairness = _compute_subgroup_metrics(
            holdout_df,
            y_test,
            holdout_scores,
            thresholds["best_f1"]["threshold"],
        )

        return ModelResult(
            name="logistic",
            oof=_compute_binary_metrics(y_train, oof_scores, oof_pred),
            holdout=_compute_binary_metrics(y_test, holdout_scores, holdout_pred),
            thresholds=thresholds,
            fairness=fairness,
            feature_importance=coef_rows[:20],
            metadata={
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "evaluation": "oof_on_train_plus_holdout",
            },
        )

    def _run_catboost(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        holdout_df: pl.DataFrame,
        cfg: RunConfig,
    ) -> ModelResult:
        from catboost import CatBoostClassifier, Pool
        import torch

        categorical_cols = [
            col
            for col in X_train.columns
            if X_train[col].dtype == "object" or str(X_train[col].dtype).startswith("string")
        ]
        numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

        params: dict[str, Any] = {
            "iterations": 300,
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "allow_writing_files": False,
            "verbose": False,
            "random_seed": cfg.random_state,
            "early_stopping_rounds": 30,
        }
        if torch.cuda.is_available():
            params["task_type"] = "GPU"
            params["devices"] = "0"

        cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.random_state)
        oof_scores = np.zeros(len(X_train), dtype=float)

        for fold_train_idx, fold_val_idx in cv.split(X_train, y_train):
            fold_X_train = X_train.iloc[fold_train_idx].reset_index(drop=True)
            fold_X_val = X_train.iloc[fold_val_idx].reset_index(drop=True)
            fold_y_train = y_train[fold_train_idx]
            fold_y_val = y_train[fold_val_idx]

            fold_X_train, fill_values = _prepare_catboost_frame(
                fold_X_train,
                categorical_cols=categorical_cols,
                numeric_cols=numeric_cols,
            )
            fold_X_val, _ = _prepare_catboost_frame(
                fold_X_val,
                categorical_cols=categorical_cols,
                numeric_cols=numeric_cols,
                fill_values=fill_values,
            )

            model = CatBoostClassifier(**params, class_weights=self._class_weights(fold_y_train))
            train_pool = Pool(fold_X_train, fold_y_train, cat_features=categorical_cols)
            val_pool = Pool(fold_X_val, fold_y_val, cat_features=categorical_cols)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            oof_scores[fold_val_idx] = model.predict_proba(val_pool)[:, 1]

        oof_pred = (oof_scores >= 0.5).astype(int)

        X_train_prepared, fill_values = _prepare_catboost_frame(
            X_train,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
        )
        X_test_prepared, _ = _prepare_catboost_frame(
            X_test,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            fill_values=fill_values,
        )

        final_model = CatBoostClassifier(**params, class_weights=self._class_weights(y_train))
        train_pool = Pool(X_train_prepared, y_train, cat_features=categorical_cols)
        test_pool = Pool(X_test_prepared, y_test, cat_features=categorical_cols)
        final_model.fit(train_pool, eval_set=test_pool, use_best_model=True)

        holdout_scores = final_model.predict_proba(test_pool)[:, 1]
        holdout_pred = (holdout_scores >= 0.5).astype(int)
        importance = final_model.get_feature_importance(train_pool)
        feature_rows = [
            {"feature": feature, "importance": round(float(score), 6)}
            for feature, score in zip(X_train_prepared.columns, importance)
        ]
        feature_rows.sort(key=lambda row: row["importance"], reverse=True)

        thresholds = _compute_threshold_report(y_test, holdout_scores)
        fairness = _compute_subgroup_metrics(
            holdout_df,
            y_test,
            holdout_scores,
            thresholds["best_f1"]["threshold"],
        )

        return ModelResult(
            name="catboost",
            oof=_compute_binary_metrics(y_train, oof_scores, oof_pred),
            holdout=_compute_binary_metrics(y_test, holdout_scores, holdout_pred),
            thresholds=thresholds,
            fairness=fairness,
            feature_importance=feature_rows[:20],
            metadata={
                "cat_features": categorical_cols,
                "used_gpu": bool(torch.cuda.is_available()),
                "evaluation": "oof_on_train_plus_holdout",
            },
        )

    def _run_gam(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        holdout_df: pl.DataFrame,
        cfg: RunConfig,
        rust_status: dict[str, Any] | None = None,
    ) -> ModelResult:
        from src.ml_models.gam_model import GAMClassifier

        rust_status = rust_status or _rust_engine_status()
        X_train_matrix, X_test_matrix, feature_names, feature_types = _build_gam_design(X_train, X_test)
        n_splines = 12 if cfg.preset is RunPreset.RESEARCH else 10
        use_rust = bool(rust_status["available"] and len(X_train) >= MIN_RUST_GAM_ROWS)
        rust_failure = rust_status["error"]
        if rust_status["available"] and not use_rust:
            rust_failure = (
                f"Rust GAM chi bat dau tu khoang {MIN_RUST_GAM_ROWS} train rows; "
                f"split hien tai co {len(X_train)} rows."
            )

        def fit_and_score(use_rust: bool) -> tuple[dict[str, Any], np.ndarray]:
            gam = GAMClassifier(random_state=cfg.random_state)
            optimize_splines = bool(use_rust and cfg.preset is RunPreset.RESEARCH)
            gam_metrics = gam.train(
                X_train_matrix,
                y_train,
                feature_types=feature_types,
                feature_names=feature_names,
                n_splines=n_splines,
                optimize_splines=optimize_splines,
                use_rust=use_rust,
            )
            holdout_scores = np.asarray(gam.predict_proba(X_test_matrix), dtype=float)
            return gam_metrics, holdout_scores

        try:
            gam_metrics, holdout_scores = fit_and_score(use_rust)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)) or not use_rust:
                raise
            rust_failure = str(exc)
            logger.warning("Rust GAM failed during train/predict, retrying with pyGAM: %s", exc)
            gam_metrics, holdout_scores = fit_and_score(False)

        holdout_pred = (holdout_scores >= 0.5).astype(int)

        feature_rows: list[dict[str, Any]] = []
        for row in gam_metrics.get("feature_importance", []):
            feature_rows.append(
                {
                    "feature": row.get("feature"),
                    "variance_importance": round(float(row.get("variance_importance", 0.0)), 6),
                    "term_index": row.get("term_index"),
                }
            )
        feature_rows.sort(key=lambda row: row["variance_importance"], reverse=True)

        thresholds = _compute_threshold_report(y_test, holdout_scores)
        fairness = _compute_subgroup_metrics(
            holdout_df,
            y_test,
            holdout_scores,
            thresholds["best_f1"]["threshold"],
        )

        oof = {
            "n_samples": int(len(y_train)),
            "positive_rate": round(float(y_train.mean()), 4),
            "accuracy": None,
            "f1": round(float(gam_metrics.get("f1", 0.0)), 4),
            "recall": round(float(gam_metrics.get("recall", 0.0)), 4),
            "precision": round(float(gam_metrics.get("precision", 0.0)), 4),
            "specificity": None,
            "fpr": None,
            "fnr": None,
            "confusion_matrix": {},
            "roc_auc": round(float(gam_metrics.get("roc_auc", 0.0)), 4),
            "pr_auc": round(float(gam_metrics.get("pr_auc", 0.0)), 4),
            "brier_score": round(float(gam_metrics.get("brier_score", 0.0)), 4),
            "calibration": {"prob_true": [], "prob_pred": []},
            "roc_auc_std": round(float(gam_metrics.get("roc_auc_std", 0.0)), 4),
            "pr_auc_std": round(float(gam_metrics.get("pr_auc_std", 0.0)), 4),
            "f1_std": round(float(gam_metrics.get("f1_std", 0.0)), 4),
        }

        return ModelResult(
            name="gam",
            oof=oof,
            holdout=_compute_binary_metrics(y_test, holdout_scores, holdout_pred),
            thresholds=thresholds,
            fairness=fairness,
            feature_importance=feature_rows[:20],
            metadata={
                "engine": gam_metrics.get("_engine", "pygam"),
                "rust_available": rust_status["available"],
                "rust_used": use_rust and gam_metrics.get("_engine", "pygam") == "rust",
                "rust_version": rust_status["version"],
                "rust_error": rust_failure if gam_metrics.get("_engine", "pygam") != "rust" else None,
                "n_splines": gam_metrics.get("n_splines", n_splines),
                "optimize_splines": gam_metrics.get("optimize_splines"),
                "evaluation": "cv_summary_on_train_plus_holdout",
            },
        )

    def _class_weights(self, y: np.ndarray) -> list[float]:
        n_samples = len(y)
        negatives = max(int((y == 0).sum()), 1)
        positives = max(int((y == 1).sum()), 1)
        return [n_samples / (2 * negatives), n_samples / (2 * positives)]

    def _research_summary(
        self,
        df: pl.DataFrame,
        rust_status: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        investigator = LabelLeakageInvestigator(random_state=42)
        cross_tab = investigator.cross_tab_analysis(df)
        feature_comp = investigator.feature_importance_comparison(df)
        synthetic = investigator.synthetic_data_check(df)
        rust_status = rust_status or _rust_engine_status()

        warnings: list[str] = []
        auc_drop_pct = feature_comp.get("auc_drop_pct", 0.0)
        if auc_drop_pct > 3:
            warnings.append(
                f"Profile full tang AUC them {auc_drop_pct}% nho bien suicidal-thoughts - can xem la mode nghien cuu."
            )
        if synthetic.get("likely_synthetic"):
            warnings.append(
                f"Dataset co {synthetic.get('total_warnings', 0)} dau hieu synthetic - uu tien bao cao holdout."
            )
        if not rust_status["available"]:
            warnings.append(
                "Rust engine khong kha dung, preset research se cham hon vi GAM dang fallback sang pyGAM."
            )

        return {
            "cross_tab": cross_tab,
            "feature_importance_comparison": feature_comp,
            "synthetic_check": synthetic,
            "rust_engine": rust_status,
            "warnings": warnings,
        }


_SERVICE = DepressionAnalysisService()


def load_dataset(dataset_path: str | Path) -> DatasetBundle:
    return _SERVICE.load_dataset(dataset_path)


def profile_dataset(
    bundle: DatasetBundle,
    artifact_policy: ArtifactPolicy = ArtifactPolicy.JSON,
    export_html: bool = False,
    output_dir: str | Path = "results/app",
) -> ProfileReport:
    return _SERVICE.profile_dataset(bundle, artifact_policy, export_html, output_dir)


def run_pipeline(
    bundle: DatasetBundle,
    profile: RunProfile = RunProfile.SAFE,
    preset: RunPreset = RunPreset.QUICK,
    artifact_policy: ArtifactPolicy = ArtifactPolicy.JSON,
    output_dir: str | Path = "results/app",
    split: DatasetSplit | None = None,
    config: RunConfig | None = None,
) -> RunReport:
    return _SERVICE.run_pipeline(bundle, profile, preset, artifact_policy, output_dir, split, config)


def compare_profiles(
    bundle: DatasetBundle,
    preset: RunPreset = RunPreset.QUICK,
    artifact_policy: ArtifactPolicy = ArtifactPolicy.JSON,
    output_dir: str | Path = "results/app",
) -> ComparisonReport:
    return _SERVICE.compare_profiles(bundle, preset, artifact_policy, output_dir)
