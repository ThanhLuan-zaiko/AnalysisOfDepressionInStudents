from __future__ import annotations

import ast
import contextlib
import importlib
import io
import json
import os
import re
import webbrowser
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.app import (
    ArtifactPolicy,
    RunConfig,
    RunPreset,
    RunProfile,
    compare_profiles,
    load_dataset,
    profile_dataset,
    run_pipeline,
)


RESULT_DIRS = (Path("results"), Path("results/app"))
LOG_DIRS = (Path("results/app/console_logs"), Path("logs"))
LEGACY_WORKFLOW_IDS = (
    "eda",
    "stats",
    "models",
    "full",
    "review",
    "standardize",
    "famd",
    "split",
    "fairness",
    "subgroups",
    "robustness",
    "analysis",
    "report",
)


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    workflow_id: str
    label: str
    family: str
    description: str
    default_preset: str = "quick"
    supports_variant: bool = True
    supports_export_html: bool = False
    supports_budget: bool = False


@dataclass(slots=True)
class WorkflowRequest:
    workflow_id: str
    dataset_path: str = "Student_Depression_Dataset.csv"
    variant: str = "A"
    preset: str = "quick"
    output_dir: str = "results/app"
    export_html: bool = False
    console_only: bool = False
    training_budget_mode: str = "default"


@dataclass(slots=True)
class WorkflowResult:
    workflow_id: str
    family: str
    payload: Any | None = None
    transcript: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    html_artifacts: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


WORKFLOW_SPECS: dict[str, WorkflowSpec] = {
    "profile": WorkflowSpec(
        workflow_id="profile",
        label="Profile Dataset",
        family="modern",
        description="Dataset profile and warnings.",
        supports_export_html=True,
    ),
    "run": WorkflowSpec(
        workflow_id="run",
        label="Run Pipeline",
        family="modern",
        description="Holdout-first pipeline for A/B profile.",
        supports_budget=True,
    ),
    "compare": WorkflowSpec(
        workflow_id="compare",
        label="Compare A/B",
        family="modern",
        description="Compare safe and full on same split.",
        supports_budget=True,
    ),
    "eda": WorkflowSpec("eda", "EDA", "legacy", "Legacy stage 1 EDA.", supports_export_html=True),
    "stats": WorkflowSpec("stats", "Stats", "legacy", "Legacy statistical analysis."),
    "models": WorkflowSpec("models", "Models", "legacy", "Legacy ML training.", supports_budget=True),
    "full": WorkflowSpec("full", "Full Pipeline", "legacy", "Legacy full run.", supports_export_html=True, supports_budget=True),
    "review": WorkflowSpec("review", "Review", "legacy", "Legacy data review."),
    "standardize": WorkflowSpec("standardize", "Standardize", "legacy", "Legacy standardization."),
    "famd": WorkflowSpec("famd", "FAMD", "legacy", "Legacy FAMD analysis.", supports_export_html=True),
    "split": WorkflowSpec("split", "Split", "legacy", "Legacy split report."),
    "fairness": WorkflowSpec("fairness", "Fairness", "legacy", "Legacy fairness analysis.", supports_export_html=True, supports_budget=True),
    "subgroups": WorkflowSpec("subgroups", "Subgroups", "legacy", "Legacy subgroup analysis.", supports_export_html=True, supports_budget=True),
    "robustness": WorkflowSpec("robustness", "Robustness", "legacy", "Legacy robustness analysis.", supports_export_html=True, supports_budget=True),
    "analysis": WorkflowSpec("analysis", "Advanced Analysis", "legacy", "Legacy fairness + subgroup + robustness.", supports_export_html=True, supports_budget=True),
    "report": WorkflowSpec("report", "Final Report", "legacy", "Generate final Markdown and HTML report.", supports_export_html=True),
}


def list_workflow_specs() -> list[WorkflowSpec]:
    return list(WORKFLOW_SPECS.values())


def execute_workflow(request: WorkflowRequest) -> WorkflowResult:
    spec = WORKFLOW_SPECS[request.workflow_id]
    before_html = scan_html_artifacts()

    if spec.family == "modern":
        result = _run_modern_workflow(request)
    else:
        result = _run_legacy_workflow(request)

    after_html = scan_html_artifacts()
    result.html_artifacts = _new_paths(before_html, after_html)
    if not result.html_artifacts:
        result.html_artifacts = [str(path) for path in after_html[:10]]
    console_log = _persist_console_result(request, result)
    if console_log is not None:
        result.artifacts["console_log"] = str(console_log)
        _attach_console_log_to_json_artifacts(result, console_log)
    return result


def scan_html_artifacts(
    search_roots: tuple[Path, ...] = RESULT_DIRS,
    *,
    exclude_parts: tuple[str, ...] = ("app_cache", "test-fixtures", "__pycache__"),
) -> list[Path]:
    html_files: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for current, dirnames, filenames in os.walk(root, onerror=lambda _exc: None):
            current_path = Path(current)
            if exclude_parts:
                dirnames[:] = [name for name in dirnames if name not in exclude_parts]
            for filename in filenames:
                if not filename.lower().endswith(".html"):
                    continue
                path = current_path / filename
                if path.is_file():
                    html_files.append(path)

    unique_paths = {path.resolve() for path in html_files}

    def mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    return sorted(
        unique_paths,
        key=mtime,
        reverse=True,
    )


def scan_json_artifacts(
    search_roots: tuple[Path, ...] = RESULT_DIRS,
    *,
    exclude_parts: tuple[str, ...] = ("app_cache", "test-fixtures", "__pycache__"),
) -> list[Path]:
    json_files: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if not path.is_file():
                continue
            if exclude_parts and any(part in exclude_parts for part in path.parts):
                continue
            json_files.append(path)
    return sorted(
        {path.resolve() for path in json_files},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def scan_log_artifacts(
    search_roots: tuple[Path, ...] = LOG_DIRS,
    *,
    exclude_parts: tuple[str, ...] = ("test-fixtures", "__pycache__"),
) -> list[Path]:
    log_files: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.log"):
            if not path.is_file():
                continue
            if exclude_parts and any(part in exclude_parts for part in path.parts):
                continue
            log_files.append(path)
    return sorted(
        {path.resolve() for path in log_files},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def open_html_artifact(path: str | Path) -> Path:
    html_path = Path(path).expanduser().resolve()
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    if os.name == "nt":
        try:
            os.startfile(str(html_path))  # type: ignore[attr-defined]
            return html_path
        except OSError:
            pass

    uri = html_path.as_uri()
    opened = webbrowser.open(uri)
    if not opened:
        raise RuntimeError(f"Failed to open HTML artifact: {html_path}")
    return html_path


def latest_html_artifact(candidates: list[str] | None = None) -> str | None:
    if candidates:
        for item in candidates:
            if Path(item).exists():
                return str(Path(item).resolve())
    scanned = scan_html_artifacts()
    return str(scanned[0]) if scanned else None


def latest_json_artifact(candidates: list[str] | None = None) -> str | None:
    if candidates:
        for item in candidates:
            if Path(item).exists():
                return str(Path(item).resolve())
    scanned = scan_json_artifacts()
    return str(scanned[0]) if scanned else None


def latest_log_artifact(candidates: list[str] | None = None) -> str | None:
    if candidates:
        for item in candidates:
            if Path(item).exists():
                return str(Path(item).resolve())
    scanned = scan_log_artifacts()
    return str(scanned[0]) if scanned else None


def describe_json_artifact(path: str | Path) -> str:
    artifact_path = Path(path).resolve()
    workflow_id = _infer_history_workflow_id(artifact_path)
    modified = datetime.fromtimestamp(artifact_path.stat().st_mtime).strftime("%m-%d %H:%M")
    return f"{modified} | {workflow_id:<10} | {artifact_path.name}"


def describe_log_artifact(path: str | Path) -> str:
    artifact_path = Path(path).resolve()
    metadata = _parse_console_log_metadata(_safe_read_text(artifact_path, max_chars=800))
    workflow_id = str(metadata.get("workflow", _infer_log_workflow_id(artifact_path)))
    modified = datetime.fromtimestamp(artifact_path.stat().st_mtime).strftime("%m-%d %H:%M")
    return f"{modified} | {workflow_id:<10} | {artifact_path.name}"


def load_history_result(path: str | Path) -> WorkflowResult:
    json_path = Path(path).expanduser().resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    workflow_id = _infer_history_workflow_id(json_path, data)
    family = WORKFLOW_SPECS.get(workflow_id, WorkflowSpec(workflow_id, workflow_id.title(), "legacy", "history")).family
    artifacts = {"history_json": str(json_path)}
    if isinstance(data, dict) and isinstance(data.get("artifacts"), dict):
        artifacts.update({str(key): str(value) for key, value in data["artifacts"].items()})

    html_artifacts = _collect_artifact_paths(data, suffixes={".html"}, base_dir=json_path.parent)
    summary = _history_summary(workflow_id, data, json_path)

    payload = data if workflow_id in {"profile", "run", "compare"} else None
    transcript = ""
    if payload is None:
        transcript = json.dumps(data, indent=2, ensure_ascii=False)

    return WorkflowResult(
        workflow_id=workflow_id,
        family=family,
        payload=payload,
        transcript=transcript,
        artifacts=artifacts,
        html_artifacts=html_artifacts,
        summary=summary,
    )


def load_console_log_result(path: str | Path) -> WorkflowResult:
    log_path = Path(path).expanduser().resolve()
    if not log_path.exists():
        raise FileNotFoundError(f"Console log not found: {log_path}")

    text = _repair_mojibake(log_path.read_text(encoding="utf-8", errors="replace"))
    metadata = _parse_console_log_metadata(text)
    workflow_id = str(metadata.get("workflow", _infer_log_workflow_id(log_path)))
    family = str(metadata.get("family", WORKFLOW_SPECS.get(workflow_id, WorkflowSpec(workflow_id, workflow_id.title(), "legacy", "log")).family))
    transcript = _extract_console_log_body(text)
    summary = {
        "log_file": log_path.name,
        "workflow": workflow_id,
        "family": family,
        "dataset": metadata.get("dataset"),
        "variant": metadata.get("variant"),
        "preset": metadata.get("preset"),
        "training_budget_mode": metadata.get("training_budget_mode"),
        "timestamp": metadata.get("timestamp"),
    }
    html_artifacts = _collect_artifact_paths(text, suffixes={".html"}, base_dir=log_path.parent)
    analysis = analyze_console_log(workflow_id, transcript, html_artifacts=html_artifacts)
    summary["assessment_rows"] = analysis["assessment_rows"]
    summary["benchmark_rows"] = analysis["benchmark_rows"]
    return WorkflowResult(
        workflow_id=workflow_id,
        family="log",
        transcript=transcript,
        artifacts={"console_log": str(log_path)},
        html_artifacts=html_artifacts,
        summary={key: value for key, value in summary.items() if value is not None and value != ""},
    )


def analyze_console_log(
    workflow_id: str,
    transcript: str,
    *,
    html_artifacts: list[str] | None = None,
) -> dict[str, list[tuple[str, str]]]:
    transcript = _repair_mojibake(transcript)
    html_artifacts = html_artifacts or []
    normalized = transcript.lower()

    if workflow_id == "eda":
        return _analyze_eda_console_log(transcript, html_artifacts=html_artifacts)
    if workflow_id == "profile":
        return _analyze_profile_console_log(transcript, html_artifacts=html_artifacts)
    if workflow_id in {"run", "models", "full"}:
        return _analyze_training_console_log(workflow_id, transcript, html_artifacts=html_artifacts)
    if workflow_id == "compare":
        return _analyze_compare_console_log(transcript, html_artifacts=html_artifacts)
    if workflow_id in {"fairness", "subgroups", "robustness", "analysis"}:
        return _analyze_advanced_console_log(workflow_id, transcript, html_artifacts=html_artifacts)
    if workflow_id in {"review", "stats", "split", "famd", "standardize", "report"}:
        return _analyze_stage_console_log(workflow_id, transcript, html_artifacts=html_artifacts)
    return _analyze_generic_console_log(workflow_id, transcript, html_artifacts=html_artifacts)

    assessment_rows = [
        ("outcome", "completed successfully" if "workflow failed" not in normalized else "failed"),
        ("artifact_coverage", f"{len(html_artifacts)} html artifacts linked"),
        ("warning_markers", str(transcript.count("⚠") + transcript.count("[Quality Gate]"))),
        ("readiness", "review log body for workflow-specific conclusions"),
    ]
    benchmark_rows = [
        ("log_lines", str(len(transcript.splitlines()))),
        ("log_chars", str(len(transcript))),
        ("html_artifacts", str(len(html_artifacts))),
    ]
    return {"assessment_rows": assessment_rows, "benchmark_rows": benchmark_rows}


def resolve_variant_profile(variant: str) -> RunProfile:
    return RunProfile.SAFE if variant.upper() == "A" else RunProfile.FULL


def resolve_variant_label(variant: str) -> str:
    variant_name = variant.upper()
    return "A / safe / conservative" if variant_name == "A" else "B / full / default"


def _run_modern_workflow(request: WorkflowRequest) -> WorkflowResult:
    bundle = load_dataset(request.dataset_path)
    profile = resolve_variant_profile(request.variant)
    preset = RunPreset(request.preset)
    budget_mode = request.training_budget_mode

    if request.workflow_id == "profile":
        artifact_policy = (
            ArtifactPolicy.CONSOLE_ONLY
            if request.console_only
            else (ArtifactPolicy.FULL_EXPORT if request.export_html else ArtifactPolicy.JSON)
        )
        report = profile_dataset(
            bundle=bundle,
            artifact_policy=artifact_policy,
            export_html=request.export_html,
            output_dir=request.output_dir,
        )
        return WorkflowResult(
            workflow_id=request.workflow_id,
            family="modern",
            payload=report,
            artifacts=report.artifacts,
            summary={"variant": resolve_variant_label(request.variant)},
        )

    if request.workflow_id == "compare":
        artifact_policy = (
            ArtifactPolicy.CONSOLE_ONLY
            if request.console_only
            else (ArtifactPolicy.FULL_EXPORT if request.export_html else ArtifactPolicy.JSON)
        )
        report = compare_profiles(
            bundle=bundle,
            preset=preset,
            artifact_policy=artifact_policy,
            output_dir=request.output_dir,
            training_budget_mode=budget_mode,
        )
        return WorkflowResult(
            workflow_id=request.workflow_id,
            family="modern",
            payload=report,
            artifacts=report.artifacts,
            summary={"preset": preset.value, "budget": budget_mode},
        )

    policy = ArtifactPolicy.CONSOLE_ONLY if request.console_only else ArtifactPolicy.JSON
    config = RunConfig(training_budget_mode=budget_mode)
    report = run_pipeline(
        bundle=bundle,
        profile=profile,
        preset=preset,
        artifact_policy=policy,
        output_dir=request.output_dir,
        config=config,
    )
    return WorkflowResult(
        workflow_id=request.workflow_id,
        family="modern",
        payload=report,
        artifacts=report.artifacts,
        summary={"variant": resolve_variant_label(request.variant), "budget": budget_mode},
    )


def _run_legacy_workflow(request: WorkflowRequest) -> WorkflowResult:
    main_module = importlib.import_module("main")
    target = getattr(main_module, "main")
    transcript = io.StringIO()
    conservative = request.variant.upper() == "A"
    kwargs = {
        "dataset_path": request.dataset_path,
        "run_ethical": True,
        "training_budget_mode": request.training_budget_mode,
    }
    kwargs.update(_legacy_flags_for(request.workflow_id, request.export_html, conservative))

    with contextlib.redirect_stdout(transcript), contextlib.redirect_stderr(transcript):
        target(**kwargs)

    artifacts = {
        "dataset": str(Path(request.dataset_path).resolve()),
        "variant": resolve_variant_label(request.variant),
    }
    return WorkflowResult(
        workflow_id=request.workflow_id,
        family="legacy",
        transcript=transcript.getvalue().strip(),
        artifacts=artifacts,
        summary={"budget": request.training_budget_mode, "variant": resolve_variant_label(request.variant)},
    )


def _legacy_flags_for(workflow_id: str, export_html: bool, conservative: bool) -> dict[str, Any]:
    flags: dict[str, Any] = {
        "run_eda_flag": False,
        "run_stats": False,
        "run_models": False,
        "run_leakage": False,
        "run_review": False,
        "run_standardize": False,
        "run_famd": False,
        "run_split": False,
        "conservative": conservative,
        "run_fairness": False,
        "run_subgroups": False,
        "run_robustness": False,
        "run_report": False,
    }

    if workflow_id == "eda":
        flags["run_eda_flag"] = True
    elif workflow_id == "stats":
        flags["run_stats"] = True
    elif workflow_id == "models":
        flags["run_models"] = True
    elif workflow_id == "full":
        flags["run_eda_flag"] = True
        flags["run_stats"] = True
        flags["run_models"] = True
        flags["run_review"] = True
    elif workflow_id == "review":
        flags["run_review"] = True
    elif workflow_id == "standardize":
        flags["run_standardize"] = True
    elif workflow_id == "famd":
        flags["run_famd"] = True
    elif workflow_id == "split":
        flags["run_split"] = True
    elif workflow_id == "fairness":
        flags["run_fairness"] = True
    elif workflow_id == "subgroups":
        flags["run_subgroups"] = True
    elif workflow_id == "robustness":
        flags["run_robustness"] = True
    elif workflow_id == "analysis":
        flags["run_fairness"] = True
        flags["run_subgroups"] = True
        flags["run_robustness"] = True
    elif workflow_id == "report":
        flags["run_report"] = True
    else:
        raise ValueError(f"Unsupported legacy workflow: {workflow_id}")

    if export_html and workflow_id == "profile":
        flags["run_eda_flag"] = True
    return flags


def _infer_history_workflow_id(path: Path, data: Any | None = None) -> str:
    if isinstance(data, dict):
        if {"config", "dataset", "split", "models"} <= set(data):
            return "run"
        if {"preset", "profiles", "summary"} <= set(data):
            return "compare"
        if "summary" in data and "warnings" in data:
            return "profile"

    stem = path.stem.lower()
    if stem.startswith("run_"):
        return "run"
    if stem.startswith("compare_profiles"):
        return "compare"
    if stem in {"dataset_summary", "dataset_profile", "eda_data_profile"}:
        return "profile"
    if stem.startswith("fairness_"):
        return "fairness"
    if stem.startswith("subgroup_"):
        return "subgroups"
    if stem.startswith("robustness_"):
        return "robustness"
    if stem.startswith("split_report"):
        return "split"
    if stem.startswith("leakage_investigation"):
        return "review"
    if stem.startswith("model_results_"):
        return "models"
    if stem.startswith("model_comparison_report"):
        return "report"
    if stem.startswith("gam_interpretation") or stem.startswith("research_"):
        return "analysis"
    return "report"


def _infer_log_workflow_id(path: Path) -> str:
    stem = path.stem.lower()
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0].isdigit():
        return parts[2]
    if len(parts) >= 2 and parts[0].isdigit():
        return parts[1]
    return "report"


def _history_summary(workflow_id: str, data: Any, json_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "history_file": json_path.name,
        "workflow": workflow_id,
    }

    if not isinstance(data, dict):
        summary["payload_type"] = type(data).__name__
        return summary

    if workflow_id == "run":
        config = data.get("config", {}) if isinstance(data.get("config"), dict) else {}
        dataset = data.get("dataset", {}) if isinstance(data.get("dataset"), dict) else {}
        split = data.get("split", {}) if isinstance(data.get("split"), dict) else {}
        summary.update(
            {
                "profile": config.get("profile"),
                "preset": config.get("preset"),
                "budget": config.get("training_budget_mode"),
                "rows": dataset.get("rows"),
                "test_size": split.get("test_size"),
                "models": ", ".join(config.get("models", [])) if isinstance(config.get("models"), list) else config.get("models"),
            }
        )
        return summary

    if workflow_id == "compare":
        report_summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
        summary.update(
            {
                "preset": data.get("preset"),
                "models": ", ".join(report_summary.keys()),
                "delta_count": len(report_summary),
            }
        )
        return summary

    if workflow_id == "profile":
        profile_summary = data.get("summary", data)
        if isinstance(profile_summary, dict):
            rows = profile_summary.get("rows")
            cols = profile_summary.get("cols")
            if rows is None and isinstance(data.get("shape"), dict):
                rows = data["shape"].get("rows")
                cols = data["shape"].get("cols")
            summary.update(
                {
                    "rows": rows,
                    "cols": cols,
                    "warning_count": len(data.get("warnings", [])) if isinstance(data.get("warnings"), list) else 0,
                }
            )
        return summary

    summary["keys"] = ", ".join(list(data.keys())[:8])
    return summary


def _analyze_generic_console_log(
    workflow_id: str,
    transcript: str,
    *,
    html_artifacts: list[str],
) -> dict[str, list[tuple[str, str]]]:
    html_count = len(html_artifacts) or len(_artifact_mentions(transcript, {".html"}))
    json_count = len(_artifact_mentions(transcript, {".json"}))
    warning_count = _warning_line_count(transcript)
    durations = _extract_completed_seconds(transcript)
    workflow_label = _workflow_label_vi(workflow_id)
    success = not _looks_like_failure(transcript)

    assessment_rows = [
        ("kết luận", f"{workflow_label} {'đã hoàn tất' if success else 'đã gặp lỗi'}"),
        ("độ phủ", f"{html_count} HTML, {json_count} JSON liên quan"),
        ("cảnh báo", f"{warning_count} dòng cảnh báo hoặc quality gate"),
        ("khuyến nghị", _workflow_recommendation_vi(workflow_id)),
    ]
    benchmark_rows = [
        ("dòng log", f"{len(transcript.splitlines()):,}"),
        ("ký tự log", f"{len(transcript):,}"),
        ("HTML", str(html_count)),
        ("JSON", str(json_count)),
        ("mốc hoàn tất", str(len(durations))),
        ("thời gian cuối", _fmt_optional_float(durations[-1] if durations else None)),
    ]
    return {"assessment_rows": assessment_rows, "benchmark_rows": benchmark_rows}


def _analyze_profile_console_log(
    transcript: str,
    *,
    html_artifacts: list[str],
) -> dict[str, list[tuple[str, str]]]:
    summary = _extract_first_literal_dict(transcript)
    rows = _to_intish(summary.get("rows")) if summary else None
    cols = _to_intish(summary.get("cols")) if summary else None
    positive_rate = _to_floatish(summary.get("target_positive_rate")) if summary else None
    safe_columns = summary.get("selected_columns_safe", []) if isinstance(summary, dict) else []
    full_columns = summary.get("selected_columns_full", []) if isinstance(summary, dict) else []
    warning_count = _warning_line_count(transcript)
    html_count = len(html_artifacts) or len(_artifact_mentions(transcript, {".html"}))

    assessment_rows = [
        ("kết luận", "Hồ sơ dữ liệu đã được dựng lại từ log"),
        (
            "cột mô hình",
            f"safe={len(safe_columns) if isinstance(safe_columns, list) else 0}, "
            f"full={len(full_columns) if isinstance(full_columns, list) else 0}",
        ),
        ("cảnh báo", f"{warning_count} cảnh báo về leakage, missing hoặc category hiếm"),
        ("sẵn sàng", "Có thể chuyển sang run, compare hoặc workflow legacy"),
    ]
    benchmark_rows = [
        ("số dòng", _fmt_optional_int(rows)),
        ("số cột", _fmt_optional_int(cols)),
        ("tỷ lệ dương", _fmt_optional_percent(positive_rate)),
        ("HTML", str(html_count)),
        ("cảnh báo", str(warning_count)),
    ]
    return {"assessment_rows": assessment_rows, "benchmark_rows": benchmark_rows}


def _analyze_training_console_log(
    workflow_id: str,
    transcript: str,
    *,
    html_artifacts: list[str],
) -> dict[str, list[tuple[str, str]]]:
    metrics = _extract_model_metrics(transcript)
    warning_count = _warning_line_count(transcript)
    html_count = len(html_artifacts) or len(_artifact_mentions(transcript, {".html"}))
    workflow_label = _workflow_label_vi(workflow_id)

    if not metrics:
        return _analyze_generic_console_log(workflow_id, transcript, html_artifacts=html_artifacts)

    ordered = sorted(metrics, key=lambda item: item["roc_auc"] if item["roc_auc"] is not None else -1.0, reverse=True)
    best = ordered[0]
    weakest = ordered[-1]
    spread = None
    if best["roc_auc"] is not None and weakest["roc_auc"] is not None:
        spread = best["roc_auc"] - weakest["roc_auc"]

    readiness = "Có thể dùng kết quả này để chọn model hoặc chuyển sang fairness/robustness."
    if best["roc_auc"] is not None and best["roc_auc"] >= 0.9:
        readiness = "Điểm rất cao; nên kiểm tra leakage trước khi chốt kết luận."

    assessment_rows = [
        ("kết luận", f"{workflow_label} đã hoàn tất"),
        ("mô hình mạnh nhất", _format_model_metric_line(best)),
        ("mô hình yếu nhất", _format_model_metric_line(weakest)),
        ("độ phủ", f"{len(metrics)} mô hình, {html_count} HTML, {warning_count} cảnh báo"),
        ("khuyến nghị", readiness),
    ]
    benchmark_rows = [
        ("số mô hình", str(len(metrics))),
        ("roc_auc tốt nhất", _fmt_metric(best["roc_auc"])),
        ("f1 tốt nhất", _fmt_metric(_best_metric(metrics, "f1"))),
        ("độ lệch roc_auc", _fmt_metric(spread)),
        ("HTML", str(html_count)),
        ("cảnh báo", str(warning_count)),
    ]
    return {"assessment_rows": assessment_rows, "benchmark_rows": benchmark_rows}


def _analyze_compare_console_log(
    transcript: str,
    *,
    html_artifacts: list[str],
) -> dict[str, list[tuple[str, str]]]:
    summary = _extract_first_literal_dict(transcript)
    if not isinstance(summary, dict):
        return _analyze_generic_console_log("compare", transcript, html_artifacts=html_artifacts)

    rows: list[dict[str, float | str | None]] = []
    for model_name, metrics in summary.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "model": str(model_name),
                "safe_roc_auc": _to_floatish(metrics.get("safe_roc_auc")),
                "full_roc_auc": _to_floatish(metrics.get("full_roc_auc")),
                "roc_delta": _to_floatish(metrics.get("roc_auc_delta_full_minus_safe")),
                "safe_f1": _to_floatish(metrics.get("safe_f1")),
                "full_f1": _to_floatish(metrics.get("full_f1")),
                "f1_delta": _to_floatish(metrics.get("f1_delta_full_minus_safe")),
            }
        )

    if not rows:
        return _analyze_generic_console_log("compare", transcript, html_artifacts=html_artifacts)

    best_gain = max(rows, key=lambda item: item["roc_delta"] if item["roc_delta"] is not None else float("-inf"))
    worst_gain = min(rows, key=lambda item: item["roc_delta"] if item["roc_delta"] is not None else float("inf"))
    html_count = len(html_artifacts) or len(_artifact_mentions(transcript, {".html"}))
    recommendation = "Profile A/safe vẫn nên là baseline nếu chênh lệch không thật sự lớn."
    if (best_gain["roc_delta"] or 0.0) >= 0.03:
        recommendation = "Profile B/full tăng điểm rõ hơn, nhưng cần soi leakage trước khi dùng."
    elif (best_gain["roc_delta"] or 0.0) <= 0:
        recommendation = "Profile A/safe đang an toàn hơn mà không thua điểm rõ rệt."

    assessment_rows = [
        ("kết luận", "So sánh A/B trên cùng split đã hoàn tất"),
        ("tăng roc_auc lớn nhất", _format_compare_delta(best_gain)),
        ("tụt roc_auc lớn nhất", _format_compare_delta(worst_gain)),
        ("độ phủ", f"{len(rows)} mô hình, {html_count} HTML"),
        ("khuyến nghị", recommendation),
    ]
    benchmark_rows = [
        ("số mô hình", str(len(rows))),
        ("delta roc_auc tốt nhất", _fmt_signed_metric(best_gain["roc_delta"])),
        ("delta roc_auc thấp nhất", _fmt_signed_metric(worst_gain["roc_delta"])),
        ("delta f1 tốt nhất", _fmt_signed_metric(_best_metric(rows, "f1_delta"))),
        ("HTML", str(html_count)),
    ]
    return {"assessment_rows": assessment_rows, "benchmark_rows": benchmark_rows}


def _analyze_advanced_console_log(
    workflow_id: str,
    transcript: str,
    *,
    html_artifacts: list[str],
) -> dict[str, list[tuple[str, str]]]:
    html_count = len(html_artifacts) or len(_artifact_mentions(transcript, {".html"}))
    json_count = len(_artifact_mentions(transcript, {".json"}))
    warning_count = _warning_line_count(transcript)
    durations = _extract_completed_seconds(transcript)
    model_names = _extract_advanced_model_names(workflow_id, transcript)
    workflow_label = _workflow_label_vi(workflow_id)

    if workflow_id == "analysis":
        stage_count = sum(
            1
            for marker in ("fairness analysis", "subgroup analysis", "robustness analysis")
            if marker in transcript.casefold()
        )
        assessment_rows = [
            ("kết luận", f"{workflow_label} đã hoàn tất"),
            ("độ phủ", f"{stage_count} khối phân tích, {html_count} HTML, {json_count} JSON"),
            ("mô hình", f"{len(model_names)} mô hình được quét lại"),
            ("cảnh báo", f"{warning_count} dòng cảnh báo hoặc quality gate"),
            ("khuyến nghị", _workflow_recommendation_vi(workflow_id)),
        ]
        benchmark_rows = [
            ("khối phân tích", str(stage_count)),
            ("số mô hình", str(len(model_names))),
            ("HTML", str(html_count)),
            ("JSON", str(json_count)),
            ("mốc hoàn tất", str(len(durations))),
            ("thời gian cuối", _fmt_optional_float(durations[-1] if durations else None)),
        ]
        return {"assessment_rows": assessment_rows, "benchmark_rows": benchmark_rows}

    assessment_rows = [
        ("kết luận", f"{workflow_label} đã hoàn tất"),
        ("mô hình", f"{len(model_names)} mô hình được phân tích"),
        ("độ phủ", f"{html_count} HTML, {json_count} JSON liên quan"),
        ("cảnh báo", f"{warning_count} dòng cảnh báo hoặc quality gate"),
        ("khuyến nghị", _workflow_recommendation_vi(workflow_id)),
    ]
    benchmark_rows = [
        ("số mô hình", str(len(model_names))),
        ("HTML", str(html_count)),
        ("JSON", str(json_count)),
        ("cảnh báo", str(warning_count)),
        ("mốc hoàn tất", str(len(durations))),
        ("thời gian cuối", _fmt_optional_float(durations[-1] if durations else None)),
    ]
    return {"assessment_rows": assessment_rows, "benchmark_rows": benchmark_rows}


def _analyze_stage_console_log(
    workflow_id: str,
    transcript: str,
    *,
    html_artifacts: list[str],
) -> dict[str, list[tuple[str, str]]]:
    html_count = len(html_artifacts) or len(_artifact_mentions(transcript, {".html"}))
    json_count = len(_artifact_mentions(transcript, {".json"}))
    markdown_count = len(_artifact_mentions(transcript, {".md"}))
    warning_count = _warning_line_count(transcript)
    durations = _extract_completed_seconds(transcript)
    workflow_label = _workflow_label_vi(workflow_id)
    success = not _looks_like_failure(transcript)

    assessment_rows = [
        ("kết luận", f"{workflow_label} {'đã hoàn tất' if success else 'đã gặp lỗi'}"),
        ("độ phủ", f"{html_count} HTML, {json_count} JSON, {markdown_count} Markdown"),
        ("cảnh báo", f"{warning_count} dòng cảnh báo hoặc quality gate"),
        ("khuyến nghị", _workflow_recommendation_vi(workflow_id)),
    ]
    benchmark_rows = [
        ("HTML", str(html_count)),
        ("JSON", str(json_count)),
        ("Markdown", str(markdown_count)),
        ("cảnh báo", str(warning_count)),
        ("mốc hoàn tất", str(len(durations))),
        ("thời gian cuối", _fmt_optional_float(durations[-1] if durations else None)),
    ]
    return {"assessment_rows": assessment_rows, "benchmark_rows": benchmark_rows}


def _analyze_eda_console_log(
    transcript: str,
    *,
    html_artifacts: list[str],
) -> dict[str, list[tuple[str, str]]]:
    rows, cols = _extract_shape_flexible(transcript)
    positive_rate = _extract_percent(r"\(1\):\s*[\d,]+\s*\(([\d.]+)%\)", transcript)
    leakage_rate = _extract_percent(r"Suicidal thoughts\s*=\s*Yes.*?=\s*([\d.]+)%", transcript, flags=re.IGNORECASE)
    visualizations = _extract_int(r"(\d+)\s+visualizations saved", transcript)
    eda_seconds = _extract_float(r"Exploratory Data Analysis completed in\s*([\d.]+)s", transcript)
    review_seconds = _extract_float(r"Data Review completed in\s*([\d.]+)s", transcript)
    missing_columns = _extract_int(r"\[Quality Gate\]\s*(\d+)\s*.*?(giá trị thiếu|missing)", transcript, flags=re.IGNORECASE)
    rare_category_columns = _extract_int(r"\[Quality Gate\]\s*(\d+)\s*.*?(category hiếm|rare categor)", transcript, flags=re.IGNORECASE)
    missing_columns = _extract_int(r"\[Quality Gate\]\s*(\d+)\s*cột có tổng", transcript)
    rare_category_columns = _extract_int(r"\[Quality Gate\]\s*(\d+)\s*cột có category hiếm", transcript)
    quality_gate_count = _warning_line_count(transcript)
    missing_columns = _extract_int(r"\[Quality Gate\]\s*(\d+)\s*.*?(giá trị thiếu|missing)", transcript, flags=re.IGNORECASE) or missing_columns
    rare_category_columns = (
        _extract_int(r"\[Quality Gate\]\s*(\d+)\s*.*?(category hiếm|rare categor)", transcript, flags=re.IGNORECASE)
        or rare_category_columns
    )
    missing_columns = missing_columns or _extract_int(r"\[Quality Gate\]\s*(\d+)\s*cá»™t cÃ³ tá»•ng", transcript)
    rare_category_columns = rare_category_columns or _extract_int(r"\[Quality Gate\]\s*(\d+)\s*cá»™t cÃ³ category hiáº¿m", transcript)

    leakage_posture = "cảnh báo cao"
    if leakage_rate is None:
        leakage_posture = "chưa phát hiện"
    elif leakage_rate < 40:
        leakage_posture = "cảnh báo thấp"
    elif leakage_rate < 60:
        leakage_posture = "cảnh báo vừa"

    split_readiness = "Sẵn sàng sang bước chia train/test phân tầng"
    if positive_rate is not None and (positive_rate > 55 or positive_rate < 45):
        split_readiness = "Nên dùng stratified split và kiểm tra class weighting"

    data_quality = "Dữ liệu đủ sạch để sang bước tiếp theo"
    quality_parts: list[str] = []
    if missing_columns is not None:
        quality_parts.append(f"{missing_columns} cột có thiếu dữ liệu")
    if rare_category_columns is not None:
        quality_parts.append(f"{rare_category_columns} cột có category hiếm")
    if quality_parts:
        data_quality = ", ".join(quality_parts)

    assessment_rows = [
        ("kết luận", "EDA đã hoàn tất"),
        ("độ phủ", f"{visualizations or len(html_artifacts)} dashboard HTML + 1 hồ sơ dữ liệu JSON"),
        (
            "rò rỉ nhãn",
            f"{leakage_posture}; Suicidal thoughts = Yes ở mức {_fmt_optional_percent(leakage_rate)}",
        ),
        ("chất lượng dữ liệu", data_quality),
        ("sẵn sàng", split_readiness),
    ]

    benchmark_rows = [
        ("số dòng", _fmt_optional_int(rows)),
        ("số cột", _fmt_optional_int(cols)),
        ("tỷ lệ dương", _fmt_optional_percent(positive_rate)),
        ("số HTML", _fmt_optional_int(visualizations or len(html_artifacts))),
        ("thời gian EDA", _fmt_optional_float(eda_seconds)),
        ("thời gian review", _fmt_optional_float(review_seconds)),
        ("quality gate", _fmt_optional_int(quality_gate_count)),
    ]
    if missing_columns is not None:
        benchmark_rows.append(("cột thiếu dữ liệu", str(missing_columns)))
    if rare_category_columns is not None:
        benchmark_rows.append(("cột category hiếm", str(rare_category_columns)))

    return {"assessment_rows": assessment_rows, "benchmark_rows": benchmark_rows}


def _collect_artifact_paths(value: Any, *, suffixes: set[str], base_dir: Path) -> list[str]:
    collected: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for item in node.values():
                visit(item)
            return
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, str):
            return

        candidate = _resolve_artifact_candidate(node, base_dir)
        if candidate is None:
            return
        if candidate.suffix.lower() not in suffixes:
            return
        collected.append(str(candidate))

    visit(value)
    return list(dict.fromkeys(collected))


def _repair_mojibake(text: str) -> str:
    suspicious = text.count("Ã") + text.count("â") + text.count("ð") + text.count("�")
    if suspicious < 3:
        return text
    try:
        repaired = text.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
    except UnicodeError:
        return text
    repaired_score = repaired.count("Ã") + repaired.count("â") + repaired.count("ð") + repaired.count("�")
    return repaired if repaired and repaired_score < suspicious else text


def _artifact_mentions(text: str, suffixes: set[str]) -> list[str]:
    hits: list[str] = []
    for match in re.findall(r"[\w./\\-]+\.(?:html|json|md)", text, flags=re.IGNORECASE):
        if Path(match).suffix.lower() in suffixes:
            hits.append(match)
    return list(dict.fromkeys(hits))


def _warning_line_count(text: str) -> int:
    count = 0
    for line in text.splitlines():
        lowered = line.casefold()
        if any(marker in lowered for marker in ("[quality gate]", "warning", "cảnh báo", "⚠")):
            count += 1
    return count


def _extract_completed_seconds(text: str) -> list[float]:
    values: list[float] = []
    for match in re.finditer(r"completed in\s*([\d.]+)s", text, flags=re.IGNORECASE):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    return values


def _looks_like_failure(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in ("workflow failed", "traceback", "exception", "attributeerror", "typeerror"))


def _workflow_label_vi(workflow_id: str) -> str:
    return {
        "profile": "Hồ sơ dữ liệu",
        "run": "Huấn luyện holdout-first",
        "compare": "So sánh A/B",
        "eda": "Khối EDA",
        "stats": "Phân tích thống kê",
        "models": "Khối huấn luyện mô hình",
        "full": "Pipeline đầy đủ",
        "review": "Rà soát dữ liệu",
        "standardize": "Chuẩn hóa dữ liệu",
        "famd": "Phân tích FAMD",
        "split": "Phân tích chia tập",
        "fairness": "Phân tích fairness",
        "subgroups": "Phân tích theo nhóm con",
        "robustness": "Phân tích robustness",
        "analysis": "Phân tích nâng cao",
        "report": "Tạo báo cáo cuối",
    }.get(workflow_id, workflow_id)


def _workflow_recommendation_vi(workflow_id: str) -> str:
    return {
        "profile": "Đối chiếu lại leakage và danh sách cột trước khi huấn luyện.",
        "run": "Ưu tiên so ROC-AUC holdout và kiểm tra feature importance.",
        "compare": "Giữ A/safe làm baseline nếu lợi ích của B/full chưa thật sự đáng kể.",
        "stats": "Dùng kết quả thống kê để chọn giả thuyết và giải thích biến quan trọng.",
        "models": "So tiếp ROC-AUC, F1 và fairness trước khi chốt model.",
        "full": "Đọc lại toàn bộ artifact vì pipeline đầy đủ dễ tạo cảm giác điểm số quá đẹp.",
        "review": "Kiểm tra leakage, thiếu dữ liệu và cột low-variance trước khi train.",
        "standardize": "Xác nhận các biến số đã được scale đúng trước khi diễn giải model.",
        "famd": "Dùng biểu đồ FAMD để xem cấu trúc ẩn và nhóm biến cùng chiều.",
        "split": "Xác nhận stratified split và tỷ lệ dương giữa train/test.",
        "fairness": "Ưu tiên xem chênh lệch theo giới tính, tuổi và tiền sử gia đình.",
        "subgroups": "Tập trung vào nhóm tuổi, giới tính và city có điểm yếu rõ rệt.",
        "robustness": "Đọc kỹ CV stability, noise test và label-flip trước khi triển khai.",
        "analysis": "Tổng hợp fairness, subgroup và robustness trước khi viết kết luận cuối.",
        "report": "Mở HTML cuối và đối chiếu lại các dashboard nguồn trước khi chia sẻ.",
        "eda": "Kiểm tra thêm các dashboard leakage và missing value trước khi train.",
    }.get(workflow_id, "Đọc lại log và artifact để rút ra kết luận cho workflow này.")


def _extract_first_literal_dict(text: str) -> dict[str, Any] | None:
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        candidates = [raw]
        brace_index = raw.find("{")
        if brace_index > 0:
            candidates.append(raw[brace_index:])
        for candidate in candidates:
            try:
                parsed = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _extract_named_literal_dicts(text: str) -> dict[str, dict[str, Any]]:
    named: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        raw = line.strip()
        if not raw or "{" not in raw:
            continue
        brace_index = raw.find("{")
        label = raw[:brace_index].strip(" :-")
        if not label:
            continue
        try:
            parsed = ast.literal_eval(raw[brace_index:])
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, dict):
            named[label.casefold()] = parsed
    return named


def _extract_model_metrics(text: str) -> list[dict[str, float | str | None]]:
    metrics: list[dict[str, float | str | None]] = []

    for label, payload in _extract_named_literal_dicts(text).items():
        holdout = payload.get("holdout") if isinstance(payload.get("holdout"), dict) else payload
        if not isinstance(holdout, dict):
            continue
        roc_auc = _to_floatish(holdout.get("roc_auc"))
        f1 = _to_floatish(holdout.get("f1"))
        if roc_auc is None and f1 is None:
            continue
        metrics.append({"model": label, "roc_auc": roc_auc, "f1": f1})

    if metrics:
        return metrics

    current_model: str | None = None
    legacy: dict[str, dict[str, float | str | None]] = {}
    for line in text.splitlines():
        lowered = line.casefold()
        if "logistic" in lowered:
            current_model = "logistic"
        elif "catboost" in lowered:
            current_model = "catboost"
        elif re.search(r"\bgam\b", lowered):
            current_model = "gam"
        elif "dummy" in lowered:
            current_model = "dummy"

        if current_model is None:
            continue

        roc_match = re.search(r"roc[- ]?auc[:=\s]+([0-9.]+)", line, flags=re.IGNORECASE)
        f1_match = re.search(r"\bf1[:=\s]+([0-9.]+)", line, flags=re.IGNORECASE)
        if not roc_match and not f1_match:
            continue

        row = legacy.setdefault(current_model, {"model": current_model, "roc_auc": None, "f1": None})
        if roc_match and row["roc_auc"] is None:
            row["roc_auc"] = _to_floatish(roc_match.group(1))
        if f1_match and row["f1"] is None:
            row["f1"] = _to_floatish(f1_match.group(1))

    return list(legacy.values())


def _extract_advanced_model_names(workflow_id: str, text: str) -> set[str]:
    if workflow_id == "fairness":
        patterns = (r"Analyzing fairness for (\w+)",)
    elif workflow_id == "subgroups":
        patterns = (r"Analyzing subgroups for (\w+)",)
    elif workflow_id == "robustness":
        patterns = (r"Analyzing robustness for (\w+)",)
    else:
        patterns = (
            r"Analyzing fairness for (\w+)",
            r"Analyzing subgroups for (\w+)",
            r"Analyzing robustness for (\w+)",
        )

    names: set[str] = set()
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            names.add(str(match).lower())
    return names


def _format_model_metric_line(row: dict[str, float | str | None]) -> str:
    return f"{row['model']} | ROC-AUC={_fmt_metric(row.get('roc_auc'))} | F1={_fmt_metric(row.get('f1'))}"


def _format_compare_delta(row: dict[str, float | str | None]) -> str:
    model = str(row.get("model", "?"))
    safe_roc = _fmt_metric(row.get("safe_roc_auc"))
    full_roc = _fmt_metric(row.get("full_roc_auc"))
    delta = _fmt_signed_metric(row.get("roc_delta"))
    return f"{model} | safe={safe_roc} | full={full_roc} | delta={delta}"


def _best_metric(rows: list[dict[str, float | str | None]], key: str) -> float | None:
    values = [float(item[key]) for item in rows if isinstance(item.get(key), (float, int))]
    return max(values) if values else None


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "không có"
    return f"{float(value):.4f}"


def _fmt_signed_metric(value: Any) -> str:
    if value is None:
        return "không có"
    return f"{float(value):+.4f}"


def _extract_shape(text: str) -> tuple[int | None, int | None]:
    match = re.search(r"Shape:\s*([\d,]+)\s*rows\s*[×x]\s*([\d,]+)\s*cols", text)
    if not match:
        return None, None
    return _to_int(match.group(1)), _to_int(match.group(2))


def _extract_shape_flexible(text: str) -> tuple[int | None, int | None]:
    for pattern in (
        r"Shape:\s*([\d,]+)\s*rows\s*[×x]\s*([\d,]+)\s*cols",
        r"Shape:\s*([\d,]+)\s*rows\s*[Ã—x]\s*([\d,]+)\s*cols",
    ):
        match = re.search(pattern, text)
        if match:
            return _to_int(match.group(1)), _to_int(match.group(2))
    return None, None


def _extract_percent(pattern: str, text: str, *, flags: int = 0) -> float | None:
    match = re.search(pattern, text, flags=flags)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _extract_float(pattern: str, text: str, *, flags: int = 0) -> float | None:
    match = re.search(pattern, text, flags=flags)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _extract_int(pattern: str, text: str, *, flags: int = 0) -> int | None:
    match = re.search(pattern, text, flags=flags)
    if not match:
        return None
    return _to_int(match.group(1))


def _to_floatish(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _to_intish(value: Any) -> int | None:
    return _to_int(str(value)) if value is not None else None


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).replace(",", "").strip())
    except ValueError:
        return None


def _fmt_optional_int(value: int | None) -> str:
    return f"{value:,}" if value is not None else "không có"


def _fmt_optional_float(value: float | None) -> str:
    return f"{value:.2f}s" if value is not None else "không có"


def _fmt_optional_percent(value: float | None) -> str:
    return f"{value:.2f}%" if value is not None else "không có"


def _resolve_artifact_candidate(value: str, base_dir: Path) -> Path | None:
    raw = Path(value)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(base_dir / raw)

    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded.exists():
            return expanded.resolve()
    return None


def _new_paths(before: list[Path], after: list[Path]) -> list[str]:
    before_set = {path.resolve() for path in before}
    return [str(path) for path in after if path.resolve() not in before_set]


def _safe_read_text(path: Path, *, max_chars: int | None = None) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if max_chars is None:
        return text
    return text[:max_chars]


def _parse_console_log_metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in text.splitlines():
        if line.strip() == "=== console result ===":
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"workflow", "family", "dataset", "variant", "preset", "training_budget_mode", "timestamp"}:
            metadata[key] = value.strip()
    return metadata


def _extract_console_log_body(text: str) -> str:
    marker = "=== console result ==="
    if marker not in text:
        return text.strip()
    _, _, body = text.partition(marker)
    return body.strip()


def _persist_console_result(request: WorkflowRequest, result: WorkflowResult) -> Path | None:
    console_text = _render_console_result(result, request)
    if not console_text:
        return None

    output_dir = Path(request.output_dir).expanduser()
    log_dir = output_dir / "console_logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    variant = request.variant.upper() if request.variant else "NA"
    filename = f"{timestamp}_{request.workflow_id}_{variant}_{request.preset}_{request.training_budget_mode}.log"
    log_path = log_dir / filename
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path.write_text(console_text, encoding="utf-8")
        return log_path.resolve()
    except OSError:
        return None


def _render_console_result(result: WorkflowResult, request: WorkflowRequest) -> str:
    from .console import print_workflow_result

    header = [
        f"workflow: {request.workflow_id}",
        f"family: {result.family}",
        f"dataset: {request.dataset_path}",
        f"variant: {resolve_variant_label(request.variant)}",
        f"preset: {request.preset}",
        f"training_budget_mode: {request.training_budget_mode}",
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "=== console result ===",
    ]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print_workflow_result(result, console=None)

    body = buffer.getvalue().strip()
    return "\n".join(header + ([body] if body else ["<empty>"])).strip() + "\n"


def _attach_console_log_to_json_artifacts(result: WorkflowResult, console_log: Path) -> None:
    for artifact_key, artifact_value in result.artifacts.items():
        if artifact_key == "console_log" or not str(artifact_value).lower().endswith(".json"):
            continue
        json_path = Path(str(artifact_value)).expanduser()
        if not json_path.exists():
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue

        artifacts = data.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = {}
        artifacts["console_log"] = str(console_log)
        data["artifacts"] = artifacts
        try:
            json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            continue
