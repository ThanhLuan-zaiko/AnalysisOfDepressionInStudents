from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
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
    return result


def scan_html_artifacts(search_roots: tuple[Path, ...] = RESULT_DIRS) -> list[Path]:
    html_files: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        html_files.extend(path for path in root.rglob("*.html") if path.is_file())
    return sorted(
        {path.resolve() for path in html_files},
        key=lambda path: path.stat().st_mtime,
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


def open_html_artifact(path: str | Path) -> Path:
    html_path = Path(path).expanduser().resolve()
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    uri = html_path.as_uri()
    opened = webbrowser.open(uri)
    if not opened and os.name == "nt":
        os.startfile(str(html_path))  # type: ignore[attr-defined]
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


def describe_json_artifact(path: str | Path) -> str:
    artifact_path = Path(path).resolve()
    workflow_id = _infer_history_workflow_id(artifact_path)
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
        report = compare_profiles(
            bundle=bundle,
            preset=preset,
            artifact_policy=ArtifactPolicy.CONSOLE_ONLY if request.console_only else ArtifactPolicy.JSON,
            output_dir=request.output_dir,
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
