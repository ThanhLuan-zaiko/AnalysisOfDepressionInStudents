from __future__ import annotations

from typing import Any
import warnings

warnings.filterwarnings(
    "ignore",
    message="'return' in a 'finally' block",
    category=SyntaxWarning,
    module=r"rich\.segment",
)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:  # pragma: no cover
    Console = None
    Panel = None
    Table = None


PALETTE = {
    "primary": "bold #0F766E",
    "secondary": "#1E3A5F",
    "accent": "#D4A373",
    "danger": "bold #B91C1C",
}

LOTUS_BANNER = """\
    _/\\_
  .(    ).
 /  SEN   \\
 \\  DATA  /
  `-.__.-'
"""


def get_console() -> Any:
    if Console is None:
        return None
    return Console()


def print_banner(console: Any | None = None) -> None:
    if console is None:
        print(LOTUS_BANNER)
        print("Sen Analytics CLI")
        print("Calm screening workflow for Vietnamese depression-analysis research")
        return

    console.print(
        Panel.fit(
            f"[{PALETTE['primary']}]{LOTUS_BANNER}[/{PALETTE['primary']}]\n"
            f"[{PALETTE['secondary']}]Sen Analytics CLI[/]\n"
            "[#475569]Pipeline accuracy, leakage checks, and holdout-first reporting[/]",
            border_style=PALETTE["accent"],
            title="Vietnamese Depression Analysis",
        )
    )


def print_status(message: str, console: Any | None = None) -> None:
    if console is None:
        print(message)
        return
    console.print(f"[{PALETTE['accent']}]{message}[/]")


def prompt_text(
    message: str,
    default: str | None = None,
    console: Any | None = None,
) -> str:
    if console is not None:
        console.print(message, end="")
    else:
        print(message, end="")
    value = input().strip()
    if value:
        return value
    return default or ""


def print_warnings(warnings: list[str], console: Any | None = None) -> None:
    if not warnings:
        return
    if console is None:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
        return

    table = Table(title="Warnings", border_style=PALETTE["danger"], show_header=False)
    table.add_column("Warning", style=PALETTE["danger"])
    for warning in warnings:
        table.add_row(warning)
    console.print(table)


def print_profile_report(report: Any, console: Any | None = None) -> None:
    summary = _value(report, "summary", report if isinstance(report, dict) else {})
    warnings_list = _value(report, "warnings", [])
    if console is None or Table is None:
        print(summary)
        print_warnings(warnings_list, console=None)
        return

    table = Table(title="Dataset Profile", border_style=PALETTE["primary"])
    table.add_column("Field", style=PALETTE["secondary"])
    table.add_column("Value", style="white")
    for key, value in summary.items():
        table.add_row(str(key), str(value))
    console.print(table)
    print_warnings(warnings_list, console)


def print_run_report(report: Any, console: Any | None = None) -> None:
    config = _value(report, "config", {})
    models = _value(report, "models", {})
    warnings_list = _value(report, "warnings", [])
    timing_values = _value(report, "timings", {})

    if console is None or Table is None:
        print(config)
        for name, model in models.items():
            print(name, _value(model, "holdout", {}))
        print_warnings(warnings_list, console=None)
        return

    meta = Table(title="Run Config", border_style=PALETTE["secondary"])
    meta.add_column("Field")
    meta.add_column("Value")
    meta.add_row("profile", str(config.get("profile")))
    meta.add_row("preset", str(config.get("preset")))
    meta.add_row("models", ", ".join(config.get("models", [])))
    meta.add_row("selected_columns", ", ".join(config.get("selected_columns", [])))
    rust_engine = config.get("rust_engine")
    if rust_engine:
        rust_label = "available"
        if not rust_engine.get("available"):
            rust_label = f"fallback ({rust_engine.get('error', 'unavailable')})"
        elif rust_engine.get("version"):
            rust_label = f"available ({rust_engine['version']})"
        meta.add_row("rust_engine", rust_label)
    console.print(meta)

    for model_name, model in models.items():
        metadata = _value(model, "metadata", {})
        oof = _value(model, "oof", {})
        holdout = _value(model, "holdout", {})
        feature_importance = _value(model, "feature_importance", [])
        engine = metadata.get("engine")
        title = f"{model_name.title()} Holdout Metrics"
        if engine:
            title = f"{title} [{engine}]"
        table = Table(title=title, border_style=PALETTE["primary"])
        table.add_column("Metric")
        table.add_column("OOF")
        table.add_column("Holdout")
        for metric in ("roc_auc", "pr_auc", "f1", "recall", "precision", "brier_score"):
            table.add_row(metric, str(oof.get(metric)), str(holdout.get(metric)))
        console.print(table)

        if feature_importance:
            top = Table(title=f"{model_name.title()} Top Features", border_style=PALETTE["accent"])
            top.add_column("Feature")
            top.add_column("Signal")
            for row in feature_importance[:8]:
                signal = row.get(
                    "importance",
                    row.get(
                        "abs_coefficient",
                        row.get("variance_importance", row.get("coefficient")),
                    ),
                )
                top.add_row(str(row["feature"]), str(signal))
            console.print(top)

    timings_table = Table(title="Timings", border_style=PALETTE["secondary"])
    timings_table.add_column("Stage")
    timings_table.add_column("Seconds")
    for key, value in timing_values.items():
        timings_table.add_row(key, str(value))
    console.print(timings_table)
    print_warnings(warnings_list, console)


def print_comparison_report(report: Any, console: Any | None = None) -> None:
    summary = _value(report, "summary", {})
    preset = _value(report, "preset", "?")
    if console is None or Table is None:
        print(summary)
        return

    table = Table(title=f"Profile Comparison ({preset})", border_style=PALETTE["accent"])
    table.add_column("Model")
    table.add_column("Safe ROC-AUC")
    table.add_column("Full ROC-AUC")
    table.add_column("Delta")
    table.add_column("Safe F1")
    table.add_column("Full F1")
    for model_name, metrics in summary.items():
        table.add_row(
            model_name,
            str(metrics["safe_roc_auc"]),
            str(metrics["full_roc_auc"]),
            str(metrics["roc_auc_delta_full_minus_safe"]),
            str(metrics["safe_f1"]),
            str(metrics["full_f1"]),
        )
    console.print(table)


def print_workflow_result(result: Any, console: Any | None = None) -> None:
    payload = getattr(result, "payload", None)
    workflow_id = getattr(result, "workflow_id", "workflow")

    if payload is not None:
        if workflow_id == "profile":
            print_profile_report(payload, console)
            return
        if workflow_id == "compare":
            print_comparison_report(payload, console)
            return
        if workflow_id == "run":
            print_run_report(payload, console)
            return

    transcript = getattr(result, "transcript", "")
    artifacts = getattr(result, "artifacts", {})
    html_artifacts = getattr(result, "html_artifacts", [])

    if console is None or Table is None:
        if transcript:
            print(transcript)
        if artifacts:
            print("Artifacts:")
            for key, value in artifacts.items():
                print(f"- {key}: {value}")
        if html_artifacts:
            print("HTML:")
            for path in html_artifacts[:8]:
                print(f"- {path}")
        return

    if transcript:
        console.print(Panel(transcript, title=f"{workflow_id} transcript", border_style=PALETTE["secondary"]))
    if artifacts:
        table = Table(title=f"{workflow_id} artifacts", border_style=PALETTE["accent"], show_header=False)
        table.add_column("Key")
        table.add_column("Value")
        for key, value in artifacts.items():
            table.add_row(str(key), str(value))
        console.print(table)
    if html_artifacts:
        table = Table(title="HTML Artifacts", border_style=PALETTE["primary"], show_header=False)
        table.add_column("Path")
        for path in html_artifacts[:8]:
            table.add_row(path)
        console.print(table)


def _value(report: Any, key: str, default: Any) -> Any:
    if isinstance(report, dict):
        return report.get(key, default)
    return getattr(report, key, default)
