from __future__ import annotations

from typing import Any

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
    if console is None or Table is None:
        print(report.summary)
        print_warnings(report.warnings, console=None)
        return

    table = Table(title="Dataset Profile", border_style=PALETTE["primary"])
    table.add_column("Field", style=PALETTE["secondary"])
    table.add_column("Value", style="white")
    for key, value in report.summary.items():
        table.add_row(str(key), str(value))
    console.print(table)
    print_warnings(report.warnings, console)


def print_run_report(report: Any, console: Any | None = None) -> None:
    if console is None or Table is None:
        print(report.config)
        for name, model in report.models.items():
            print(name, model.holdout)
        print_warnings(report.warnings, console=None)
        return

    meta = Table(title="Run Config", border_style=PALETTE["secondary"])
    meta.add_column("Field")
    meta.add_column("Value")
    meta.add_row("profile", report.config["profile"])
    meta.add_row("preset", report.config["preset"])
    meta.add_row("models", ", ".join(report.config["models"]))
    meta.add_row("selected_columns", ", ".join(report.config["selected_columns"]))
    rust_engine = report.config.get("rust_engine")
    if rust_engine:
        rust_label = "available"
        if not rust_engine.get("available"):
            rust_label = f"fallback ({rust_engine.get('error', 'unavailable')})"
        elif rust_engine.get("version"):
            rust_label = f"available ({rust_engine['version']})"
        meta.add_row("rust_engine", rust_label)
    console.print(meta)

    for model_name, model in report.models.items():
        engine = model.metadata.get("engine")
        title = f"{model_name.title()} Holdout Metrics"
        if engine:
            title = f"{title} [{engine}]"
        table = Table(title=title, border_style=PALETTE["primary"])
        table.add_column("Metric")
        table.add_column("OOF")
        table.add_column("Holdout")
        for metric in ("roc_auc", "pr_auc", "f1", "recall", "precision", "brier_score"):
            table.add_row(metric, str(model.oof.get(metric)), str(model.holdout.get(metric)))
        console.print(table)

        if model.feature_importance:
            top = Table(title=f"{model_name.title()} Top Features", border_style=PALETTE["accent"])
            top.add_column("Feature")
            top.add_column("Signal")
            for row in model.feature_importance[:8]:
                signal = row.get(
                    "importance",
                    row.get(
                        "abs_coefficient",
                        row.get("variance_importance", row.get("coefficient")),
                    ),
                )
                top.add_row(str(row["feature"]), str(signal))
            console.print(top)

    timings = Table(title="Timings", border_style=PALETTE["secondary"])
    timings.add_column("Stage")
    timings.add_column("Seconds")
    for key, value in report.timings.items():
        timings.add_row(key, str(value))
    console.print(timings)
    print_warnings(report.warnings, console)


def print_comparison_report(report: Any, console: Any | None = None) -> None:
    if console is None or Table is None:
        print(report.summary)
        return

    table = Table(title=f"Profile Comparison ({report.preset})", border_style=PALETTE["accent"])
    table.add_column("Model")
    table.add_column("Safe ROC-AUC")
    table.add_column("Full ROC-AUC")
    table.add_column("Delta")
    table.add_column("Safe F1")
    table.add_column("Full F1")
    for model_name, summary in report.summary.items():
        table.add_row(
            model_name,
            str(summary["safe_roc_auc"]),
            str(summary["full_roc_auc"]),
            str(summary["roc_auc_delta_full_minus_safe"]),
            str(summary["safe_f1"]),
            str(summary["full_f1"]),
        )
    console.print(table)
