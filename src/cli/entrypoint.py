from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.app import ArtifactPolicy, RunPreset, RunProfile, compare_profiles, load_dataset, profile_dataset, run_pipeline

from .console import get_console, print_banner, print_comparison_report, print_profile_report, print_run_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sen Analytics CLI")
    subparsers = parser.add_subparsers(dest="command")

    profile_parser = subparsers.add_parser("profile", help="Profile dataset and optionally export HTML EDA")
    profile_parser.add_argument("--dataset", default="Student_Depression_Dataset.csv")
    profile_parser.add_argument("--full-export", action="store_true")
    profile_parser.add_argument("--export-html", action="store_true")
    profile_parser.add_argument("--output-dir", default="results/app")

    run_parser = subparsers.add_parser("run", help="Run the modern holdout-first pipeline")
    run_parser.add_argument("--dataset", default="Student_Depression_Dataset.csv")
    run_parser.add_argument("--profile", choices=[profile.value for profile in RunProfile], default=RunProfile.SAFE.value)
    run_parser.add_argument("--preset", choices=[preset.value for preset in RunPreset], default=RunPreset.QUICK.value)
    run_parser.add_argument("--console-only", action="store_true")
    run_parser.add_argument("--output-dir", default="results/app")

    compare_parser = subparsers.add_parser("compare", help="Compare safe and full profiles on the same split")
    compare_parser.add_argument("--dataset", default="Student_Depression_Dataset.csv")
    compare_parser.add_argument("--preset", choices=[preset.value for preset in RunPreset], default=RunPreset.QUICK.value)
    compare_parser.add_argument("--output-dir", default="results/app")

    return parser


def main(argv: list[str] | None = None) -> int:
    console = get_console()
    parser = build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]

    if not raw_argv:
        tui_exit_code = run_tui([], show_banner=False)
        if tui_exit_code == 0:
            return 0
        raw_argv = ["run"]

    print_banner(console)
    args = parser.parse_args(raw_argv)

    if args.command is None:
        args = parser.parse_args(["run"])

    bundle = load_dataset(args.dataset)

    if args.command == "profile":
        policy = ArtifactPolicy.FULL_EXPORT if args.full_export else ArtifactPolicy.JSON
        report = profile_dataset(
            bundle=bundle,
            artifact_policy=policy,
            export_html=args.export_html,
            output_dir=args.output_dir,
        )
        print_profile_report(report, console)
        return 0

    if args.command == "compare":
        report = compare_profiles(
            bundle=bundle,
            preset=RunPreset(args.preset),
            artifact_policy=ArtifactPolicy.JSON,
            output_dir=args.output_dir,
        )
        print_comparison_report(report, console)
        return 0

    policy = ArtifactPolicy.CONSOLE_ONLY if args.console_only else ArtifactPolicy.JSON
    report = run_pipeline(
        bundle=bundle,
        profile=RunProfile(args.profile),
        preset=RunPreset(args.preset),
        artifact_policy=policy,
        output_dir=args.output_dir,
    )
    print_run_report(report, console)
    return 0


def run_tui(argv: list[str] | None = None, show_banner: bool = True) -> int:
    del argv
    console = get_console()
    try:
        from .textual_app import launch_tui
        launch_tui(default_dataset=Path("Student_Depression_Dataset.csv"))
        return 0
    except ImportError:
        if show_banner:
            print_banner(console)
        if console is None:
            print("Textual is not installed. Use `depression-cli` now, or install the TUI extra later.")
        else:
            console.print(
                "Textual is not installed. Use [bold]depression-cli[/] now, or install the TUI extra later."
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
