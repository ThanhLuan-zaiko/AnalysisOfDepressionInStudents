from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.app import ArtifactPolicy, RunPreset, RunProfile, compare_profiles, load_dataset, profile_dataset, run_pipeline

from .console import (
    get_console,
    print_banner,
    print_comparison_report,
    print_profile_report,
    print_run_report,
    print_status,
    prompt_text,
)

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sen Analytics CLI")
    subparsers = parser.add_subparsers(dest="command")

    profile_parser = subparsers.add_parser("profile", help="Profile dataset and optionally export HTML EDA")
    profile_parser.add_argument("--dataset", default="Student_Depression_Dataset.csv")
    profile_parser.add_argument("--full-export", action="store_true")
    profile_parser.add_argument("--export-html", action="store_true")
    profile_parser.add_argument("--console-only", action="store_true")
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
    compare_parser.add_argument("--console-only", action="store_true")
    compare_parser.add_argument("--output-dir", default="results/app")

    return parser


def _interactive_fallback(console: object | None) -> list[str] | None:
    print_banner(console)
    print_status("Textual chưa được cài. Chuyển sang menu console để bạn chọn lệnh chạy.", console)
    dataset = prompt_text(
        "Dataset [mặc định: Student_Depression_Dataset.csv]: ",
        default="Student_Depression_Dataset.csv",
        console=console,
    )
    if not Path(dataset).exists():
        print_status(f"Không tìm thấy dataset: {dataset}", console)
        return None

    menu_lines = [
        "",
        "Chọn tác vụ:",
        "  1. Hồ sơ dữ liệu",
        "  2. Chạy nhanh an toàn",
        "  3. Chạy nhanh đầy đủ",
        "  4. Chạy nghiên cứu an toàn",
        "  5. So sánh safe và full",
        "  0. Thoát",
    ]
    for line in menu_lines:
        print_status(line, console)

    choice = prompt_text("Nhập lựa chọn [mặc định: 2]: ", default="2", console=console)
    mapping = {
        "1": ["profile", "--dataset", dataset, "--console-only"],
        "2": ["run", "--dataset", dataset, "--profile", "safe", "--preset", "quick", "--console-only"],
        "3": ["run", "--dataset", dataset, "--profile", "full", "--preset", "quick", "--console-only"],
        "4": ["run", "--dataset", dataset, "--profile", "safe", "--preset", "research", "--console-only"],
        "5": ["compare", "--dataset", dataset, "--preset", "quick", "--console-only"],
        "0": None,
    }
    if choice not in mapping:
        print_status("Lựa chọn không hợp lệ.", console)
        return None
    return mapping[choice]


def main(argv: list[str] | None = None) -> int:
    console = get_console()
    parser = build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    banner_rendered = False

    if not raw_argv:
        tui_exit_code = run_tui([], show_banner=False, quiet_on_missing=True)
        if tui_exit_code == 0:
            return 0
        raw_argv = _interactive_fallback(console)
        if raw_argv is None:
            return 0
        banner_rendered = True

    if not banner_rendered:
        print_banner(console)
    args = parser.parse_args(raw_argv)

    if args.command is None:
        args = parser.parse_args(["run"])

    if args.command == "profile":
        print_status(f"Đang đọc dữ liệu từ {args.dataset} ...", console)
        bundle = load_dataset(args.dataset)
        policy = (
            ArtifactPolicy.CONSOLE_ONLY
            if args.console_only
            else (ArtifactPolicy.FULL_EXPORT if args.full_export else ArtifactPolicy.JSON)
        )
        report = profile_dataset(
            bundle=bundle,
            artifact_policy=policy,
            export_html=args.export_html,
            output_dir=args.output_dir,
        )
        print_profile_report(report, console)
        return 0

    if args.command == "compare":
        print_status(f"Đang đọc dữ liệu từ {args.dataset} ...", console)
        bundle = load_dataset(args.dataset)
        print_status("Đang so sánh profile safe và full trên cùng split ...", console)
        report = compare_profiles(
            bundle=bundle,
            preset=RunPreset(args.preset),
            artifact_policy=ArtifactPolicy.CONSOLE_ONLY if args.console_only else ArtifactPolicy.JSON,
            output_dir=args.output_dir,
        )
        print_comparison_report(report, console)
        return 0

    print_status(f"Đang đọc dữ liệu từ {args.dataset} ...", console)
    bundle = load_dataset(args.dataset)
    print_status(
        f"Đang chạy pipeline: profile={args.profile}, preset={args.preset} ...",
        console,
    )
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


def run_tui(
    argv: list[str] | None = None,
    show_banner: bool = True,
    quiet_on_missing: bool = False,
) -> int:
    del argv
    console = get_console()
    try:
        from .textual_app import launch_tui

        launch_tui(default_dataset=Path("Student_Depression_Dataset.csv"))
        return 0
    except ImportError:
        if quiet_on_missing:
            return 1
        if show_banner:
            print_banner(console)
        message = 'Textual chưa được cài. Hãy dùng `robot` hoặc `robot run ...`, hoặc cài nhanh bằng `uv pip install "textual>=0.86.0"`.'
        if console is None:
            print(message)
        else:
            console.print(
                'Textual chưa được cài. Hãy dùng [bold]robot[/] hoặc [bold]robot run ...[/], hoặc cài nhanh bằng [bold]uv pip install "textual>=0.86.0"[/].'
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
