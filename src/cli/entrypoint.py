from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .console import get_console, print_banner, print_status, print_workflow_result, prompt_text
from .workflows import (
    WORKFLOW_SPECS,
    WorkflowRequest,
    execute_workflow,
    latest_json_artifact,
    latest_html_artifact,
    load_history_result,
    list_workflow_specs,
    open_html_artifact,
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
    _add_common_args(profile_parser, include_variant=False, include_preset=False)
    profile_parser.add_argument("--full-export", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run the modern holdout-first pipeline")
    _add_common_args(run_parser)

    compare_parser = subparsers.add_parser("compare", help="Compare A/B on the same split")
    _add_common_args(compare_parser, include_variant=False)

    task_parser = subparsers.add_parser("task", help="Run any workflow that the TUI exposes")
    task_parser.add_argument("workflow", choices=tuple(WORKFLOW_SPECS))
    _add_common_args(task_parser)

    html_parser = subparsers.add_parser("open-html", help="Open latest or selected HTML artifact in browser")
    html_parser.add_argument("path", nargs="?")
    html_parser.add_argument("--latest", action="store_true")

    history_parser = subparsers.add_parser("history", help="Load saved JSON history without rerunning")
    history_parser.add_argument("path", nargs="?")
    history_parser.add_argument("--latest", action="store_true")

    return parser


def _add_common_args(
    parser: argparse.ArgumentParser,
    *,
    include_variant: bool = True,
    include_preset: bool = True,
) -> None:
    parser.add_argument("--dataset", default="Student_Depression_Dataset.csv")
    if include_variant:
        parser.add_argument("--variant", choices=("A", "B"), default="A")
        parser.add_argument("--profile", choices=("safe", "full"), help=argparse.SUPPRESS)
    if include_preset:
        parser.add_argument("--preset", choices=("quick", "research"), default="quick")
    parser.add_argument("--export-html", action="store_true")
    parser.add_argument("--console-only", action="store_true")
    parser.add_argument("--output-dir", default="results/app")
    parser.add_argument("--budget", choices=("default", "auto"), default="default")


def _interactive_fallback(console: object | None) -> int:
    print_banner(console)
    print_status("Textual chưa được cài. Chuyển sang menu console đồng bộ với workflow hub.", console)

    dataset = prompt_text(
        "Dataset [mặc định: Student_Depression_Dataset.csv]: ",
        default="Student_Depression_Dataset.csv",
        console=console,
    )
    if not Path(dataset).exists():
        print_status(f"Không tìm thấy dataset: {dataset}", console)
        return 1

    specs = list_workflow_specs()
    print_status("", console)
    print_status("Chọn workflow:", console)
    for index, spec in enumerate(specs, start=1):
        print_status(f"  {index:>2}. {spec.label} [{spec.family}]", console)
    print_status("   0. Thoát", console)

    choice = prompt_text("Nhập lựa chọn [mặc định: 1]: ", default="1", console=console)
    if choice == "0":
        return 0
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(specs):
        print_status("Lựa chọn không hợp lệ.", console)
        return 1

    spec = specs[int(choice) - 1]
    variant = "A"
    if spec.supports_variant:
        variant = prompt_text("Variant [A/B, mặc định: A]: ", default="A", console=console).upper()
        if variant not in {"A", "B"}:
            variant = "A"

    preset = spec.default_preset
    if spec.family == "modern":
        preset = prompt_text("Preset [quick/research, mặc định: quick]: ", default="quick", console=console)
        if preset not in {"quick", "research"}:
            preset = "quick"

    budget = "default"
    if spec.supports_budget:
        budget = prompt_text("Training budget [default/auto, mặc định: default]: ", default="default", console=console)
        if budget not in {"default", "auto"}:
            budget = "default"

    export_html = False
    if spec.supports_export_html:
        export_html = prompt_text("Xuất HTML? [y/N]: ", default="n", console=console).lower().startswith("y")

    result = execute_workflow(
        WorkflowRequest(
            workflow_id=spec.workflow_id,
            dataset_path=dataset,
            variant=variant,
            preset=preset,
            export_html=export_html,
            console_only=True,
            training_budget_mode=budget,
        )
    )
    print_workflow_result(result, console)
    return 0


def main(argv: list[str] | None = None) -> int:
    console = get_console()
    parser = build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]

    if not raw_argv:
        tui_exit_code = run_tui([], show_banner=False, quiet_on_missing=True)
        if tui_exit_code == 0:
            return 0
        return _interactive_fallback(console)

    print_banner(console)
    args = parser.parse_args(raw_argv)

    if args.command == "open-html":
        target = args.path
        if args.latest or not target:
            target = latest_html_artifact()
        if target is None:
            print_status("Không tìm thấy file HTML để mở.", console)
            return 1
        opened = open_html_artifact(target)
        print_status(f"Đã mở HTML: {opened}", console)
        return 0

    if args.command == "history":
        target = args.path
        if args.latest or not target:
            target = latest_json_artifact()
        if target is None:
            print_status("KhÃ´ng tÃ¬m tháº¥y file JSON lá»‹ch sá»­ Ä‘á»ƒ xem.", console)
            return 1
        print_status(f"Äang xem láº¡i artifact: {target}", console)
        result = load_history_result(target)
        print_workflow_result(result, console)
        return 0

    request = _request_from_args(args)
    print_status(f"Đang chạy workflow: {request.workflow_id} ...", console)
    result = execute_workflow(request)
    print_workflow_result(result, console)
    return 0


def _request_from_args(args: argparse.Namespace) -> WorkflowRequest:
    if args.command == "profile":
        return WorkflowRequest(
            workflow_id="profile",
            dataset_path=args.dataset,
            export_html=args.export_html,
            output_dir=args.output_dir,
            console_only=args.console_only,
            training_budget_mode=args.budget,
        )
    if args.command == "compare":
        return WorkflowRequest(
            workflow_id="compare",
            dataset_path=args.dataset,
            preset=args.preset,
            output_dir=args.output_dir,
            console_only=args.console_only,
            training_budget_mode=args.budget,
        )
    if args.command == "task":
        workflow_id = args.workflow
    elif args.command == "run":
        workflow_id = "run"
    else:
        workflow_id = args.command

    return WorkflowRequest(
        workflow_id=workflow_id,
        dataset_path=getattr(args, "dataset", "Student_Depression_Dataset.csv"),
        variant=_resolve_variant_arg(args),
        preset=getattr(args, "preset", "quick"),
        export_html=getattr(args, "export_html", False),
        output_dir=getattr(args, "output_dir", "results/app"),
        console_only=getattr(args, "console_only", False),
        training_budget_mode=getattr(args, "budget", "default"),
    )


def _resolve_variant_arg(args: argparse.Namespace) -> str:
    profile = getattr(args, "profile", None)
    if profile == "full":
        return "B"
    if profile == "safe":
        return "A"
    return getattr(args, "variant", "A")


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
        message = 'Textual chưa được cài. Hãy dùng `robot task ...` hoặc cài nhanh bằng `uv pip install "textual>=0.86.0"`.'
        if console is None:
            print(message)
        else:
            console.print(
                'Textual chưa được cài. Hãy dùng [bold]robot task ...[/] hoặc cài nhanh bằng [bold]uv pip install "textual>=0.86.0"[/].'
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
