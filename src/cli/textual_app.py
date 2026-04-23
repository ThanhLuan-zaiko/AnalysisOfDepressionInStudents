from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from rich.align import Align
from rich import box
from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.app import (
    ArtifactPolicy,
    RunPreset,
    RunProfile,
    compare_profiles,
    load_dataset,
    profile_dataset,
    run_pipeline,
)


def launch_tui(default_dataset: Path) -> None:
    from textual import work
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import Button, Checkbox, Footer, Input, Select, Static

    class DepressionTUI(App[None]):
        TITLE = "Sen Analytics"
        SUB_TITLE = "Monitor Mode"
        BINDINGS = [
            ("1", "run_profile", "Profile"),
            ("2", "run_pipeline_hotkey", "Pipeline"),
            ("3", "run_compare", "Compare"),
            (":", "toggle_command_palette", "Command"),
            ("r", "rerun_current", "Rerun"),
            ("q", "quit", "Quit"),
        ]

        STATUS_THEME = {
            "ready": {"fg": "#93F5C6", "bg": "#0A141B", "border": "#2CC585"},
            "running": {"fg": "#FFD67A", "bg": "#1D1508", "border": "#E7A93F"},
            "success": {"fg": "#96FFB8", "bg": "#0D1712", "border": "#43D17A"},
            "error": {"fg": "#FF9FA4", "bg": "#1A0D10", "border": "#E05D5D"},
        }

        def compose(self) -> ComposeResult:
            with Vertical(id="root"):
                yield Static(id="hero")
                with Horizontal(id="layout"):
                    with VerticalScroll(id="sidebar_scroll"):
                        with Vertical(id="sidebar"):
                            yield Static(id="control_header")
                            yield Static("dataset", id="dataset_label")
                            yield Input(
                                value=str(default_dataset),
                                placeholder="Student_Depression_Dataset.csv",
                                id="dataset",
                            )
                            yield Static("profile", id="profile_label")
                            yield Select(
                                [(label.title(), label) for label in (RunProfile.SAFE.value, RunProfile.FULL.value)],
                                value=RunProfile.SAFE.value,
                                prompt="select profile",
                                id="profile",
                            )
                            yield Static("preset", id="preset_label")
                            yield Select(
                                [(label.title(), label) for label in (RunPreset.QUICK.value, RunPreset.RESEARCH.value)],
                                value=RunPreset.QUICK.value,
                                prompt="select preset",
                                id="preset",
                            )
                            yield Checkbox("export html eda", value=False, id="export_html")
                            yield Button("1  PROFILE", id="profile_btn")
                            yield Button("2  RUN PIPELINE", id="run_btn")
                            yield Button("3  COMPARE", id="compare_btn")
                            yield Button("R  RERUN CURRENT", id="rerun_btn")
                            yield Static(id="help_box")
                    with Vertical(id="workspace"):
                        yield Static(id="status_bar")
                        yield Input(placeholder=": command", id="cmdline")
                        with VerticalScroll(id="output_scroll"):
                            yield Static(id="output")
                yield Footer()

        def on_mount(self) -> None:
            self._tick = 0
            self._status_state = "ready"
            self._status_message = "awaiting command"
            self._last_action = "idle"
            self._last_action_mode = "run"
            self._boot_lines = [
                "[00] boot sequence start",
                "[01] probing terminal capabilities",
                "[02] loading sen analytics monitor profile",
                "[03] binding hotkeys 1 2 3 r : q",
                "[04] initializing holdout-first pipeline client",
                "[05] telemetry clock online",
                "[06] command panel online",
                "[07] output renderer online",
                "[08] system ready",
            ]
            self._boot_index = 1
            self._apply_theme()
            self._render_static_panels()
            self._set_status("running", "boot sequence starting")
            self._set_output(self._build_boot_log(self._boot_index))
            self._render_live_panels()
            self._boot_timer = self.set_interval(0.16, self._advance_boot)
            self.set_interval(0.55, self._refresh_dashboard)
            self.set_timer(1.75, self._finish_boot)

        def _apply_theme(self) -> None:
            self.styles.background = "#061018"

            root = self.query_one("#root", Vertical)
            root.styles.width = "100%"
            root.styles.height = "100%"
            root.styles.padding = (1, 1)
            root.styles.background = "#061018"

            hero = self.query_one("#hero", Static)
            hero.styles.height = 11
            hero.styles.margin = (0, 0, 1, 0)

            layout = self.query_one("#layout", Horizontal)
            layout.styles.height = "1fr"

            sidebar_scroll = self.query_one("#sidebar_scroll", VerticalScroll)
            sidebar_scroll.styles.width = 42
            sidebar_scroll.styles.min_width = 38
            sidebar_scroll.styles.max_width = 46
            sidebar_scroll.styles.height = "1fr"
            sidebar_scroll.styles.margin = (0, 1, 0, 0)
            sidebar_scroll.styles.border = ("round", "#14303F")
            sidebar_scroll.styles.background = "#07131B"
            sidebar_scroll.styles.padding = (0, 1, 1, 0)

            sidebar = self.query_one("#sidebar", Vertical)
            sidebar.styles.width = "100%"
            sidebar.styles.height = "auto"

            workspace = self.query_one("#workspace", Vertical)
            workspace.styles.width = "1fr"

            for label_id in ("#dataset_label", "#profile_label", "#preset_label"):
                label = self.query_one(label_id, Static)
                label.styles.color = "#67D5FF"
                label.styles.text_style = "bold"
                label.styles.margin = (1, 0, 0, 0)

            dataset = self.query_one("#dataset", Input)
            dataset.styles.border = ("round", "#2B5C75")
            dataset.styles.background = "#0A1720"
            dataset.styles.color = "#E9FFF2"
            dataset.styles.margin = (0, 0, 1, 0)

            for select_id in ("#profile", "#preset"):
                select = self.query_one(select_id, Select)
                select.styles.border = ("round", "#2B5C75")
                select.styles.background = "#0A1720"
                select.styles.color = "#E9FFF2"
                select.styles.margin = (0, 0, 1, 0)

            checkbox = self.query_one("#export_html", Checkbox)
            checkbox.styles.color = "#9DEFC8"
            checkbox.styles.margin = (1, 0, 1, 0)

            for button_id, border, bg, fg in (
                ("#profile_btn", "#2E8DB5", "#102B3E", "#DDF8FF"),
                ("#run_btn", "#43D17A", "#103021", "#E3FFE9"),
                ("#compare_btn", "#E0A95A", "#37240F", "#FFE0AF"),
                ("#rerun_btn", "#8A6CFF", "#1B1437", "#E2DCFF"),
            ):
                button = self.query_one(button_id, Button)
                button.styles.width = "100%"
                button.styles.height = 3
                button.styles.margin = (0, 0, 1, 0)
                button.styles.border = ("round", border)
                button.styles.background = bg
                button.styles.color = fg
                button.styles.text_style = "bold"

            control_header = self.query_one("#control_header", Static)
            control_header.styles.height = 5
            control_header.styles.margin = (0, 0, 1, 0)

            help_box = self.query_one("#help_box", Static)
            help_box.styles.height = "auto"
            help_box.styles.margin = (1, 0, 0, 0)

            status_bar = self.query_one("#status_bar", Static)
            status_bar.styles.height = 3
            status_bar.styles.margin = (0, 0, 1, 0)

            cmdline = self.query_one("#cmdline", Input)
            cmdline.styles.height = 3
            cmdline.styles.margin = (0, 0, 1, 0)
            cmdline.styles.border = ("round", "#7A5DFF")
            cmdline.styles.background = "#120F24"
            cmdline.styles.color = "#ECE7FF"
            cmdline.display = False

            output_scroll = self.query_one("#output_scroll", VerticalScroll)
            output_scroll.styles.height = "1fr"
            output_scroll.styles.border = ("round", "#1E6B8F")
            output_scroll.styles.background = "#050D13"
            output_scroll.styles.padding = (0, 1)

            output = self.query_one("#output", Static)
            output.styles.width = "100%"
            output.styles.color = "#D8FEE3"

        def _render_static_panels(self) -> None:
            self.query_one("#control_header", Static).update(self._build_control_header())
            self.query_one("#help_box", Static).update(self._build_help_panel())

        def _refresh_dashboard(self) -> None:
            self._tick += 1
            self._render_live_panels()

        def _advance_boot(self) -> None:
            if self._last_action != "idle":
                self._boot_timer.stop()
                return
            if self._boot_index >= len(self._boot_lines):
                self._boot_timer.stop()
                self._set_status("ready", "terminal monitor online")
                return
            self._boot_index += 1
            self._set_output(self._build_boot_log(self._boot_index))

        def _render_live_panels(self) -> None:
            self.query_one("#hero", Static).update(self._build_hero())
            self.query_one("#status_bar", Static).update(self._build_status_panel())

        def _finish_boot(self) -> None:
            if self._last_action == "idle":
                self._set_output(self._build_idle_output())

        def _dataset(self) -> str:
            return self.query_one("#dataset", Input).value.strip()

        def _profile(self) -> RunProfile:
            return RunProfile(self.query_one("#profile", Select).value)

        def _preset(self) -> RunPreset:
            return RunPreset(self.query_one("#preset", Select).value)

        def _export_html(self) -> bool:
            return self.query_one("#export_html", Checkbox).value

        def _short_dataset(self) -> str:
            dataset_path = self._dataset()
            if not dataset_path:
                return "none"
            return Path(dataset_path).name

        def _format_value(self, value: Any) -> str:
            if value is None:
                return "n/a"
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)

        def _metric_pair(self, oof_value: Any, holdout_value: Any) -> str:
            return f"oof {self._format_value(oof_value):>8} | holdout {self._format_value(holdout_value):>8}"

        def _sparkline(self, values: list[float | None]) -> str:
            blocks = "._-:=+*#@"
            filtered = [float(value) for value in values if value is not None]
            if not filtered:
                return "n/a"
            lo = min(filtered)
            hi = max(filtered)
            if hi - lo < 1e-12:
                return blocks[-2] * len(filtered)
            chars: list[str] = []
            for value in filtered:
                idx = round((value - lo) / (hi - lo) * (len(blocks) - 1))
                chars.append(blocks[idx])
            return "".join(chars)

        def _noise_line(self, width: int = 92, offset: int = 0) -> str:
            glyphs = " .:-=+*#"
            seed = (self._tick * 7) + offset + len(self._short_dataset())
            chars: list[str] = []
            for index in range(width):
                code = (seed + index * 5 + (index // 3) * 11) % 17
                if code > 13:
                    chars.append(" ")
                else:
                    chars.append(glyphs[code % len(glyphs)])
            return "".join(chars)

        def _wrap_with_scanlines(self, *renderables: Any) -> Group:
            top = Text(self._noise_line(94, 1), style="#1A3949")
            bottom = Text(self._noise_line(94, 9), style="#1A3949")
            return Group(top, *renderables, bottom)

        def _set_status(self, state: str, message: str) -> None:
            self._status_state = state
            self._status_message = message
            self._render_live_panels()

        def _set_output(self, renderable: Any) -> None:
            self.query_one("#output", Static).update(renderable)

        def _build_hero(self) -> Panel:
            pulse = "|/-\\"[self._tick % 4] if self._status_state == "running" else ">"
            clock = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            banner = Text(
                "\n".join(
                    [
                        "  ________  ___  __       ___    ________      ___    ________   ________  ___    ___ _________  ___  ________  ________     ",
                        " |\\   ____\\|\\  \\|\\  \\    |\\  \\  |\\   ___  \\   |\\  \\  |\\   ___  \\|\\   __  \\|\\  \\  /  /|\\___   ___\\\\  \\|\\   ____\\|\\   ____\\    ",
                        " \\ \\  \\___|\\ \\  \\/  /|_  \\ \\  \\ \\ \\  \\\\ \\  \\  \\ \\  \\ \\ \\  \\\\ \\  \\ \\  \\|\\  \\ \\  \\/  / ||___ \\  \\_\\ \\  \\ \\  \\___|\\ \\  \\___|    ",
                        "  \\ \\_____  \\ \\   ___  \\  \\ \\  \\ \\ \\  \\\\ \\  \\  \\ \\  \\ \\ \\  \\\\ \\  \\ \\   __  \\ \\    / /     \\ \\  \\ \\ \\  \\ \\  \\    \\ \\_____  \\   ",
                        "   \\|____|\\  \\ \\  \\\\ \\  \\  \\ \\  \\ \\ \\  \\\\ \\  \\  \\ \\  \\ \\ \\  \\\\ \\  \\ \\  \\ \\  \\/  /  /       \\ \\  \\ \\ \\  \\ \\  \\____\\|____|\\  \\  ",
                        "     ____\\_\\  \\ \\__\\\\ \\__\\  \\ \\__\\ \\ \\__\\\\ \\__\\  \\ \\__\\ \\ \\__\\\\ \\__\\ \\__\\ \\__/  / /          \\ \\__\\ \\ \\__\\ \\_______\\____\\_\\  \\ ",
                        "    |\\_________\\|__| \\|__|   \\|__|  \\|__| \\|__|   \\|__|  \\|__| \\|__|\\|__|\\|__|/ /            \\|__|  \\|__|\\|_______|\\_________\\",
                        "    \\|_________|                                                                     \\/                                     \\|_________|",
                    ]
                ),
                style="bold #7AF5C2",
            )

            telemetry = Text()
            telemetry.append(f" {pulse} ", style="bold #0A141B on #93F5C6")
            telemetry.append("clock ", style="bold #67D5FF")
            telemetry.append(clock, style="bold #D8FEE3")
            telemetry.append("  ||  ", style="bold #2E8DB5")
            telemetry.append("dataset ", style="bold #67D5FF")
            telemetry.append(self._short_dataset(), style="bold #D8FEE3")
            telemetry.append("  ||  ", style="bold #2E8DB5")
            telemetry.append("profile ", style="bold #67D5FF")
            telemetry.append(self._profile().value, style="bold #93F5C6")
            telemetry.append("  ||  ", style="bold #2E8DB5")
            telemetry.append("preset ", style="bold #67D5FF")
            telemetry.append(self._preset().value, style="bold #FFD67A")
            telemetry.append("  ||  ", style="bold #2E8DB5")
            telemetry.append("last ", style="bold #67D5FF")
            telemetry.append(self._last_action, style="bold #D8FEE3")

            hotkeys = Text(
                " hotkeys :: [1] profile  [2] pipeline  [3] compare  [r] rerun current  [q] quit ",
                style="bold #9EC6D6",
            )

            return Panel(
                Group(
                    Text(self._noise_line(118, 3), style="#1A3949"),
                    Align.left(banner),
                    Text(""),
                    Align.left(telemetry),
                    Align.left(hotkeys),
                    Text(self._noise_line(118, 6), style="#1A3949"),
                ),
                title="[bold #FFD67A]TERMINAL MONITOR[/]",
                border_style="#1E6B8F",
                box=box.DOUBLE,
            )

        def _build_status_panel(self) -> Panel:
            theme = self.STATUS_THEME[self._status_state]
            pulse = "|/-\\"[self._tick % 4] if self._status_state == "running" else "+"
            line = Text()
            line.append(f" {pulse} ", style=f"bold {theme['bg']} on {theme['fg']}")
            line.append(f" {self._status_state.upper():<7} ", style=f"bold {theme['fg']}")
            line.append("|| ", style=f"bold {theme['border']}")
            line.append(self._status_message, style=f"bold {theme['fg']}")
            line.append(" || ", style=f"bold {theme['border']}")
            line.append(f"last={self._last_action}", style="bold #9EC6D6")
            return Panel(line, title="[bold]STATUS BAR[/]", border_style=theme["border"], box=box.HEAVY)

        def _build_control_header(self) -> Panel:
            text = Text()
            text.append("COMMAND PANEL\n", style="bold #74F0B8")
            text.append("configure dataset / profile / preset\n", style="#D8FEE3")
            text.append("then fire with buttons or hotkeys", style="#87B7C8")
            return Panel(text, border_style="#23536C", box=box.HEAVY)

        def _build_help_panel(self) -> Panel:
            text = Text()
            text.append("OPERATOR NOTES\n\n", style="bold #74F0B8")
            text.append("1  PROFILE      ", style="bold #DDF8FF")
            text.append("dataset shape + warnings\n", style="#87B7C8")
            text.append("2  RUN PIPELINE ", style="bold #E3FFE9")
            text.append("metrics for current profile/preset\n", style="#87B7C8")
            text.append("3  COMPARE      ", style="bold #FFE0AF")
            text.append("safe vs full delta on same split\n", style="#87B7C8")
            text.append("R  RERUN        ", style="bold #E2DCFF")
            text.append("run current controls again\n", style="#87B7C8")
            text.append(":  COMMAND      ", style="bold #E6DDFF")
            text.append("open palette, type :help for commands\n\n", style="#87B7C8")
            text.append("safe     ", style="bold #93F5C6")
            text.append("reduced leakage risk\n", style="#87B7C8")
            text.append("research ", style="bold #FFD67A")
            text.append("adds GAM + rust metadata\n", style="#87B7C8")
            return Panel(text, title="[bold]NOTES[/]", border_style="#1C394A", box=box.HEAVY)

        def _build_idle_output(self) -> Group:
            text = Text()
            text.append("SYSTEM READY\n\n", style="bold #93F5C6")
            text.append("no report rendered yet\n\n", style="#D8FEE3")
            text.append("> press 1 to inspect dataset profile\n", style="#67D5FF")
            text.append("> press 2 to run pipeline with current controls\n", style="#67D5FF")
            text.append("> press 3 to compare safe vs full\n", style="#67D5FF")
            text.append("> press r to rerun pipeline with current profile/preset\n", style="#B9A7FF")
            text.append("> press : to open command palette\n", style="#E0B8FF")
            text.append("> type :help to list supported commands\n", style="#E0B8FF")
            return self._wrap_with_scanlines(
                Panel(text, title="[bold]OUTPUT[/]", border_style="#1E6B8F", box=box.DOUBLE)
            )

        def _build_boot_log(self, visible_count: int | None = None) -> Group:
            visible = visible_count if visible_count is not None else len(self._boot_lines)
            lines = self._boot_lines[:visible]
            if visible >= len(self._boot_lines):
                lines.append("[09] operator shell attached")
            else:
                lines.append(f"[..] loading module {visible:02d}/{len(self._boot_lines):02d}")
            text = Text("\n".join(lines), style="#9DEFC8")
            return self._wrap_with_scanlines(
                Panel(text, title="[bold]BOOT LOG[/]", border_style="#43D17A", box=box.DOUBLE)
            )

        def _build_placeholder(self, title: str, message: str) -> Group:
            return self._wrap_with_scanlines(
                Panel(
                    Text(message, style="bold #FFD67A"),
                    title=f"[bold]{title}[/]",
                    border_style="#E7A93F",
                    box=box.HEAVY,
                )
            )

        def _kv_table(self, title: str, rows: list[tuple[str, str]], border: str) -> Panel:
            table = Table(box=box.HEAVY, border_style=border, show_header=False, expand=True)
            table.add_column("key", style="bold #67D5FF", width=24)
            table.add_column("value", style="#D8FEE3")
            for key, value in rows:
                table.add_row(key, value)
            return Panel(table, title=f"[bold]{title}[/]", border_style=border, box=box.DOUBLE)

        def _build_profile_output(self, report: Any) -> Group:
            renderables: list[Any] = [
                self._kv_table(
                    "DATASET PROFILE",
                    [
                        ("rows", f"{report.summary['rows']:,}"),
                        ("cols", str(report.summary["cols"])),
                        ("positive_rate", f"{report.summary['target_positive_rate']}%"),
                        ("cache", "used" if report.summary["loaded_from_cache"] else "fresh load"),
                    ],
                    "#1E6B8F",
                ),
                self._kv_table(
                    "SELECTED COLUMNS",
                    [
                        ("safe", ", ".join(report.summary["selected_columns_safe"])),
                        ("full", ", ".join(report.summary["selected_columns_full"])),
                    ],
                    "#23536C",
                ),
            ]
            if report.warnings:
                warning_text = Text("\n".join(f"- {warning}" for warning in report.warnings), style="#FFD2A6")
                renderables.append(
                    Panel(warning_text, title="[bold]WARNINGS[/]", border_style="#E0A95A", box=box.HEAVY)
                )
            return self._wrap_with_scanlines(*renderables)

        def _model_metric_panel(self, model_name: str, result: Any) -> Panel:
            rows = [
                (
                    "roc_auc",
                    f"{self._metric_pair(result.oof.get('roc_auc'), result.holdout.get('roc_auc'))} | {self._sparkline([result.oof.get('roc_auc'), result.holdout.get('roc_auc')])}",
                ),
                (
                    "pr_auc",
                    f"{self._metric_pair(result.oof.get('pr_auc'), result.holdout.get('pr_auc'))} | {self._sparkline([result.oof.get('pr_auc'), result.holdout.get('pr_auc')])}",
                ),
                (
                    "f1",
                    f"{self._metric_pair(result.oof.get('f1'), result.holdout.get('f1'))} | {self._sparkline([result.oof.get('f1'), result.holdout.get('f1')])}",
                ),
                (
                    "recall",
                    f"{self._metric_pair(result.oof.get('recall'), result.holdout.get('recall'))} | {self._sparkline([result.oof.get('recall'), result.holdout.get('recall')])}",
                ),
                (
                    "precision",
                    f"{self._metric_pair(result.oof.get('precision'), result.holdout.get('precision'))} | {self._sparkline([result.oof.get('precision'), result.holdout.get('precision')])}",
                ),
                (
                    "brier_score",
                    f"{self._metric_pair(result.oof.get('brier_score'), result.holdout.get('brier_score'))} | {self._sparkline([result.oof.get('brier_score'), result.holdout.get('brier_score')])}",
                ),
            ]
            if result.metadata.get("engine"):
                rows.append(("engine", str(result.metadata["engine"])))
            table = Table(box=box.HEAVY, border_style="#1E6B8F", show_header=False, expand=True)
            table.add_column("metric", style="bold #67D5FF", width=18)
            table.add_column("value", style="#D8FEE3")
            for key, value in rows:
                table.add_row(key, value)

            renderables: list[Any] = [table]
            if result.feature_importance:
                top = Table(box=box.HEAVY, border_style="#43D17A", show_header=False, expand=True)
                top.add_column("feature", style="bold #93F5C6", width=28)
                top.add_column("signal", style="#D8FEE3")
                for row in result.feature_importance[:6]:
                    signal = row.get(
                        "importance",
                        row.get(
                            "abs_coefficient",
                            row.get("variance_importance", row.get("coefficient")),
                        ),
                    )
                    top.add_row(str(row["feature"]), self._format_value(signal))
                renderables.append(top)

            return Panel(Group(*renderables), title=f"[bold]{model_name.upper()}[/]", border_style="#1E6B8F", box=box.DOUBLE)

        def _build_run_output(self, report: Any) -> Group:
            renderables: list[Any] = [
                self._kv_table(
                    "RUN CONFIG",
                    [
                        ("profile", report.config["profile"]),
                        ("preset", report.config["preset"]),
                        ("models", ", ".join(report.config["models"])),
                        ("selected_columns", ", ".join(report.config["selected_columns"])),
                    ],
                    "#23536C",
                )
            ]

            for model_name, result in report.models.items():
                renderables.append(self._model_metric_panel(model_name, result))

            renderables.append(
                self._kv_table(
                    "TIMINGS",
                    [
                        (
                            key,
                            f"{value}s | {self._sparkline([float(v) for v in report.timings.values()])}",
                        )
                        for key, value in report.timings.items()
                    ],
                    "#23536C",
                )
            )

            if report.warnings:
                warning_text = Text("\n".join(f"- {warning}" for warning in report.warnings), style="#FFD2A6")
                renderables.append(
                    Panel(warning_text, title="[bold]WARNINGS[/]", border_style="#E0A95A", box=box.HEAVY)
                )

            return self._wrap_with_scanlines(*renderables)

        def _build_compare_output(self, report: Any) -> Group:
            safe_panel = self._profile_compare_panel("SAFE", report.profiles["safe"], "#43D17A")
            full_panel = self._profile_compare_panel("FULL", report.profiles["full"], "#E0A95A")

            summary = Table(box=box.HEAVY, border_style="#1E6B8F", expand=True)
            summary.add_column("model", style="bold #74F0B8")
            summary.add_column("roc_auc delta", style="#FFD67A")
            summary.add_column("f1 delta", style="#FFD67A")
            for model_name, model_summary in report.summary.items():
                summary.add_row(
                    model_name,
                    self._format_value(model_summary["roc_auc_delta_full_minus_safe"]),
                    self._format_value(model_summary["f1_delta_full_minus_safe"]),
                )

            return self._wrap_with_scanlines(
                self._kv_table("COMPARE CONFIG", [("preset", report.preset)], "#23536C"),
                Columns([safe_panel, full_panel], equal=True, expand=True),
                Panel(summary, title="[bold]DELTA SUMMARY[/]", border_style="#1E6B8F", box=box.DOUBLE),
            )

        def _profile_compare_panel(self, label: str, run_report: Any, border: str) -> Panel:
            table = Table(box=box.HEAVY, border_style=border, show_header=False, expand=True)
            table.add_column("metric", style="bold #67D5FF", width=20)
            table.add_column("value", style="#D8FEE3")
            table.add_row("profile", run_report.config["profile"])
            table.add_row("preset", run_report.config["preset"])
            for model_name, result in run_report.models.items():
                table.add_row(
                    f"{model_name}.roc_auc",
                    f"{self._format_value(result.holdout.get('roc_auc'))} | {self._sparkline([result.oof.get('roc_auc'), result.holdout.get('roc_auc')])}",
                )
                table.add_row(
                    f"{model_name}.f1",
                    f"{self._format_value(result.holdout.get('f1'))} | {self._sparkline([result.oof.get('f1'), result.holdout.get('f1')])}",
                )
                table.add_row(
                    f"{model_name}.recall",
                    f"{self._format_value(result.holdout.get('recall'))} | {self._sparkline([result.oof.get('recall'), result.holdout.get('recall')])}",
                )
            table.add_row("timing.scan", self._sparkline([float(value) for value in run_report.timings.values()]))
            return Panel(table, title=f"[bold]{label} PANEL[/]", border_style=border, box=box.DOUBLE)

        def _build_command_reference(self) -> Group:
            table = Table(box=box.HEAVY, border_style="#7A5DFF", show_header=False, expand=True)
            table.add_column("command", style="bold #E6DDFF", width=26)
            table.add_column("effect", style="#D8FEE3")
            rows = [
                (":profile", "profile current dataset"),
                (":run", "run current profile/preset"),
                (":compare", "compare safe vs full"),
                (":rerun", "repeat last action"),
                (":set profile safe", "switch to safe profile"),
                (":set profile full", "switch to full profile"),
                (":set preset quick", "switch to quick preset"),
                (":set preset research", "switch to research preset"),
                (":set dataset <path>", "update dataset path"),
                (":help", "show command palette reference"),
            ]
            for command, effect in rows:
                table.add_row(command, effect)
            return self._wrap_with_scanlines(
                Panel(table, title="[bold]COMMAND PALETTE[/]", border_style="#7A5DFF", box=box.DOUBLE)
            )

        def _build_error_output(self, exc: Exception) -> Group:
            return self._wrap_with_scanlines(
                Panel(
                    Text(str(exc), style="bold #FF9FA4"),
                    title="[bold]ERROR[/]",
                    border_style="#E05D5D",
                    box=box.HEAVY,
                )
            )

        def _validate_dataset(self) -> str | None:
            dataset_path = self._dataset()
            if not dataset_path:
                self._set_status("error", "dataset path is empty")
                self._set_output(self._build_error_output(ValueError("Dataset path is empty.")))
                return None
            return dataset_path

        def _start_profile(self) -> None:
            dataset_path = self._validate_dataset()
            if dataset_path is None:
                return
            self._last_action = "profile"
            self._last_action_mode = "profile"
            self._set_status("running", f"profiling {Path(dataset_path).name}")
            self._set_output(self._build_placeholder("PROFILE", "collecting shape, columns, and leakage warnings ..."))
            self._profile_worker(dataset_path, self._export_html())

        def _start_run(self) -> None:
            dataset_path = self._validate_dataset()
            if dataset_path is None:
                return
            self._last_action = f"run:{self._profile().value}/{self._preset().value}"
            self._last_action_mode = "run"
            self._set_status("running", f"pipeline profile={self._profile().value} preset={self._preset().value}")
            self._set_output(self._build_placeholder("PIPELINE", "training models and building monitor report ..."))
            self._run_worker(dataset_path, self._profile(), self._preset())

        def _start_compare(self) -> None:
            dataset_path = self._validate_dataset()
            if dataset_path is None:
                return
            self._last_action = f"compare:{self._preset().value}"
            self._last_action_mode = "compare"
            self._set_status("running", "comparing safe vs full")
            self._set_output(self._build_placeholder("COMPARE", "running same split across both profiles ..."))
            self._compare_worker(dataset_path, self._preset())

        def action_run_profile(self) -> None:
            self._start_profile()

        def action_run_pipeline_hotkey(self) -> None:
            self._start_run()

        def action_run_compare(self) -> None:
            self._start_compare()

        def action_toggle_command_palette(self) -> None:
            cmdline = self.query_one("#cmdline", Input)
            cmdline.display = not cmdline.display
            if cmdline.display:
                cmdline.value = ":"
                cmdline.focus()
                self._set_status("ready", "command palette open")
            else:
                self.set_focus(None)
                self._set_status("ready", "command palette closed")

        def action_rerun_current(self) -> None:
            if self._last_action_mode == "profile":
                self._start_profile()
            elif self._last_action_mode == "compare":
                self._start_compare()
            else:
                self._start_run()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "profile_btn":
                self._start_profile()
            elif event.button.id == "run_btn":
                self._start_run()
            elif event.button.id == "compare_btn":
                self._start_compare()
            elif event.button.id == "rerun_btn":
                self.action_rerun_current()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            if event.input.id != "cmdline":
                return
            command = event.value.strip()
            event.input.display = False
            if not command or command == ":":
                self._set_status("ready", "command palette closed")
                return
            self._run_command(command)

        def _run_command(self, command: str) -> None:
            normalized = command.lower().strip()
            if normalized.startswith(":"):
                normalized = normalized[1:].strip()

            if normalized in {"profile", "p"}:
                self._start_profile()
                return
            if normalized in {"run", "pipeline", "2"}:
                self._start_run()
                return
            if normalized in {"compare", "cmp", "3"}:
                self._start_compare()
                return
            if normalized in {"rerun", "repeat", "r"}:
                self.action_rerun_current()
                return
            if normalized in {"help", "h", "?"}:
                self._set_output(self._build_command_reference())
                self._set_status("success", "command reference rendered")
                return
            if normalized.startswith("set profile "):
                value = normalized.removeprefix("set profile ").strip()
                if value in {profile.value for profile in RunProfile}:
                    self.query_one("#profile", Select).value = value
                    self._set_status("success", f"profile set to {value}")
                    return
            if normalized.startswith("set preset "):
                value = normalized.removeprefix("set preset ").strip()
                if value in {preset.value for preset in RunPreset}:
                    self.query_one("#preset", Select).value = value
                    self._set_status("success", f"preset set to {value}")
                    return
            if normalized.startswith("set dataset "):
                value = command.split(" ", 2)[2].strip() if len(command.split(" ", 2)) == 3 else ""
                if value:
                    self.query_one("#dataset", Input).value = value
                    self._set_status("success", f"dataset set to {Path(value).name}")
                    return

            self._set_status("error", f"unknown command: {command}")
            self._set_output(
                self._build_error_output(
                    ValueError(
                        "Unknown command. Try :profile, :run, :compare, :rerun, :set profile safe, :set preset research"
                    )
                )
            )

        @work(thread=True, exclusive=True)
        def _profile_worker(self, dataset_path: str, export_html: bool) -> None:
            try:
                bundle = load_dataset(dataset_path)
                report = profile_dataset(
                    bundle=bundle,
                    artifact_policy=ArtifactPolicy.FULL_EXPORT if export_html else ArtifactPolicy.CONSOLE_ONLY,
                    export_html=export_html,
                )
                self.call_from_thread(self._profile_done, report)
            except Exception as exc:  # pragma: no cover - interactive path
                self.call_from_thread(self._task_failed, exc)

        @work(thread=True, exclusive=True)
        def _run_worker(self, dataset_path: str, profile: RunProfile, preset: RunPreset) -> None:
            try:
                bundle = load_dataset(dataset_path)
                report = run_pipeline(
                    bundle=bundle,
                    profile=profile,
                    preset=preset,
                    artifact_policy=ArtifactPolicy.CONSOLE_ONLY,
                )
                self.call_from_thread(self._run_done, report)
            except Exception as exc:  # pragma: no cover - interactive path
                self.call_from_thread(self._task_failed, exc)

        @work(thread=True, exclusive=True)
        def _compare_worker(self, dataset_path: str, preset: RunPreset) -> None:
            try:
                bundle = load_dataset(dataset_path)
                report = compare_profiles(
                    bundle=bundle,
                    preset=preset,
                    artifact_policy=ArtifactPolicy.CONSOLE_ONLY,
                )
                self.call_from_thread(self._compare_done, report)
            except Exception as exc:  # pragma: no cover - interactive path
                self.call_from_thread(self._task_failed, exc)

        def _profile_done(self, report: Any) -> None:
            self._set_output(self._build_profile_output(report))
            self._set_status("success", "dataset profile completed")

        def _run_done(self, report: Any) -> None:
            self._set_output(self._build_run_output(report))
            self._set_status("success", "pipeline completed")

        def _compare_done(self, report: Any) -> None:
            self._set_output(self._build_compare_output(report))
            self._set_status("success", "comparison completed")

        def _task_failed(self, exc: Exception) -> None:
            self._set_output(self._build_error_output(exc))
            self._set_status("error", "task failed")

    DepressionTUI().run()
