from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.align import Align
from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .workflows import (
    analyze_console_log,
    WORKFLOW_SPECS,
    WorkflowRequest,
    WorkflowResult,
    describe_json_artifact,
    describe_log_artifact,
    execute_workflow,
    latest_json_artifact,
    latest_log_artifact,
    latest_html_artifact,
    list_workflow_specs,
    load_console_log_result,
    load_history_result,
    open_html_artifact,
    scan_html_artifacts,
    scan_json_artifacts,
    scan_log_artifacts,
)


def launch_tui(default_dataset: Path) -> None:
    from textual import events, work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import Button, Checkbox, Footer, Input, Select, Static

    class DepressionTUI(App[None]):
        TITLE = "Sen Analytics"
        SUB_TITLE = "Monitor Mode"
        BINDINGS = [
            Binding("1", "run_current", "Run", priority=True),
            Binding("2", "open_latest", "Latest", priority=True),
            Binding("3", "open_selected_html", "Open HTML", priority=True),
            Binding("4", "load_history", "History", priority=True),
            Binding("5", "toggle_json_dump", "JSON", priority=True),
            Binding("6", "load_console_log", "Log", priority=True),
            Binding("f5", "refresh_html", "Refresh", priority=True),
            Binding(":", "toggle_command_palette", "Command", priority=True),
            Binding("r", "rerun_current", "Rerun", priority=True),
            Binding("q", "quit", "Quit", priority=True),
        ]

        STATUS_THEME = {
            "ready": {"fg": "#93F5C6", "bg": "#0A141B", "border": "#2CC585"},
            "running": {"fg": "#FFD67A", "bg": "#1D1508", "border": "#E7A93F"},
            "success": {"fg": "#96FFB8", "bg": "#0D1712", "border": "#43D17A"},
            "error": {"fg": "#FF9FA4", "bg": "#1A0D10", "border": "#E05D5D"},
        }
        DANGER_STATUS_THEME = {
            "ready": {"fg": "#FFC0B8", "bg": "#17090B", "border": "#C24652"},
            "running": {"fg": "#FFD3A8", "bg": "#261208", "border": "#FF8A5B"},
            "success": {"fg": "#FFD3A8", "bg": "#241108", "border": "#D8692E"},
            "error": {"fg": "#FFE2DE", "bg": "#23080B", "border": "#FF5252"},
        }
        PALETTES = {
            "default": {
                "screen_bg": "#061018",
                "sidebar_bg": "#050D13",
                "sidebar_border": "#14303F",
                "label_fg": "#67D5FF",
                "label_border": "#14303F",
                "label_bg": "#07131B",
                "field_border": "#194A63",
                "field_bg": "#081119",
                "field_fg": "#DDF8FF",
                "output_border": "#1E6B8F",
                "output_bg": "#030A10",
                "footer_bg": "#101820",
                "footer_fg": "#FFD67A",
                "footer_border": "#2E8DB5",
                "cmd_border": "#7A5DFF",
                "cmd_bg": "#120F24",
                "cmd_fg": "#ECE7FF",
                "hero_border": "#1E6B8F",
                "hero_title": "#FFD67A",
                "banner": "#7AF5C2",
                "noise": "#1A3949",
                "grid": "#14303F",
                "meta_label": "#67D5FF",
                "meta_value": "#D8FEE3",
                "separator": "#2E8DB5",
                "hotkeys": "#9EC6D6",
                "accent": "#43D17A",
                "accent_soft": "#93F5C6",
                "accent_bg": "#103021",
                "signal_border": "#1E6B8F",
                "delta_border": "#8A6CFF",
                "artifact_border": "#23536C",
                "skull": "#FF9FA4",
            },
            "danger": {
                "screen_bg": "#12070A",
                "sidebar_bg": "#0D0507",
                "sidebar_border": "#4A1419",
                "label_fg": "#FF8E78",
                "label_border": "#4A1419",
                "label_bg": "#18090C",
                "field_border": "#74232B",
                "field_bg": "#12080B",
                "field_fg": "#FFE7DD",
                "output_border": "#84252D",
                "output_bg": "#090204",
                "footer_bg": "#1A0B0F",
                "footer_fg": "#FFB16E",
                "footer_border": "#A2353A",
                "cmd_border": "#C24652",
                "cmd_bg": "#190B10",
                "cmd_fg": "#FFF0EA",
                "hero_border": "#A2353A",
                "hero_title": "#FFB16E",
                "banner": "#FF8E78",
                "noise": "#42151A",
                "grid": "#3A1116",
                "meta_label": "#FF8E78",
                "meta_value": "#FFE7DD",
                "separator": "#B3412F",
                "hotkeys": "#FFC89C",
                "accent": "#FF5252",
                "accent_soft": "#FFB3A6",
                "accent_bg": "#341014",
                "signal_border": "#C24652",
                "delta_border": "#FF8A5B",
                "artifact_border": "#B3412F",
                "skull": "#FFD0C7",
            },
        }

        def compose(self) -> ComposeResult:
            workflow_options = [(spec.label, spec.workflow_id) for spec in list_workflow_specs()]
            html_options = [("refresh list first", "")]
            history_options = [("refresh list first", "")]
            log_options = [("refresh list first", "")]

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
                            yield Static("workflow", id="workflow_label")
                            yield Select(workflow_options, value="profile", prompt="select workflow", id="workflow")
                            yield Static("variant", id="variant_label")
                            yield Select([("A / safe", "A"), ("B / full", "B")], value="A", prompt="select variant", id="variant")
                            yield Static("preset", id="preset_label")
                            yield Select([("Quick", "quick"), ("Research", "research")], value="quick", prompt="select preset", id="preset")
                            yield Static("budget", id="budget_label")
                            yield Select([("Default", "default"), ("Auto", "auto")], value="default", prompt="select budget", id="budget")
                            yield Checkbox("export html", value=False, id="export_html")
                            yield Checkbox("auto-open latest html", value=False, id="auto_open_html")
                            yield Static("html picker", id="html_label")
                            yield Select(html_options, value="", prompt="select html", id="html_pick")
                            yield Static("history json", id="history_label")
                            yield Select(history_options, value="", prompt="select history", id="history_pick")
                            yield Static("console log", id="log_label")
                            yield Select(log_options, value="", prompt="select log", id="log_pick")
                            yield Button("1  RUN WORKFLOW", id="run_btn")
                            yield Button("2  OPEN LATEST", id="open_latest_btn")
                            yield Button("3  OPEN SELECTED", id="open_selected_btn")
                            yield Button("4  LOAD HISTORY", id="load_history_btn")
                            yield Button("6  LOAD CONSOLE LOG", id="load_log_btn")
                            yield Button("R  RERUN CURRENT", id="rerun_btn")
                            yield Button("REFRESH ARTIFACT LISTS", id="refresh_btn")
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
            self._status_message = "boot sequence starting"
            self._last_request: WorkflowRequest | None = None
            self._last_result: WorkflowResult | None = None
            self._last_html: list[str] = []
            self._last_json: list[str] = []
            self._last_logs: list[str] = []
            self._last_action = "idle"
            self._show_json_dump = False
            self._boot_lines = [
                "[00] cold start",
                "[01] probing terminal capabilities",
                "[02] loading sen analytics monitor shell",
                "[03] binding workflow controls",
                "[04] indexing html artifact channels",
                "[05] warming telemetry renderers",
                "[06] routing command palette",
                "[07] operator handshake established",
                "[08] monitor grid online",
            ]
            self._boot_index = 1
            self._apply_theme()
            self._render_static_panels()
            self._refresh_html_options()
            self._refresh_history_options()
            self._refresh_log_options()
            self._set_output(self._build_boot_log(self._boot_index))
            self.set_interval(0.55, self._refresh_dashboard)
            self._boot_timer = self.set_interval(0.16, self._advance_boot)
            self.set_timer(1.75, self._finish_boot)

        def _apply_theme(self) -> None:
            palette = self._palette()
            self.styles.background = palette["screen_bg"]

            root = self.query_one("#root", Vertical)
            root.styles.width = "100%"
            root.styles.height = "100%"
            root.styles.padding = (1, 1)
            root.styles.background = palette["screen_bg"]

            hero = self.query_one("#hero", Static)
            hero.styles.height = 14
            hero.styles.margin = (0, 0, 1, 0)

            layout = self.query_one("#layout", Horizontal)
            layout.styles.height = "1fr"

            sidebar_scroll = self.query_one("#sidebar_scroll", VerticalScroll)
            sidebar_scroll.styles.width = 46
            sidebar_scroll.styles.min_width = 42
            sidebar_scroll.styles.max_width = 50
            sidebar_scroll.styles.height = "1fr"
            sidebar_scroll.styles.margin = (0, 1, 0, 0)
            sidebar_scroll.styles.border = ("round", palette["sidebar_border"])
            sidebar_scroll.styles.background = palette["sidebar_bg"]
            sidebar_scroll.styles.padding = (0, 1, 1, 0)
            sidebar_scroll.show_vertical_scrollbar = True

            sidebar = self.query_one("#sidebar", Vertical)
            sidebar.styles.width = "100%"
            sidebar.styles.height = "auto"
            sidebar.styles.padding = (1, 1, 2, 1)

            for label_id in (
                "#dataset_label",
                "#workflow_label",
                "#variant_label",
                "#preset_label",
                "#budget_label",
                "#html_label",
                "#history_label",
                "#log_label",
            ):
                label = self.query_one(label_id, Static)
                label.styles.color = palette["label_fg"]
                label.styles.text_style = "bold"
                label.styles.margin = (1, 0, 0, 0)
                label.styles.border = ("heavy", palette["label_border"])
                label.styles.background = palette["label_bg"]
                label.styles.padding = (0, 1)

            control_header = self.query_one("#control_header", Static)
            control_header.styles.margin = (0, 0, 1, 0)

            self._style_field("#dataset", palette)
            for select_id in ("#workflow", "#variant", "#preset", "#budget", "#html_pick", "#history_pick", "#log_pick"):
                self._style_field(select_id, palette)

            for check_id in ("#export_html", "#auto_open_html"):
                checkbox = self.query_one(check_id, Checkbox)
                checkbox.styles.color = palette["accent_soft"]
                checkbox.styles.margin = (1, 0, 0, 0)

            for button_id, border, bg, fg in (
                ("#run_btn", "#43D17A", "#103021", "#E3FFE9"),
                ("#open_latest_btn", "#E0A95A", "#37240F", "#FFE0AF"),
                ("#open_selected_btn", "#2E8DB5", "#102B3E", "#DDF8FF"),
                ("#load_history_btn", "#C24652", "#260E11", "#FFE7DD"),
                ("#load_log_btn", "#2E8DB5", "#0D2030", "#E3F8FF"),
                ("#rerun_btn", "#8A6CFF", "#1B1437", "#E2DCFF"),
                ("#refresh_btn", "#4FC3F7", "#092233", "#E3F8FF"),
            ):
                button = self.query_one(button_id, Button)
                button.styles.width = "100%"
                button.styles.height = 3
                button.styles.margin = (1, 0, 0, 0)
                button.styles.border = ("round", border)
                button.styles.background = bg
                button.styles.color = fg
                button.styles.text_style = "bold"

            status_bar = self.query_one("#status_bar", Static)
            status_bar.styles.height = 3
            status_bar.styles.margin = (0, 0, 1, 0)

            cmdline = self.query_one("#cmdline", Input)
            cmdline.styles.height = 3
            cmdline.styles.margin = (0, 0, 1, 0)
            cmdline.styles.border = ("round", palette["cmd_border"])
            cmdline.styles.background = palette["cmd_bg"]
            cmdline.styles.color = palette["cmd_fg"]
            cmdline.display = False

            output_scroll = self.query_one("#output_scroll", VerticalScroll)
            output_scroll.styles.height = "1fr"
            output_scroll.styles.border = ("round", palette["output_border"])
            output_scroll.styles.background = palette["output_bg"]
            output_scroll.styles.padding = (0, 1)
            output_scroll.show_vertical_scrollbar = True

            footer = self.query_one(Footer)
            footer.styles.background = palette["footer_bg"]
            footer.styles.color = palette["footer_fg"]
            footer.styles.border_top = ("heavy", palette["footer_border"])

        def _style_field(self, selector: str, palette: dict[str, str] | None = None) -> None:
            palette = palette or self._palette()
            widget = self.query_one(selector)
            widget.styles.border = ("heavy", palette["field_border"])
            widget.styles.background = palette["field_bg"]
            widget.styles.color = palette["field_fg"]
            widget.styles.margin = (0, 0, 1, 0)
            widget.styles.padding = (0, 1)

        def _palette(self) -> dict[str, str]:
            return self.PALETTES["danger"] if self._danger_workflow() else self.PALETTES["default"]

        def _status_theme_map(self) -> dict[str, dict[str, str]]:
            return self.DANGER_STATUS_THEME if self._danger_workflow() else self.STATUS_THEME

        def _apply_dynamic_palette(self) -> None:
            palette = self._palette()

            self.styles.background = palette["screen_bg"]
            self.query_one("#root", Vertical).styles.background = palette["screen_bg"]

            hero = self.query_one("#hero", Static)
            hero.styles.height = 18 if self._danger_workflow() else 14

            sidebar_scroll = self.query_one("#sidebar_scroll", VerticalScroll)
            sidebar_scroll.styles.border = ("round", palette["sidebar_border"])
            sidebar_scroll.styles.background = palette["sidebar_bg"]
            sidebar_scroll.show_vertical_scrollbar = True

            for label_id in (
                "#dataset_label",
                "#workflow_label",
                "#variant_label",
                "#preset_label",
                "#budget_label",
                "#html_label",
                "#history_label",
                "#log_label",
            ):
                label = self.query_one(label_id, Static)
                label.styles.color = palette["label_fg"]
                label.styles.border = ("heavy", palette["label_border"])
                label.styles.background = palette["label_bg"]

            self._style_field("#dataset", palette)
            for select_id in ("#workflow", "#variant", "#preset", "#budget", "#html_pick", "#history_pick", "#log_pick"):
                self._style_field(select_id, palette)

            cmdline = self.query_one("#cmdline", Input)
            cmdline.styles.border = ("round", palette["cmd_border"])
            cmdline.styles.background = palette["cmd_bg"]
            cmdline.styles.color = palette["cmd_fg"]

            output_scroll = self.query_one("#output_scroll", VerticalScroll)
            output_scroll.styles.border = ("round", palette["output_border"])
            output_scroll.styles.background = palette["output_bg"]
            output_scroll.show_vertical_scrollbar = True

            footer = self.query_one(Footer)
            footer.styles.background = palette["footer_bg"]
            footer.styles.color = palette["footer_fg"]
            footer.styles.border_top = ("heavy", palette["footer_border"])

            run_btn = self.query_one("#run_btn", Button)
            run_btn.styles.border = ("round", palette["accent"])
            run_btn.styles.background = palette["accent_bg"]
            run_btn.styles.color = palette["field_fg"]

            history_btn = self.query_one("#load_history_btn", Button)
            history_btn.styles.border = ("round", palette["artifact_border"])
            history_btn.styles.background = "#10202B" if not self._danger_workflow() else "#2A1115"
            history_btn.styles.color = palette["field_fg"]

            log_btn = self.query_one("#load_log_btn", Button)
            log_btn.styles.border = ("round", palette["output_border"])
            log_btn.styles.background = "#0F2230" if not self._danger_workflow() else "#261317"
            log_btn.styles.color = palette["field_fg"]

            rerun_btn = self.query_one("#rerun_btn", Button)
            rerun_btn.styles.border = ("round", palette["delta_border"])
            rerun_btn.styles.background = "#1B1437" if not self._danger_workflow() else "#2A1115"
            rerun_btn.styles.color = "#E2DCFF" if not self._danger_workflow() else "#FFE4DE"

        def _render_static_panels(self) -> None:
            self.query_one("#control_header", Static).update(self._build_control_header())
            self.query_one("#help_box", Static).update(self._build_help_panel())
            self._render_live_panels()

        def _refresh_dashboard(self) -> None:
            self._tick += 1
            self._render_live_panels()

        def _render_live_panels(self) -> None:
            self._apply_dynamic_palette()
            self.query_one("#hero", Static).update(self._build_hero())
            self.query_one("#status_bar", Static).update(self._build_status_panel())

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

        def _finish_boot(self) -> None:
            if self._last_action == "idle":
                self._set_output(self._build_idle_output())
                self._set_status("ready", "awaiting workflow")

        def _noise_line(self, width: int = 94, offset: int = 0) -> str:
            glyphs = " .:-=+*#"
            seed = (self._tick * 7) + offset + len(self._dataset_short())
            chars: list[str] = []
            for index in range(width):
                code = (seed + index * 5 + (index // 3) * 11) % 17
                if code > 13:
                    chars.append(" ")
                else:
                    chars.append(glyphs[code % len(glyphs)])
            return "".join(chars)

        def _wrap_with_scanlines(self, *renderables: Any) -> Group:
            palette = self._palette()
            top = Text(self._noise_line(112, 1), style=palette["noise"])
            bottom = Text(self._noise_line(112, 9), style=palette["noise"])
            stacked: list[Any] = [top]
            for index, renderable in enumerate(renderables):
                stacked.append(renderable)
                if index < len(renderables) - 1:
                    stacked.append(Text(""))
            stacked.append(bottom)
            return Group(*stacked)

        def _grid_line(self, width: int = 108, offset: int = 0) -> str:
            chars: list[str] = []
            for index in range(width):
                code = (self._tick + offset + index * 3 + (index // 7)) % 19
                if code in {0, 6, 12}:
                    chars.append("+")
                elif code in {2, 9, 15}:
                    chars.append(":")
                elif code in {4, 11}:
                    chars.append(".")
                elif code == 17:
                    chars.append("#")
                else:
                    chars.append(" ")
            return "".join(chars)

        def _radar_sweep(self, width: int = 100, offset: int = 0) -> str:
            sweep = (self._tick * 3 + offset) % max(width, 1)
            chars: list[str] = []
            for index in range(width):
                distance = abs(index - sweep)
                if distance == 0:
                    chars.append("@")
                elif distance == 1:
                    chars.append("#")
                elif distance == 2:
                    chars.append("*")
                elif index % 13 == 0:
                    chars.append("+")
                elif index % 7 == 0:
                    chars.append(":")
                else:
                    chars.append(".")
            return "".join(chars)

        def _danger_workflow(self) -> bool:
            return self._workflow_id() in {"analysis", "robustness", "fairness", "subgroups"}

        def _workflow_id(self) -> str:
            return str(self.query_one("#workflow", Select).value)

        def _selected_spec(self):
            return WORKFLOW_SPECS[self._workflow_id()]

        def _dataset_short(self) -> str:
            value = self.query_one("#dataset", Input).value.strip()
            return Path(value).name if value else "none"

        def _field(self, payload: Any, key: str, default: Any = None) -> Any:
            if isinstance(payload, dict):
                return payload.get(key, default)
            return getattr(payload, key, default)

        def _summary_dict(self, payload: Any) -> dict[str, Any]:
            summary = self._field(payload, "summary", payload if isinstance(payload, dict) else {})
            if not isinstance(summary, dict):
                return {}
            if "rows" not in summary and isinstance(payload, dict) and isinstance(payload.get("shape"), dict):
                summary = {
                    **summary,
                    "rows": payload["shape"].get("rows"),
                    "cols": payload["shape"].get("cols"),
                }
            return summary

        def _dict_field(self, payload: Any, key: str) -> dict[str, Any]:
            value = self._field(payload, key, {})
            return value if isinstance(value, dict) else {}

        def _request(self) -> WorkflowRequest:
            spec = self._selected_spec()
            return WorkflowRequest(
                workflow_id=spec.workflow_id,
                dataset_path=self.query_one("#dataset", Input).value.strip(),
                variant=str(self.query_one("#variant", Select).value or "A"),
                preset=str(self.query_one("#preset", Select).value or spec.default_preset),
                export_html=self.query_one("#export_html", Checkbox).value and spec.supports_export_html,
                console_only=spec.family == "legacy",
                training_budget_mode=str(self.query_one("#budget", Select).value or "default"),
            )

        def _set_status(self, state: str, message: str) -> None:
            self._status_state = state
            self._status_message = message
            self._render_live_panels()

        def _set_output(self, renderable: Any) -> None:
            self.query_one("#output", Static).update(renderable)

        def _build_hero(self) -> Panel:
            palette = self._palette()
            spec = self._selected_spec()
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
                style=f"bold {palette['banner']}",
            )

            telemetry = Text()
            telemetry.append(f" {pulse} ", style=f"bold #0A141B on {palette['accent_soft']}")
            telemetry.append("clock ", style=f"bold {palette['meta_label']}")
            telemetry.append(clock, style=f"bold {palette['meta_value']}")
            telemetry.append("  ||  ", style=f"bold {palette['separator']}")
            telemetry.append("dataset ", style=f"bold {palette['meta_label']}")
            telemetry.append(self._dataset_short(), style=f"bold {palette['meta_value']}")
            telemetry.append("  ||  ", style=f"bold {palette['separator']}")
            telemetry.append("workflow ", style=f"bold {palette['meta_label']}")
            telemetry.append(spec.workflow_id, style=f"bold {palette['hero_title']}")
            telemetry.append("  ||  ", style=f"bold {palette['separator']}")
            telemetry.append("family ", style=f"bold {palette['meta_label']}")
            telemetry.append(spec.family, style=f"bold {palette['accent_soft']}")
            telemetry.append("  ||  ", style=f"bold {palette['separator']}")
            telemetry.append("last ", style=f"bold {palette['meta_label']}")
            telemetry.append(self._last_action, style=f"bold {palette['meta_value']}")

            hotkeys = Text(
                " hotkeys :: [1] run workflow  [2] latest html  [3] open selected html  [4] load history  [5] json dump  [6] console log  [F5] refresh lists  [r] rerun  [:] command  [q] quit ",
                style=f"bold {palette['hotkeys']}",
            )

            renderables: list[Any] = [Text(self._noise_line(118, 3), style=palette["noise"])]
            if self._danger_workflow():
                skull = Text(
                    "\n".join(
                        [
                            "            .ed\"\"\"\" \"\"\"$$$$be.",
                            "          -\"           ^\"\"**$$$e.",
                            "        .\"                   '$$$c",
                            "       /                      \"4$$b",
                            "      d  3                      $$$$",
                            "      $  *                   .$$$$$$",
                            "     .$  ^c           $$$$$e$$$$$$$$.",
                            "     d$L  4.         4$$$$$$$$$$$$$$b",
                        ]
                    ),
                    style=f"bold {palette['skull']}",
                )
                renderables.append(Align.left(skull))
                renderables.append(Text(self._radar_sweep(118, 3), style=palette["accent"]))

            renderables.extend(
                [
                    Align.left(banner),
                    Text(""),
                    Align.left(telemetry),
                    Align.left(hotkeys),
                    Text(self._noise_line(118, 6), style=palette["noise"]),
                ]
            )

            return Panel(
                Group(*renderables),
                title=f"[bold {palette['hero_title']}]{'THREAT MONITOR' if self._danger_workflow() else 'TERMINAL MONITOR'}[/]",
                border_style=palette["hero_border"],
                box=box.DOUBLE,
            )

        def _build_status_panel(self) -> Panel:
            theme = self._status_theme_map()[self._status_state]
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
            palette = self._palette()
            text = Text()
            text.append("CONTROL CARTRIDGES\n", style=f"bold {palette['accent_soft']}")
            text.append("workflow | variant | preset | budget\n", style=palette["meta_value"])
            text.append("html | history | console log | wheel / PgUp / PgDn = scroll", style=palette["hotkeys"])
            return Panel(text, border_style=palette["artifact_border"], box=box.HEAVY)

        def _build_help_panel(self) -> Panel:
            palette = self._palette()
            text = Text()
            text.append("OPERATOR NOTES\n\n", style=f"bold {palette['accent_soft']}")
            text.append("1  RUN WORKFLOW    ", style="bold #E3FFE9")
            text.append("run current workflow selection\n", style="#87B7C8")
            text.append("2  OPEN LATEST     ", style="bold #FFE0AF")
            text.append("open newest html artifact\n", style="#87B7C8")
            text.append("3  OPEN SELECTED   ", style="bold #DDF8FF")
            text.append("open file chosen in picker\n", style="#87B7C8")
            text.append("4  LOAD HISTORY    ", style="bold #FFE7DD")
            text.append("render saved json artifact without rerunning\n", style="#87B7C8")
            text.append("5  JSON DUMP       ", style="bold #D7EEFF")
            text.append("toggle raw json channel with line numbers\n", style="#87B7C8")
            text.append("6  CONSOLE LOG     ", style="bold #DDF8FF")
            text.append("load saved console output with line numbers\n", style="#87B7C8")
            text.append("R  RERUN           ", style="bold #E2DCFF")
            text.append("repeat previous workflow\n", style="#87B7C8")
            text.append(":  COMMAND         ", style="bold #E6DDFF")
            text.append("set workflow / variant / preset / budget\n\n", style="#87B7C8")
            text.append("A / safe     ", style="bold #93F5C6")
            text.append("reduced leakage risk\n", style="#87B7C8")
            text.append("B / full     ", style="bold #FFD67A")
            text.append("full feature research mode\n", style="#87B7C8")
            text.append("auto budget  ", style="bold #67D5FF")
            text.append("maps train budget to max_iter / iterations / n_splines\n", style="#87B7C8")
            text.append("PgUp / PgDn  ", style=f"bold {palette['hero_title']}")
            text.append("scroll sidebar and telemetry stack", style="#87B7C8")
            return Panel(text, title="[bold]NOTES[/]", border_style=palette["sidebar_border"], box=box.HEAVY)

        def _build_idle_output(self) -> Group:
            palette = self._palette()
            text = Text()
            text.append("SYSTEM READY\n\n", style=f"bold {palette['accent_soft']}")
            text.append("telemetry modules will stack here after the first workflow run\n\n", style=palette["meta_value"])
            text.append("> choose workflow then press 1 to run\n", style=palette["meta_label"])
            text.append("> press 2 to open latest html artifact\n", style=palette["meta_label"])
            text.append("> press 3 to open selected html file\n", style=palette["meta_label"])
            text.append("> press 4 to load selected json history artifact\n", style=palette["meta_label"])
            text.append("> press 5 to toggle forensic json dump for current result\n", style=palette["meta_label"])
            text.append("> press 6 to load selected console log artifact\n", style=palette["meta_label"])
            text.append("> press r to rerun previous workflow\n", style=palette["delta_border"])
            text.append("> press : to open command palette\n", style="#E0B8FF")
            text.append("> press F5 to refresh html/json/log artifact lists\n", style=palette["hotkeys"])
            text.append("> use PgUp / PgDn to scroll long control stacks and output modules\n", style=palette["hotkeys"])
            return self._wrap_with_scanlines(
                Panel(
                    Group(
                        Text(self._grid_line(96, 1), style=palette["grid"]),
                        Text(self._radar_sweep(96, 2), style=palette["accent"]),
                        text,
                        Text(self._grid_line(96, 4), style=palette["grid"]),
                    ),
                    title="[bold]OUTPUT[/]",
                    border_style=palette["output_border"],
                    box=box.DOUBLE,
                )
            )

        def _build_boot_log(self, visible_count: int | None = None) -> Group:
            palette = self._palette()
            visible = visible_count if visible_count is not None else len(self._boot_lines)
            lines = self._boot_lines[:visible]
            if visible < len(self._boot_lines):
                lines.append(f"[..] sync {visible:02d}/{len(self._boot_lines):02d}")
            else:
                lines.append("[09] threat posture elevated")
            text = Text("\n".join(lines), style=palette["accent_soft"])
            return self._wrap_with_scanlines(
                Panel(
                    Group(
                        Text(self._radar_sweep(96, 7), style=palette["accent"]),
                        text,
                    ),
                    title="[bold]BOOT LOG[/]",
                    border_style=palette["accent"],
                    box=box.DOUBLE,
                )
            )

        def _telemetry_frame(self, title: str, rows: list[tuple[str, str]], border: str, *, skull: bool = False) -> Panel:
            palette = self._palette()
            table = Table(box=box.SIMPLE_HEAVY, border_style=border, show_header=False, expand=True)
            table.add_column("metric", style="bold #67D5FF", width=20)
            table.add_column("signal", style="#D8FEE3")
            for key, value in rows:
                table.add_row(key, value)
            renderables: list[Any] = [
                Text(self._grid_line(96, 2), style=palette["grid"]),
                Text(self._radar_sweep(96, 5), style=border),
            ]
            if skull:
                renderables.append(
                    Text(
                        "\n".join(
                            [
                                "  .-.",
                                " (o o)",
                                " | O \\",
                                "  \\   \\",
                                "   `~~~'",
                            ]
                        ),
                        style=f"bold {palette['skull']}",
                    )
                )
            renderables.extend([table, Text(self._grid_line(96, 5), style=palette["grid"])])
            return Panel(
                Group(*renderables),
                title=f"[bold]{title}[/]",
                border_style=border,
                box=box.DOUBLE,
            )

        def _channel_frame(self, title: str, body: Any, border: str) -> Panel:
            palette = self._palette()
            return Panel(
                Group(
                    Text(self._grid_line(96, 3), style=palette["grid"]),
                    Text(self._radar_sweep(96, 8), style=border),
                    body,
                    Text(self._grid_line(96, 9), style=palette["grid"]),
                ),
                title=f"[bold]{title}[/]",
                border_style=border,
                box=box.DOUBLE,
            )

        def _json_artifact_paths(self, result: WorkflowResult) -> list[Path]:
            candidates: list[Path] = []

            preferred_keys = {
                "history_json",
                "run_json",
                "comparison_json",
                "profile_json",
                "summary_json",
                "research_json",
            }
            for key, value in result.artifacts.items():
                if not isinstance(value, str):
                    continue
                if not (key in preferred_keys or value.lower().endswith(".json")):
                    continue
                path = Path(value)
                if path.exists() and path.is_file():
                    candidates.append(path.resolve())

            unique = list(dict.fromkeys(candidates))
            priority = {
                "history_json": 0,
                "run_json": 1,
                "comparison_json": 2,
                "profile_json": 3,
                "summary_json": 4,
                "research_json": 5,
            }
            return sorted(
                unique,
                key=lambda path: min(
                    (
                        priority.get(key, 99)
                        for key, value in result.artifacts.items()
                        if value == str(path)
                    ),
                    default=99,
                ),
            )

        def _json_channel_index(self, result: WorkflowResult, paths: list[Path]) -> Panel:
            rows = [("json_channels", str(len(paths)))]
            if paths:
                rows.append(("primary", paths[0].name))
                rows.append(("source", str(paths[0].parent)))
            for index, path in enumerate(paths[1:4], start=2):
                rows.append((f"alt_{index}", path.name))
            return self._telemetry_frame("JSON CHANNEL", rows, self._palette()["artifact_border"])

        def _json_dump_panel(self, path: Path) -> Panel:
            palette = self._palette()
            try:
                text = path.read_text(encoding="utf-8")
                try:
                    parsed = json.loads(text)
                    text = json.dumps(parsed, indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass
            except OSError as exc:
                text = f"Unable to read JSON artifact: {exc}"

            syntax = Syntax(
                text,
                "json",
                theme="monokai",
                line_numbers=True,
                word_wrap=False,
                background_color="default",
            )
            return self._channel_frame(f"FORENSIC JSON :: {path.name}", syntax, palette["output_border"])

        def _console_log_panel(self, title: str, text: str) -> Panel:
            palette = self._palette()
            syntax = Syntax(
                text,
                "text",
                theme="monokai",
                line_numbers=True,
                word_wrap=False,
                background_color="default",
            )
            return self._channel_frame(title, syntax, palette["output_border"])

        def _format_signal(self, value: Any, limit: int = 88) -> str:
            if value is None:
                rendered = "n/a"
            elif isinstance(value, float):
                rendered = f"{value:.4f}"
            elif isinstance(value, dict):
                items = list(value.items())[:4]
                rendered = ", ".join(f"{key}={self._format_signal(item, 18)}" for key, item in items)
                if len(value) > 4:
                    rendered += ", ..."
            elif isinstance(value, (list, tuple, set)):
                seq = list(value)
                rendered = ", ".join(str(item) for item in seq[:6])
                if len(seq) > 6:
                    rendered += ", ..."
            else:
                rendered = str(value)
            return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."

        def _risk_spread_rows(self, report: Any) -> list[tuple[str, str]]:
            metrics: list[tuple[str, float, float | None]] = []
            for model_name, model in self._dict_field(report, "models").items():
                holdout = self._dict_field(model, "holdout")
                roc_auc = holdout.get("roc_auc")
                f1 = holdout.get("f1")
                if roc_auc is not None:
                    metrics.append((model_name, float(roc_auc), float(f1) if f1 is not None else None))

            if not metrics:
                return [("status", "no model telemetry available")]

            best = max(metrics, key=lambda item: item[1])
            worst = min(metrics, key=lambda item: item[1])
            rows = [
                ("best_model", f"{best[0]} roc_auc={best[1]:.4f}"),
                ("weakest_model", f"{worst[0]} roc_auc={worst[1]:.4f}"),
                ("roc_auc_spread", f"{best[1] - worst[1]:.4f}"),
            ]
            if best[2] is not None:
                rows.append(("best_model_f1", f"{best[2]:.4f}"))
            return rows

        def _compare_delta_rows(self, report: Any) -> list[tuple[str, str]]:
            roc_candidates: list[tuple[str, float]] = []
            f1_candidates: list[tuple[str, float]] = []
            for model_name, summary in self._summary_dict(report).items():
                roc_delta = summary.get("roc_auc_delta_full_minus_safe")
                f1_delta = summary.get("f1_delta_full_minus_safe")
                if roc_delta is not None:
                    roc_candidates.append((model_name, float(roc_delta)))
                if f1_delta is not None:
                    f1_candidates.append((model_name, float(f1_delta)))

            rows: list[tuple[str, str]] = []
            if roc_candidates:
                best_roc = max(roc_candidates, key=lambda item: item[1])
                worst_roc = min(roc_candidates, key=lambda item: item[1])
                rows.append(("top_roc_gain", f"{best_roc[0]} {best_roc[1]:+.4f}"))
                rows.append(("worst_roc_drop", f"{worst_roc[0]} {worst_roc[1]:+.4f}"))
            if f1_candidates:
                best_f1 = max(f1_candidates, key=lambda item: item[1])
                worst_f1 = min(f1_candidates, key=lambda item: item[1])
                rows.append(("top_f1_gain", f"{best_f1[0]} {best_f1[1]:+.4f}"))
                rows.append(("worst_f1_drop", f"{worst_f1[0]} {worst_f1[1]:+.4f}"))
            return rows or [("status", "no profile delta available")]

        def _build_result_output(self, result: WorkflowResult) -> Group:
            palette = self._palette()
            json_paths = self._json_artifact_paths(result)
            json_renderables: list[Any] = []
            if self._show_json_dump and json_paths:
                json_renderables = [self._json_channel_index(result, json_paths), self._json_dump_panel(json_paths[0])]

            if result.family == "log":
                assessment_rows = result.summary.get("assessment_rows")
                benchmark_rows = result.summary.get("benchmark_rows")
                if not isinstance(assessment_rows, list) or not isinstance(benchmark_rows, list):
                    analysis = analyze_console_log(
                        str(result.summary.get("workflow", result.workflow_id)),
                        result.transcript,
                        html_artifacts=result.html_artifacts,
                    )
                    assessment_rows = analysis["assessment_rows"]
                    benchmark_rows = analysis["benchmark_rows"]
                return self._wrap_with_scanlines(
                    self._telemetry_frame(
                        "THREAT",
                        [
                            ("workflow", self._format_signal(result.summary.get("workflow", result.workflow_id))),
                            ("family", self._format_signal(result.summary.get("family", "legacy"))),
                            ("dataset", self._format_signal(result.summary.get("dataset"))),
                            ("timestamp", self._format_signal(result.summary.get("timestamp"))),
                        ],
                        palette["signal_border"],
                        skull=self._danger_workflow(),
                    ),
                    self._telemetry_frame(
                        "SIGNAL",
                        [
                            ("variant", self._format_signal(result.summary.get("variant"))),
                            ("preset", self._format_signal(result.summary.get("preset"))),
                            ("budget", self._format_signal(result.summary.get("training_budget_mode"))),
                            ("log_file", self._format_signal(result.summary.get("log_file"))),
                        ],
                        palette["signal_border"],
                    ),
                    self._console_log_panel(
                        f"CONSOLE TRACE :: {self._format_signal(result.summary.get('log_file', 'log'))}",
                        result.transcript or "No console log body loaded.",
                    ),
                    self._telemetry_frame(
                        "OVERALL ASSESSMENT",
                        [(str(key), str(value)) for key, value in assessment_rows],
                        palette["signal_border"],
                    ),
                    self._telemetry_frame(
                        "FLAG BENCHMARK",
                        [(str(key), str(value)) for key, value in benchmark_rows],
                        palette["delta_border"],
                    ),
                    self._telemetry_frame(
                        "RISK DELTA",
                        [
                            ("log_chars", str(len(result.transcript))),
                            ("log_lines", str(len(result.transcript.splitlines()))),
                            ("html_channels", str(len(result.html_artifacts))),
                            ("artifact_keys", str(len(result.artifacts))),
                        ],
                        palette["delta_border"],
                    ),
                    self._artifact_panel(result),
                )

            if result.payload is not None and result.workflow_id == "profile":
                report = result.payload
                summary = self._summary_dict(report)
                warnings_list = self._field(report, "warnings", [])
                safe_columns = summary.get("selected_columns_safe", [])
                full_columns = summary.get("selected_columns_full", [])
                return self._wrap_with_scanlines(
                    self._telemetry_frame(
                        "THREAT",
                        [
                            ("workflow", "dataset profile"),
                            ("rows", self._format_signal(summary.get("rows"))),
                            ("cols", self._format_signal(summary.get("cols"))),
                            ("positive_rate", self._format_signal(summary.get("target_positive_rate"))),
                        ],
                        palette["signal_border"],
                        skull=self._danger_workflow(),
                    ),
                    self._telemetry_frame(
                        "SIGNAL",
                        [
                            ("warnings", str(len(warnings_list))),
                            ("safe_columns", self._format_signal(safe_columns)),
                            ("full_columns", self._format_signal(full_columns)),
                        ],
                        palette["signal_border"],
                    ),
                    self._telemetry_frame(
                        "RISK DELTA",
                        [
                            ("safe_count", str(len(safe_columns))),
                            ("full_count", str(len(full_columns))),
                            ("feature_delta", str(len(full_columns) - len(safe_columns))),
                            ("artifact_keys", str(len(result.artifacts))),
                        ],
                        palette["delta_border"],
                    ),
                    self._artifact_panel(result),
                    *json_renderables,
                )
            if result.payload is not None and result.workflow_id == "run":
                report = result.payload
                config = self._dict_field(report, "config")
                split = self._dict_field(report, "split")
                signal_rows: list[tuple[str, str]] = []
                for model_name, model in self._dict_field(report, "models").items():
                    holdout = self._dict_field(model, "holdout")
                    signal_rows.append(
                        (
                            model_name,
                            f"roc_auc={self._format_signal(holdout.get('roc_auc'), 16)}"
                            f" | f1={self._format_signal(holdout.get('f1'), 16)}",
                        )
                    )
                return self._wrap_with_scanlines(
                    self._telemetry_frame(
                        "THREAT",
                        [
                            ("profile", self._format_signal(config.get("profile"))),
                            ("preset", self._format_signal(config.get("preset"))),
                            ("budget", self._format_signal(config.get("training_budget_mode", "default"))),
                            ("rust_engine", self._format_signal(config.get("rust_engine"))),
                        ],
                        palette["signal_border"],
                        skull=self._danger_workflow(),
                    ),
                    self._telemetry_frame(
                        "SIGNAL",
                        signal_rows
                        + [
                            ("resolved", self._format_signal(config.get("resolved_training_params", {}))),
                            ("test_positive_rate", self._format_signal(split.get("test_positive_rate"))),
                        ],
                        palette["signal_border"],
                    ),
                    self._telemetry_frame("RISK DELTA", self._risk_spread_rows(report), palette["delta_border"]),
                    self._artifact_panel(result),
                    *json_renderables,
                )
            if result.payload is not None and result.workflow_id == "compare":
                report = result.payload
                summary = self._summary_dict(report)
                dataset = self._dict_field(report, "dataset")
                split = self._dict_field(report, "split")
                signal_rows = [
                    (
                        model_name,
                        "safe "
                        f"{self._format_signal(model_summary.get('safe_roc_auc'), 12)} -> "
                        f"full {self._format_signal(model_summary.get('full_roc_auc'), 12)}"
                        " | f1 "
                        f"{self._format_signal(model_summary.get('safe_f1'), 12)} -> "
                        f"{self._format_signal(model_summary.get('full_f1'), 12)}",
                    )
                    for model_name, model_summary in summary.items()
                ]
                return self._wrap_with_scanlines(
                    self._telemetry_frame(
                        "THREAT",
                        [
                            ("preset", self._format_signal(self._field(report, "preset", "quick"))),
                            ("rows", self._format_signal(dataset.get("rows"))),
                            ("train_size", self._format_signal(split.get("train_size"))),
                            ("test_size", self._format_signal(split.get("test_size"))),
                        ],
                        palette["signal_border"],
                        skull=self._danger_workflow(),
                    ),
                    self._telemetry_frame("SIGNAL", signal_rows or [("status", "no compare signal")], palette["signal_border"]),
                    self._telemetry_frame("RISK DELTA", self._compare_delta_rows(report), palette["delta_border"]),
                    self._artifact_panel(result),
                    *json_renderables,
                )

            text = Text()
            if result.summary:
                for key, value in result.summary.items():
                    text.append(f"{key}: {self._format_signal(value)}\n", style=palette["meta_value"])
            if result.transcript:
                text.append("\n" + result.transcript[-5000:], style=palette["meta_value"])
            return self._wrap_with_scanlines(
                self._telemetry_frame(
                    "THREAT",
                    [
                        ("workflow", result.workflow_id),
                        ("family", result.family),
                        ("summary_fields", str(len(result.summary))),
                        ("html_channels", str(len(result.html_artifacts))),
                    ],
                    palette["signal_border"],
                    skull=self._danger_workflow(),
                ),
                self._telemetry_frame(
                    "SIGNAL",
                    [(str(key), self._format_signal(value)) for key, value in list(result.summary.items())[:6]]
                    or [("status", "legacy transcript captured")],
                    palette["signal_border"],
                ),
                self._telemetry_frame(
                    "RISK DELTA",
                    [
                        ("artifacts", str(len(result.artifacts))),
                        ("transcript_chars", str(len(result.transcript))),
                        ("transcript_lines", str(len(result.transcript.splitlines()))),
                    ],
                    palette["delta_border"],
                ),
                self._artifact_panel(result),
                self._channel_frame(
                    "SIGNAL TRACE",
                    text or Text("No transcript captured.", style=palette["meta_value"]),
                    palette["output_border"],
                ),
                *json_renderables,
            )

        def _artifact_panel(self, result: WorkflowResult) -> Panel:
            palette = self._palette()
            table = Table(box=box.SIMPLE_HEAVY, border_style=palette["artifact_border"], show_header=False, expand=True)
            table.add_column("key", style="bold #67D5FF", width=18)
            table.add_column("value", style="#D8FEE3")
            for key, value in result.artifacts.items():
                table.add_row(str(key), str(value))
            for html_path in result.html_artifacts[:6]:
                table.add_row("html", html_path)
            if not result.artifacts and not result.html_artifacts:
                table.add_row("status", "no artifact channel data yet")
            return Panel(
                Group(
                    Text(self._grid_line(96, 8), style=palette["grid"]),
                    Text(self._radar_sweep(96, 11), style=palette["artifact_border"]),
                    table,
                    Text(self._grid_line(96, 12), style=palette["grid"]),
                ),
                title="[bold]ARTIFACT CHANNEL[/]",
                border_style=palette["artifact_border"],
                box=box.DOUBLE,
            )

        def _refresh_html_options(self) -> None:
            html_files = scan_html_artifacts()
            self._last_html = [str(path) for path in html_files[:25]]
            select = self.query_one("#html_pick", Select)
            options = [("select html artifact", "")]
            options.extend((Path(path).name, path) for path in self._last_html)
            select.set_options(options)
            if self._last_html:
                select.value = self._last_html[0]

        def _refresh_history_options(self) -> None:
            history_files = scan_json_artifacts()
            self._last_json = [str(path) for path in history_files[:25]]
            select = self.query_one("#history_pick", Select)
            options = [("select json history", "")]
            options.extend((describe_json_artifact(path), path) for path in self._last_json)
            select.set_options(options)
            if self._last_json:
                select.value = self._last_json[0]

        def _refresh_log_options(self) -> None:
            log_files = scan_log_artifacts()
            self._last_logs = [str(path) for path in log_files[:25]]
            select = self.query_one("#log_pick", Select)
            options = [("select console log", "")]
            options.extend((describe_log_artifact(path), path) for path in self._last_logs)
            select.set_options(options)
            if self._last_logs:
                select.value = self._last_logs[0]

        def _sync_controls_from_history(self, result: WorkflowResult) -> None:
            if result.workflow_id in WORKFLOW_SPECS:
                self.query_one("#workflow", Select).value = result.workflow_id

            summary = result.summary if isinstance(result.summary, dict) else {}
            summary_dataset = summary.get("dataset")
            if summary_dataset:
                self.query_one("#dataset", Input).value = str(summary_dataset)
            summary_variant = str(summary.get("variant", ""))
            if summary_variant.startswith("A"):
                self.query_one("#variant", Select).value = "A"
            elif summary_variant.startswith("B"):
                self.query_one("#variant", Select).value = "B"
            summary_preset = str(summary.get("preset", ""))
            if summary_preset in {"quick", "research"}:
                self.query_one("#preset", Select).value = summary_preset
            summary_budget = str(summary.get("training_budget_mode", ""))
            if summary_budget in {"default", "auto"}:
                self.query_one("#budget", Select).value = summary_budget

            payload = result.payload
            if payload is None:
                return

            if result.workflow_id == "run":
                config = self._dict_field(payload, "config")
                dataset = self._dict_field(payload, "dataset")
                profile = str(config.get("profile", "safe"))
                self.query_one("#variant", Select).value = "A" if profile == "safe" else "B"
                preset = str(config.get("preset", "quick"))
                if preset in {"quick", "research"}:
                    self.query_one("#preset", Select).value = preset
                budget = str(config.get("training_budget_mode", "default"))
                if budget in {"default", "auto"}:
                    self.query_one("#budget", Select).value = budget
                dataset_path = dataset.get("path")
                if dataset_path:
                    self.query_one("#dataset", Input).value = str(dataset_path)

            elif result.workflow_id == "compare":
                preset = str(self._field(payload, "preset", "quick"))
                if preset in {"quick", "research"}:
                    self.query_one("#preset", Select).value = preset
                dataset = self._dict_field(payload, "dataset")
                dataset_path = dataset.get("path")
                if dataset_path:
                    self.query_one("#dataset", Input).value = str(dataset_path)

        def _load_history(self, path: str) -> None:
            if not path:
                self._set_status("error", "no history json selected")
                return
            try:
                result = load_history_result(path)
            except Exception as exc:  # pragma: no cover - interactive path
                self._set_status("error", str(exc))
                return

            self._last_result = result
            self._last_action = f"history:{Path(path).stem}"
            self._show_json_dump = True
            self._sync_controls_from_history(result)
            self._set_output(self._build_result_output(result))
            self._set_status("success", f"history loaded: {Path(path).name}")

        def _load_console_log(self, path: str) -> None:
            if not path:
                self._set_status("error", "no console log selected")
                return
            try:
                result = load_console_log_result(path)
            except Exception as exc:  # pragma: no cover - interactive path
                self._set_status("error", str(exc))
                return

            self._last_result = result
            self._last_action = f"log:{Path(path).stem}"
            self._show_json_dump = False
            self._sync_controls_from_history(result)
            self._set_output(self._build_result_output(result))
            self._set_status("success", f"console log loaded: {Path(path).name}")

        def _validate_request(self) -> WorkflowRequest | None:
            request = self._request()
            if not request.dataset_path:
                self._set_status("error", "dataset path is empty")
                self._set_output(
                    self._wrap_with_scanlines(
                        Panel(
                            Text("Dataset path is empty.", style="bold #FF9FA4"),
                            title="[bold]ERROR[/]",
                            border_style="#E05D5D",
                            box=box.HEAVY,
                        )
                    )
                )
                return None
            return request

        def _start_workflow(self) -> None:
            request = self._validate_request()
            if request is None:
                return
            self._last_request = request
            self._last_action = request.workflow_id
            self._set_status("running", f"running {request.workflow_id}")
            self._set_output(
                self._wrap_with_scanlines(
                    Panel(
                        Text(f"running {request.workflow_id} ...", style="bold #FFD67A"),
                        title="[bold]WORKFLOW[/]",
                        border_style="#E7A93F",
                        box=box.HEAVY,
                    )
                )
            )
            self._workflow_worker(request)

        def action_run_current(self) -> None:
            self._start_workflow()

        def action_rerun_current(self) -> None:
            if self._last_request is None:
                self._start_workflow()
                return
            self._last_action = self._last_request.workflow_id
            self._set_status("running", f"rerunning {self._last_request.workflow_id}")
            self._workflow_worker(self._last_request)

        def action_open_latest(self) -> None:
            target = latest_html_artifact(self._last_result.html_artifacts if self._last_result else self._last_html)
            if target is None:
                self._set_status("error", "no html artifact available")
                return
            try:
                open_html_artifact(target)
            except Exception as exc:  # pragma: no cover - interactive path
                self._set_status("error", str(exc))
                return
            self._last_action = f"open:{Path(target).name}"
            self._set_status("success", f"opened {Path(target).name}")

        def action_open_selected_html(self) -> None:
            self._open_selected_html()

        def action_refresh_html(self) -> None:
            self._refresh_html_options()
            self._refresh_history_options()
            self._refresh_log_options()
            self._last_action = "refresh-html"
            self._set_status(
                "success",
                f"refreshed {len(self._last_html)} html, {len(self._last_json)} json, {len(self._last_logs)} log artifacts",
            )

        def action_load_history(self) -> None:
            target = self.query_one("#history_pick", Select).value
            if not target:
                target = latest_json_artifact(self._last_json)
            if not target:
                self._set_status("error", "no history json available")
                return
            self._load_history(str(target))

        def action_load_console_log(self) -> None:
            target = self.query_one("#log_pick", Select).value
            if not target:
                target = latest_log_artifact(self._last_logs)
            if not target:
                self._set_status("error", "no console log available")
                return
            self._load_console_log(str(target))

        def action_toggle_json_dump(self) -> None:
            self._show_json_dump = not self._show_json_dump
            if self._last_result is not None:
                self._set_output(self._build_result_output(self._last_result))
            else:
                self._set_output(self._build_idle_output())
            self._set_status("success", f"json dump {'enabled' if self._show_json_dump else 'hidden'}")

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

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "run_btn":
                self._start_workflow()
            elif event.button.id == "open_latest_btn":
                self.action_open_latest()
            elif event.button.id == "open_selected_btn":
                self._open_selected_html()
            elif event.button.id == "load_history_btn":
                self.action_load_history()
            elif event.button.id == "load_log_btn":
                self.action_load_console_log()
            elif event.button.id == "rerun_btn":
                self.action_rerun_current()
            elif event.button.id == "refresh_btn":
                self.action_refresh_html()

        def _open_selected_html(self) -> None:
            value = self.query_one("#html_pick", Select).value
            if not value:
                self._refresh_html_options()
                value = self.query_one("#html_pick", Select).value
            if not value:
                value = latest_html_artifact(self._last_result.html_artifacts if self._last_result else self._last_html)
            if not value:
                self._set_status("error", "no html selected")
                return
            try:
                open_html_artifact(str(value))
            except Exception as exc:  # pragma: no cover - interactive path
                self._set_status("error", str(exc))
                return
            self._last_action = f"open:{Path(str(value)).name}"
            self._set_status("success", f"opened {Path(str(value)).name}")

        def on_input_submitted(self, event: Input.Submitted) -> None:
            if event.input.id != "cmdline":
                return
            command = event.value.strip()
            event.input.display = False
            if command and command != ":":
                self._run_command(command)

        def _run_command(self, command: str) -> None:
            normalized = command.lower().strip()
            if normalized.startswith(":"):
                normalized = normalized[1:].strip()

            if normalized in {"run", "1"}:
                self._start_workflow()
                return
            if normalized in {"rerun", "repeat", "r"}:
                self.action_rerun_current()
                return
            if normalized == "html latest":
                self.action_open_latest()
                return
            if normalized == "html open":
                self.action_open_selected_html()
                return
            if normalized == "history latest":
                target = latest_json_artifact(self._last_json)
                if target:
                    self._load_history(target)
                else:
                    self._set_status("error", "no history json available")
                return
            if normalized == "history load":
                self.action_load_history()
                return
            if normalized == "log latest":
                target = latest_log_artifact(self._last_logs)
                if target:
                    self._load_console_log(target)
                else:
                    self._set_status("error", "no console log available")
                return
            if normalized in {"log load", "console log", "console load"}:
                self.action_load_console_log()
                return
            if normalized in {"json", "json toggle", "json dump"}:
                self.action_toggle_json_dump()
                return
            if normalized in {"json on", "json show"}:
                if not self._show_json_dump:
                    self.action_toggle_json_dump()
                else:
                    self._set_status("success", "json dump enabled")
                return
            if normalized in {"json off", "json hide"}:
                if self._show_json_dump:
                    self.action_toggle_json_dump()
                else:
                    self._set_status("success", "json dump hidden")
                return
            if normalized == "refresh html":
                self.action_refresh_html()
                return
            if normalized in {"refresh history", "refresh artifacts"}:
                self.action_refresh_html()
                return
            if normalized in {"help", "h", "?"}:
                self._set_output(self._build_idle_output())
                self._set_status("success", "help rendered")
                return
            if normalized.startswith("set workflow "):
                value = normalized.removeprefix("set workflow ").strip()
                if value in WORKFLOW_SPECS:
                    self.query_one("#workflow", Select).value = value
                    self._set_status("success", f"workflow set to {value}")
                    return
            if normalized.startswith("set variant "):
                value = normalized.removeprefix("set variant ").strip().upper()
                if value in {"A", "B"}:
                    self.query_one("#variant", Select).value = value
                    self._set_status("success", f"variant set to {value}")
                    return
            if normalized.startswith("set preset "):
                value = normalized.removeprefix("set preset ").strip()
                if value in {"quick", "research"}:
                    self.query_one("#preset", Select).value = value
                    self._set_status("success", f"preset set to {value}")
                    return
            if normalized.startswith("set budget "):
                value = normalized.removeprefix("set budget ").strip()
                if value in {"default", "auto"}:
                    self.query_one("#budget", Select).value = value
                    self._set_status("success", f"budget set to {value}")
                    return
            if normalized.startswith("set dataset "):
                value = command.split(" ", 2)[2].strip() if len(command.split(" ", 2)) == 3 else ""
                if value:
                    self.query_one("#dataset", Input).value = value
                    self._set_status("success", f"dataset set to {Path(value).name}")
                    return

            self._set_status("error", f"unknown command: {command}")

        def on_select_changed(self, event: Select.Changed) -> None:
            if event.select.id != "workflow":
                return
            self._render_static_panels()
            if self._last_result is not None and self._last_action != "idle":
                self._set_output(self._build_result_output(self._last_result))
            else:
                self._set_output(self._build_idle_output())

        def on_key(self, event: events.Key) -> None:
            sidebar_scroll = self.query_one("#sidebar_scroll", VerticalScroll)
            output_scroll = self.query_one("#output_scroll", VerticalScroll)

            if event.key == "pageup":
                sidebar_scroll.scroll_page_up(animate=False)
                output_scroll.scroll_page_up(animate=False)
                event.stop()
            elif event.key == "pagedown":
                sidebar_scroll.scroll_page_down(animate=False)
                output_scroll.scroll_page_down(animate=False)
                event.stop()
            elif event.key == "home":
                sidebar_scroll.scroll_home(animate=False)
                output_scroll.scroll_home(animate=False)
                event.stop()
            elif event.key == "end":
                sidebar_scroll.scroll_end(animate=False)
                output_scroll.scroll_end(animate=False)
                event.stop()

        @work(thread=True, exclusive=True)
        def _workflow_worker(self, request: WorkflowRequest) -> None:
            try:
                result = execute_workflow(request)
                self.call_from_thread(self._workflow_done, result)
            except Exception as exc:  # pragma: no cover - interactive path
                self.call_from_thread(self._workflow_failed, exc)

        def _workflow_done(self, result: WorkflowResult) -> None:
            self._last_result = result
            self._last_action = result.workflow_id
            self._set_output(self._build_result_output(result))
            self._refresh_html_options()
            self._refresh_history_options()
            self._refresh_log_options()
            self._set_status("success", f"{result.workflow_id} completed")
            if self.query_one("#auto_open_html", Checkbox).value and result.html_artifacts:
                self.action_open_latest()

        def _workflow_failed(self, exc: Exception) -> None:
            self._set_output(
                self._wrap_with_scanlines(
                    Panel(
                        Text(str(exc), style="bold #FF9FA4"),
                        title="[bold]ERROR[/]",
                        border_style="#E05D5D",
                        box=box.HEAVY,
                    )
                )
            )
            self._last_action = "error"
            self._set_status("error", "workflow failed")

    DepressionTUI().run()
