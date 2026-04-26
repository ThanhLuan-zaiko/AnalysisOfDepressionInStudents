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
            Binding("t", "train_best_model", "Train Model"),
            Binding("p", "predict_current", "Predict"),
            Binding("f5", "refresh_html", "Refresh", priority=True),
            Binding(":", "toggle_command_palette", "Command", priority=True),
            Binding("r", "rerun_current", "Rerun", priority=True),
            Binding("q", "quit", "Quit", priority=True),
        ]
        PREDICTION_INPUT_IDS = (
            "#pred_model_path",
            "#pred_age",
            "#pred_city",
            "#pred_cgpa",
            "#pred_degree",
            "#pred_work_study_hours",
        )
        PREDICTION_SELECT_IDS = (
            "#pred_gender",
            "#pred_academic_pressure",
            "#pred_study_satisfaction",
            "#pred_sleep_duration",
            "#pred_dietary_habits",
            "#pred_financial_stress",
            "#pred_family_history",
            "#pred_suicidal",
        )
        PREDICTION_FIELD_LABEL_IDS = (
            "#pred_model_path_label",
            "#pred_gender_label",
            "#pred_age_label",
            "#pred_city_label",
            "#pred_academic_pressure_label",
            "#pred_cgpa_label",
            "#pred_study_satisfaction_label",
            "#pred_sleep_duration_label",
            "#pred_dietary_habits_label",
            "#pred_degree_label",
            "#pred_work_study_hours_label",
            "#pred_financial_stress_label",
            "#pred_family_history_label",
            "#pred_suicidal_label",
        )

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
                            yield Static(id="artifact_hint")
                            yield Button("1  RUN WORKFLOW", id="run_btn")
                            yield Button("2  OPEN LATEST", id="open_latest_btn")
                            yield Button("3  OPEN SELECTED", id="open_selected_btn")
                            yield Button("4  LOAD HISTORY", id="load_history_btn")
                            yield Button("6  LOAD CONSOLE LOG", id="load_log_btn")
                            yield Button("R  RERUN CURRENT", id="rerun_btn")
                            yield Button("REFRESH ARTIFACT LISTS", id="refresh_btn")
                            yield Button("T  TẠO MODEL DỰ ĐOÁN", id="train_best_btn")
                            yield Static("dự đoán trầm cảm", id="prediction_label")
                            yield Static(id="prediction_hint")
                            yield Static("Đường dẫn model đã train", id="pred_model_path_label")
                            yield Input(
                                value="models/best_depression_model.joblib",
                                placeholder="models/best_depression_model.joblib",
                                id="pred_model_path",
                            )
                            yield Static("Giới tính", id="pred_gender_label")
                            yield Select([("Nữ", "Female"), ("Nam", "Male")], value="Female", prompt="Chọn giới tính", id="pred_gender")
                            yield Static("Tuổi", id="pred_age_label")
                            yield Input(value="22", placeholder="Nhập tuổi", id="pred_age")
                            yield Static("Thành phố", id="pred_city_label")
                            yield Input(value="Hanoi", placeholder="VD: Hanoi", id="pred_city")
                            yield Static("Áp lực học tập (1 = thấp, 5 = cao)", id="pred_academic_pressure_label")
                            yield Select(
                                [(f"{value}", str(value)) for value in range(1, 6)],
                                value="3",
                                prompt="Chọn mức áp lực",
                                id="pred_academic_pressure",
                            )
                            yield Static("CGPA / điểm trung bình", id="pred_cgpa_label")
                            yield Input(value="7.5", placeholder="VD: 7.5", id="pred_cgpa")
                            yield Static("Mức hài lòng học tập (1 = thấp, 5 = cao)", id="pred_study_satisfaction_label")
                            yield Select(
                                [(f"{value}", str(value)) for value in range(1, 6)],
                                value="3",
                                prompt="Chọn mức hài lòng",
                                id="pred_study_satisfaction",
                            )
                            yield Static("Thời lượng ngủ mỗi ngày", id="pred_sleep_duration_label")
                            yield Select(
                                [
                                    ("Dưới 5 giờ", "Less than 5 hours"),
                                    ("5-6 giờ", "5-6 hours"),
                                    ("7-8 giờ", "7-8 hours"),
                                    ("Trên 8 giờ", "More than 8 hours"),
                                ],
                                value="5-6 hours",
                                prompt="Chọn thời lượng ngủ",
                                id="pred_sleep_duration",
                            )
                            yield Static("Thói quen ăn uống", id="pred_dietary_habits_label")
                            yield Select(
                                [("Lành mạnh", "Healthy"), ("Trung bình", "Moderate"), ("Không lành mạnh", "Unhealthy")],
                                value="Moderate",
                                prompt="Chọn thói quen ăn uống",
                                id="pred_dietary_habits",
                            )
                            yield Static("Bậc học / bằng cấp", id="pred_degree_label")
                            yield Input(value="Bachelor", placeholder="VD: Bachelor", id="pred_degree")
                            yield Static("Số giờ học/làm mỗi ngày", id="pred_work_study_hours_label")
                            yield Input(value="6", placeholder="VD: 6", id="pred_work_study_hours")
                            yield Static("Áp lực tài chính (1 = thấp, 5 = cao)", id="pred_financial_stress_label")
                            yield Select(
                                [(f"{value}", str(value)) for value in range(1, 6)],
                                value="3",
                                prompt="Chọn mức tài chính",
                                id="pred_financial_stress",
                            )
                            yield Static("Gia đình có tiền sử bệnh tâm lý?", id="pred_family_history_label")
                            yield Select([("Không", "No"), ("Có", "Yes")], value="No", prompt="Chọn có/không", id="pred_family_history")
                            yield Static("Từng có ý nghĩ tự tử?", id="pred_suicidal_label")
                            yield Select([("Không", "No"), ("Có", "Yes")], value="No", prompt="Chọn có/không", id="pred_suicidal")
                            yield Button("P  DỰ ĐOÁN NGUY CƠ", id="predict_btn")
                            yield Static(id="help_box")
                    with Vertical(id="workspace"):
                        yield Static(id="status_bar")
                        yield Input(placeholder=": command", id="cmdline")
                        with VerticalScroll(id="output_scroll"):
                            yield Static(id="output")
                    with VerticalScroll(id="intel_scroll"):
                        yield Static(id="intel_panel")
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
            self._json_cache: dict[str, tuple[float, dict[str, Any]]] = {}
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
            self._refresh_html_options()
            self._refresh_history_options()
            self._refresh_log_options()
            self._render_static_panels()
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
            hero.styles.height = 12
            hero.styles.margin = (0, 0, 1, 0)

            layout = self.query_one("#layout", Horizontal)
            layout.styles.height = "1fr"

            sidebar_scroll = self.query_one("#sidebar_scroll", VerticalScroll)
            sidebar_scroll.styles.width = 46
            sidebar_scroll.styles.min_width = 38
            sidebar_scroll.styles.max_width = 56
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
                "#prediction_label",
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

            for field_label_id in self.PREDICTION_FIELD_LABEL_IDS:
                self._style_prediction_field_label(field_label_id, palette)

            self._style_field("#dataset", palette)
            for input_id in self.PREDICTION_INPUT_IDS:
                self._style_field(input_id, palette)
            for select_id in ("#workflow", "#variant", "#preset", "#budget", "#html_pick", "#history_pick", "#log_pick"):
                self._style_field(select_id, palette)
            for select_id in self.PREDICTION_SELECT_IDS:
                self._style_field(select_id, palette)

            artifact_hint = self.query_one("#artifact_hint", Static)
            artifact_hint.styles.margin = (0, 0, 1, 0)
            artifact_hint.styles.width = "100%"

            prediction_hint = self.query_one("#prediction_hint", Static)
            prediction_hint.styles.margin = (0, 0, 1, 0)
            prediction_hint.styles.width = "100%"

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
                ("#train_best_btn", "#43D17A", "#102A1D", "#E3FFE9"),
                ("#predict_btn", "#FFB86B", "#33210D", "#FFF1C2"),
            ):
                button = self.query_one(button_id, Button)
                button.styles.width = "100%"
                button.styles.height = 3
                button.styles.margin = (1, 0, 0, 0)
                button.styles.border = ("round", border)
                button.styles.background = bg
                button.styles.color = fg
                button.styles.text_style = "bold"

            workspace = self.query_one("#workspace", Vertical)
            workspace.styles.width = "1fr"
            workspace.styles.min_width = 50
            workspace.styles.height = "1fr"

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

            intel_scroll = self.query_one("#intel_scroll", VerticalScroll)
            intel_scroll.styles.width = 38
            intel_scroll.styles.min_width = 28
            intel_scroll.styles.max_width = 44
            intel_scroll.styles.height = "1fr"
            intel_scroll.styles.margin = (0, 0, 0, 1)
            intel_scroll.styles.border = ("round", palette["artifact_border"])
            intel_scroll.styles.background = palette["sidebar_bg"]
            intel_scroll.styles.padding = (0, 1)
            intel_scroll.show_vertical_scrollbar = True

            intel_panel = self.query_one("#intel_panel", Static)
            intel_panel.styles.width = "100%"
            intel_panel.styles.height = "auto"

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
            if isinstance(widget, Select):
                widget.compact = True
                widget.styles.height = 3
                try:
                    overlay = widget.query_one("SelectOverlay")
                except Exception:
                    overlay = None
                if overlay is not None:
                    overlay.styles.max_height = 18
                    if selector in {"#html_pick", "#history_pick", "#log_pick"}:
                        overlay.styles.width = 96

        def _style_prediction_field_label(self, selector: str, palette: dict[str, str] | None = None) -> None:
            palette = palette or self._palette()
            label = self.query_one(selector, Static)
            label.styles.color = palette["accent_soft"]
            label.styles.text_style = "bold"
            label.styles.margin = (1, 0, 0, 0)
            label.styles.padding = (0, 1)
            label.styles.background = palette["sidebar_bg"]
            label.styles.width = "100%"

        def _palette(self) -> dict[str, str]:
            return self.PALETTES["danger"] if self._danger_workflow() else self.PALETTES["default"]

        def _status_theme_map(self) -> dict[str, dict[str, str]]:
            return self.DANGER_STATUS_THEME if self._danger_workflow() else self.STATUS_THEME

        def _apply_dynamic_palette(self) -> None:
            palette = self._palette()

            self.styles.background = palette["screen_bg"]
            self.query_one("#root", Vertical).styles.background = palette["screen_bg"]

            hero = self.query_one("#hero", Static)
            hero.styles.height = 13 if self._danger_workflow() else 12

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
                "#prediction_label",
            ):
                label = self.query_one(label_id, Static)
                label.styles.color = palette["label_fg"]
                label.styles.border = ("heavy", palette["label_border"])
                label.styles.background = palette["label_bg"]

            for field_label_id in self.PREDICTION_FIELD_LABEL_IDS:
                self._style_prediction_field_label(field_label_id, palette)

            self._style_field("#dataset", palette)
            for input_id in self.PREDICTION_INPUT_IDS:
                self._style_field(input_id, palette)
            for select_id in ("#workflow", "#variant", "#preset", "#budget", "#html_pick", "#history_pick", "#log_pick"):
                self._style_field(select_id, palette)
            for select_id in self.PREDICTION_SELECT_IDS:
                self._style_field(select_id, palette)

            cmdline = self.query_one("#cmdline", Input)
            cmdline.styles.border = ("round", palette["cmd_border"])
            cmdline.styles.background = palette["cmd_bg"]
            cmdline.styles.color = palette["cmd_fg"]

            output_scroll = self.query_one("#output_scroll", VerticalScroll)
            output_scroll.styles.border = ("round", palette["output_border"])
            output_scroll.styles.background = palette["output_bg"]
            output_scroll.show_vertical_scrollbar = True

            intel_scroll = self.query_one("#intel_scroll", VerticalScroll)
            intel_scroll.styles.border = ("round", palette["artifact_border"])
            intel_scroll.styles.background = palette["sidebar_bg"]
            intel_scroll.show_vertical_scrollbar = True

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

            predict_btn = self.query_one("#predict_btn", Button)
            predict_btn.styles.border = ("round", "#FFB86B")
            predict_btn.styles.background = "#33210D" if not self._danger_workflow() else "#30100A"
            predict_btn.styles.color = "#FFF1C2"

            train_best_btn = self.query_one("#train_best_btn", Button)
            train_best_btn.styles.border = ("round", palette["accent"])
            train_best_btn.styles.background = palette["accent_bg"]
            train_best_btn.styles.color = palette["field_fg"]

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
            self.query_one("#intel_panel", Static).update(self._build_intel_panel())
            self.query_one("#artifact_hint", Static).update(self._build_artifact_hint())
            self.query_one("#prediction_hint", Static).update(self._build_prediction_hint())

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

        def _build_robot_logo(self) -> Text:
            palette = self._palette()
            width = getattr(self.size, "width", 140)
            if width < 92:
                logo = Text()
                logo.append("  ROBOT  ", style=f"bold {palette['screen_bg']} on {palette['hero_title']}")
                logo.append("  workflow command core", style=f"bold {palette['accent_soft']}")
                return logo

            return Text(
                "\n".join(
                    [
                        "  _______    ______    _______    ______   _________ ",
                        " |  ___  \\  /  __  \\  |  ___  \\  /  __  \\ |___   ___|",
                        " | |   | | |  |  |  | | |   | | |  |  |  |    | |    ",
                        " | |___| | |  |  |  | | |___| | |  |  |  |    | |    ",
                        " |  __  /  |  |  |  | |  __  /  |  |  |  |    | |    ",
                        " | |  \\ \\  |  |__|  | | |___| | |  |__|  |    | |    ",
                        " |_|   \\_\\  \\______/  |_______/  \\______/     |_|    ",
                    ]
                ),
                style=f"bold {palette['banner']}",
            )

        def _build_robot_stage(self) -> Any:
            width = getattr(self.size, "width", 140)
            if width < 118:
                return self._build_robot_logo()

            grid = Table.grid(expand=True)
            grid.add_column(width=58)
            grid.add_column(ratio=1)
            grid.add_row(self._build_robot_logo(), self._build_robot_sidecar())
            return grid

        def _build_robot_sidecar(self) -> Text:
            palette = self._palette()
            selection = self._read_json_cached(Path("results") / "best_model_selection.json")
            clustering = self._read_json_cached(Path("results") / "visualizations" / "famd_clustering_results.json")
            kmeans = clustering.get("kmeans", {}) if isinstance(clustering.get("kmeans"), dict) else {}
            dbscan = clustering.get("dbscan", {}) if isinstance(clustering.get("dbscan"), dict) else {}

            model = str(selection.get("model", "pending")).upper()
            auc = self._format_signal(selection.get("roc_auc"), 10)
            f1 = self._format_signal(selection.get("f1"), 10)
            kmeans_line = f"k={kmeans.get('best_k', 'n/a')} sil={self._format_signal(kmeans.get('silhouette'), 8)}"
            dbscan_line = (
                f"{dbscan.get('n_clusters', 'n/a')} clusters"
                if dbscan.get("found_valid_clusters")
                else "density pending"
            )

            text = Text()
            text.append("  +-- MISSION TELEMETRY", style=f"bold {palette['hero_title']}")
            text.append(" " + "-" * 28 + "\n", style=palette["grid"])
            text.append("  | CORE      ", style=f"bold {palette['meta_label']}")
            text.append("workflow command matrix\n", style=f"bold {palette['accent_soft']}")
            text.append("  | MODEL     ", style=f"bold {palette['meta_label']}")
            text.append(f"{model}  auc={auc}  f1={f1}\n", style=f"bold {palette['meta_value']}")
            text.append("  | FAMD      ", style=f"bold {palette['meta_label']}")
            text.append(f"KMeans {kmeans_line}  DBSCAN {dbscan_line}\n", style=f"bold {palette['meta_value']}")
            text.append("  | ARTIFACT  ", style=f"bold {palette['meta_label']}")
            text.append(
                f"html={len(self._last_html)}  json={len(self._last_json)}  log={len(self._last_logs)}\n",
                style=f"bold {palette['meta_value']}",
            )
            text.append("  | HOT PATH  ", style=f"bold {palette['meta_label']}")
            text.append("1 run  T model  P predict  4 history  6 log\n", style=f"bold {palette['hotkeys']}")
            text.append("  +", style=palette["grid"])
            text.append("-" * 49, style=palette["grid"])
            return text

        def _build_hero(self) -> Panel:
            palette = self._palette()
            spec = self._selected_spec()
            pulse = "|/-\\"[self._tick % 4] if self._status_state == "running" else ">"
            clock = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            banner = Text()
            banner.append("SEN ANALYTICS", style=f"bold {palette['banner']}")
            banner.append("  //  ", style=f"bold {palette['separator']}")
            banner.append("HACKER PRO OPS GRID", style=f"bold {palette['hero_title']}")
            banner.append("  //  ", style=f"bold {palette['separator']}")
            banner.append("multi-pane telemetry", style=f"bold {palette['hotkeys']}")

            robot_mark = Text()
            robot_mark.append(" < workflow command core >", style=f"bold {palette['accent_soft']}")
            robot_mark.append("  ::  ", style=f"bold {palette['separator']}")
            robot_mark.append("model | famd | report | artifact", style=f"bold {palette['meta_value']}")

            telemetry = Text()
            telemetry.append(f" {pulse} ", style=f"bold #0A141B on {palette['accent_soft']}")
            telemetry.append("clock ", style=f"bold {palette['meta_label']}")
            telemetry.append(clock, style=f"bold {palette['meta_value']}")
            telemetry.append(" | ", style=f"bold {palette['separator']}")
            telemetry.append("dataset ", style=f"bold {palette['meta_label']}")
            telemetry.append(self._dataset_short(), style=f"bold {palette['meta_value']}")
            telemetry.append(" | ", style=f"bold {palette['separator']}")
            telemetry.append("workflow ", style=f"bold {palette['meta_label']}")
            telemetry.append(spec.workflow_id, style=f"bold {palette['hero_title']}")
            telemetry.append(" | ", style=f"bold {palette['separator']}")
            telemetry.append("family ", style=f"bold {palette['meta_label']}")
            telemetry.append(spec.family, style=f"bold {palette['accent_soft']}")
            telemetry.append(" | ", style=f"bold {palette['separator']}")
            telemetry.append("last ", style=f"bold {palette['meta_label']}")
            telemetry.append(self._last_action, style=f"bold {palette['meta_value']}")

            hotkeys = Text(
                " [1] run  [2] latest html  [3] selected html  [4] history  "
                "[5] json  [6] log  [T] model  [P] predict  [F5] refresh  [r] rerun  [:] command  [q] quit ",
                style=f"bold {palette['hotkeys']}",
            )

            renderables: list[Any] = [
                Text(self._noise_line(118, 3), style=palette["noise"]),
                Align.left(banner),
                Align.left(self._build_robot_stage()),
                Align.left(robot_mark),
            ]
            if self._danger_workflow():
                alert = Text("attention mode: audit workflow selected", style=f"bold {palette['skull']}")
                renderables.append(Align.left(alert))

            renderables.extend(
                [
                    Align.left(telemetry),
                    Align.left(hotkeys),
                    Text(self._noise_line(118, 6), style=palette["noise"]),
                ]
            )

            return Panel(
                Group(*renderables),
                title=(
                    f"[bold {palette['hero_title']}]"
                    f"{'AUDIT MONITOR' if self._danger_workflow() else 'TERMINAL MONITOR'}[/]"
                ),
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
            text.append("html | history | train/export model | prediction form | scroll", style=palette["hotkeys"])
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
            text.append("T  TRAIN MODEL     ", style="bold #E3FFE9")
            text.append("train/export best model artifact for prediction\n", style="#87B7C8")
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
            text.append("scroll control, output and intel lanes", style="#87B7C8")
            return Panel(text, title="[bold]NOTES[/]", border_style=palette["sidebar_border"], box=box.HEAVY)

        def _build_intel_panel(self) -> Group:
            palette = self._palette()
            header = Text()
            header.append("HACKER PRO RAIL\n", style=f"bold {palette['hero_title']}")
            header.append("live snapshot without rerun", style=palette["hotkeys"])

            return Group(
                Panel(
                    Group(
                        Text(self._grid_line(36, 1), style=palette["grid"]),
                        header,
                        Text(self._radar_sweep(36, 2), style=palette["accent"]),
                    ),
                    title="[bold]INTEL[/]",
                    border_style=palette["hero_border"],
                    box=box.DOUBLE,
                ),
                Text(""),
                self._compact_frame("SESSION", self._session_rows(), palette["signal_border"]),
                Text(""),
                self._compact_frame("BEST MODEL", self._best_model_rows(), palette["accent"]),
                Text(""),
                self._compact_frame("FAMD CLUSTER", self._famd_rows(), palette["delta_border"]),
                Text(""),
                self._compact_frame("ARTIFACTS", self._artifact_inventory_rows(), palette["artifact_border"]),
                Text(""),
                self._compact_frame("WORKFLOW MAP", self._workflow_rows(), palette["output_border"]),
                Text(""),
                self._compact_frame("FAST PATH", self._operator_rows(), palette["hero_title"]),
            )

        def _compact_frame(self, title: str, rows: list[tuple[str, str]], border: str) -> Panel:
            palette = self._palette()
            table = Table(box=box.SIMPLE, show_header=False, expand=True, pad_edge=False)
            table.add_column("key", style=f"bold {palette['meta_label']}", width=11, no_wrap=True)
            table.add_column("value", style=palette["meta_value"], ratio=1)
            for key, value in rows:
                table.add_row(key, value)
            return Panel(table, title=f"[bold]{title}[/]", border_style=border, box=box.ROUNDED)

        def _read_json_cached(self, path: Path) -> dict[str, Any]:
            key = str(path)
            try:
                stat = path.stat()
            except OSError:
                self._json_cache.pop(key, None)
                return {}

            cached = self._json_cache.get(key)
            if cached is not None and cached[0] == stat.st_mtime:
                return cached[1]

            try:
                parsed = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                parsed = {"_error": str(exc)}

            data = parsed if isinstance(parsed, dict) else {"_value": parsed}
            self._json_cache[key] = (stat.st_mtime, data)
            return data

        def _format_pct(self, value: Any) -> str:
            if value is None:
                return "n/a"
            try:
                return f"{float(value) * 100:.1f}%"
            except (TypeError, ValueError):
                return "n/a"

        def _session_rows(self) -> list[tuple[str, str]]:
            request = self._last_request
            result = self._last_result
            rows = [
                ("state", self._status_state),
                ("workflow", self._workflow_id()),
                ("variant", str(self.query_one("#variant", Select).value or "A")),
                ("preset", str(self.query_one("#preset", Select).value or "quick")),
                ("budget", str(self.query_one("#budget", Select).value or "default")),
                ("last", self._format_signal(self._last_action, 28)),
            ]
            if request is not None:
                rows.append(("queued", f"{request.workflow_id}/{request.variant}/{request.preset}"))
            if result is not None:
                rows.append(("loaded", f"{result.workflow_id} ({result.family})"))
                rows.append(("html_out", str(len(result.html_artifacts))))
                rows.append(("json_out", str(len(self._json_artifact_paths(result)))))
            return rows

        def _best_model_rows(self) -> list[tuple[str, str]]:
            selection = self._read_json_cached(Path("results") / "best_model_selection.json")
            comparison = self._read_json_cached(Path("results") / "model_comparison_report.json")
            if not selection:
                return [("status", "run report/models first")]

            rows = [
                ("model", str(selection.get("model", "n/a")).upper()),
                ("profile", self._format_signal(selection.get("profile"), 24)),
                ("roc_auc", self._format_signal(selection.get("roc_auc"), 12)),
                ("pr_auc", self._format_signal(selection.get("pr_auc"), 12)),
                ("f1", self._format_signal(selection.get("f1"), 12)),
                ("brier", self._format_signal(selection.get("brier_score"), 12)),
            ]

            ranking = comparison.get("ranking")
            if isinstance(ranking, list) and ranking:
                top = ranking[0] if isinstance(ranking[0], dict) else {}
                runner_up = ranking[1] if len(ranking) > 1 and isinstance(ranking[1], dict) else {}
                if runner_up:
                    try:
                        gap = float(top.get("roc_auc", 0.0)) - float(runner_up.get("roc_auc", 0.0))
                        rows.append(("auc_gap", f"{gap:+.4f} vs {runner_up.get('model')}"))
                    except (TypeError, ValueError):
                        rows.append(("runner_up", self._format_signal(runner_up.get("model"), 20)))
                dummy = next(
                    (item for item in ranking if isinstance(item, dict) and item.get("model") == "dummy"),
                    None,
                )
                if isinstance(dummy, dict):
                    try:
                        lift = float(top.get("roc_auc", 0.0)) - float(dummy.get("roc_auc", 0.0))
                        rows.append(("vs_dummy", f"{lift:+.4f} ROC-AUC"))
                    except (TypeError, ValueError):
                        pass

            reason = selection.get("reason")
            if reason:
                rows.append(("why", self._format_signal(reason, 38)))
            return rows

        def _famd_rows(self) -> list[tuple[str, str]]:
            clustering = self._read_json_cached(Path("results") / "visualizations" / "famd_clustering_results.json")
            summary = self._read_json_cached(Path("results") / "visualizations" / "famd_summary.json")
            if not clustering:
                return [("status", "run FAMD first")]

            kmeans = clustering.get("kmeans", {}) if isinstance(clustering.get("kmeans"), dict) else {}
            dbscan = clustering.get("dbscan", {}) if isinstance(clustering.get("dbscan"), dict) else {}
            rows = [
                ("samples", self._format_signal(clustering.get("n_samples"), 12)),
                ("dims", self._format_signal(clustering.get("n_dims"), 12)),
                ("kmeans", f"k={kmeans.get('best_k', 'n/a')} sil={self._format_signal(kmeans.get('silhouette'), 8)}"),
            ]
            if dbscan.get("found_valid_clusters"):
                rows.append(
                    (
                        "dbscan",
                        f"{dbscan.get('n_clusters', 'n/a')} clusters, "
                        f"noise={self._format_pct(dbscan.get('noise_fraction'))}",
                    )
                )
            else:
                rows.append(("dbscan", "no stable density clusters"))

            explained = summary.get("explained_variance_ratio")
            cumulative = summary.get("cumulative_variance")
            if isinstance(explained, list) and explained:
                rows.append(("F1_var", self._format_pct(explained[0])))
            if isinstance(cumulative, list) and len(cumulative) >= 3:
                rows.append(("F1-F3", self._format_pct(cumulative[2])))
            return rows

        def _artifact_inventory_rows(self) -> list[tuple[str, str]]:
            html_groups: dict[str, int] = {}
            for path in self._last_html:
                group = self._html_artifact_group(path)
                html_groups[group] = html_groups.get(group, 0) + 1
            rows = [
                ("html", f"{len(self._last_html)} indexed"),
                ("json", f"{len(self._last_json)} indexed"),
                ("logs", f"{len(self._last_logs)} indexed"),
            ]
            if html_groups:
                rows.append(
                    (
                        "groups",
                        ", ".join(f"{name.lower()}={count}" for name, count in sorted(html_groups.items())),
                    )
                )
            if self._last_html:
                rows.append(("latest_h", self._format_signal(Path(self._last_html[0]).name, 30)))
            if self._last_json:
                rows.append(("latest_j", self._format_signal(Path(self._last_json[0]).name, 30)))
            if self._last_logs:
                rows.append(("latest_l", self._format_signal(Path(self._last_logs[0]).name, 30)))
            return rows

        def _artifact_select_label(self, path: str | Path, kind: str) -> str:
            artifact_path = Path(path)
            if kind == "json":
                name = self._middle_truncate(artifact_path.name, 32)
                try:
                    workflow = describe_json_artifact(artifact_path).split("|")[1].strip()
                except Exception:
                    workflow = "json"
                return f"{workflow}: {name}"
            if kind == "log":
                name = self._middle_truncate(artifact_path.name, 32)
                try:
                    workflow = describe_log_artifact(artifact_path).split("|")[1].strip()
                except Exception:
                    workflow = "log"
                return f"{workflow}: {name}"
            group = self._html_artifact_group(artifact_path)
            title = self._html_artifact_title(artifact_path)
            short_path = self._middle_truncate(self._relative_artifact_path(artifact_path), 46)
            return f"{group:<10} | {title:<26} | {short_path}"

        def _relative_artifact_path(self, path: str | Path) -> str:
            artifact_path = Path(path).resolve()
            try:
                return str(artifact_path.relative_to(Path.cwd()))
            except ValueError:
                return str(artifact_path)

        def _html_artifact_group(self, path: str | Path) -> str:
            name = Path(path).name.lower()
            if name.startswith("eda_"):
                return "EDA"
            if name == "final_report.html" or "evidence" in name or name.startswith("model_feature"):
                return "REPORT"
            if name in {"model_comparison.html", "calibration_curves.html", "decision_curves.html"}:
                return "MODEL"
            if name.startswith("fairness_"):
                return "FAIRNESS"
            if name.startswith("subgroup_"):
                return "SUBGROUP"
            if name.startswith("robustness_"):
                return "ROBUST"
            if name.startswith("famd_"):
                return "FAMD"
            if name.startswith("gam_"):
                return "GAM"
            return "HTML"

        def _html_artifact_title(self, path: str | Path) -> str:
            name = Path(path).stem.lower()
            title_map = {
                "final_report": "Final report",
                "model_evidence_metrics": "Model evidence",
                "model_feature_importance_safe": "Feature importance",
                "model_comparison": "Model comparison",
                "calibration_curves": "Calibration curves",
                "decision_curves": "Decision curves",
                "gam_feature_effects": "GAM effects",
                "eda_class_imbalance": "Class imbalance",
                "eda_missing_values": "Missing values",
                "eda_numeric_distributions": "Numeric distributions",
                "eda_categorical_distributions": "Categorical distributions",
                "eda_suicidal_thoughts": "Suicidal thoughts",
                "eda_correlation_numeric": "Numeric correlation",
                "famd_variance_explained": "FAMD variance",
                "famd_correlation_circle": "FAMD corr circle",
                "famd_sample_projection": "FAMD projection",
                "famd_clustering_report": "FAMD clustering report",
                "famd_clusters_kmeans": "FAMD K-Means",
                "famd_clusters_dbscan": "FAMD DBSCAN",
            }
            if name in title_map:
                return self._middle_truncate(title_map[name], 26)
            for prefix, label in (
                ("eda_", "EDA"),
                ("famd_contributions_", "FAMD contribution"),
                ("famd_correlation_circle_", "FAMD corr circle"),
                ("famd_sample_projection_", "FAMD projection"),
                ("fairness_dashboard_", "Fairness"),
                ("subgroup_dashboard_", "Subgroup"),
                ("robustness_dashboard_", "Robustness"),
            ):
                if name.startswith(prefix):
                    suffix = name.removeprefix(prefix).replace("_", " ")
                    return self._middle_truncate(f"{label} {suffix}", 26)
            return self._middle_truncate(Path(path).stem.replace("_", " ").title(), 26)

        def _rank_html_artifacts(self, paths: list[Path]) -> list[Path]:
            group_order = {
                "REPORT": 0,
                "EDA": 1,
                "MODEL": 2,
                "GAM": 3,
                "FAIRNESS": 4,
                "SUBGROUP": 5,
                "ROBUST": 6,
                "FAMD": 7,
                "HTML": 8,
            }
            title_order = {
                "final_report.html": 0,
                "model_evidence_metrics.html": 1,
                "model_feature_importance_safe.html": 2,
                "eda_class_imbalance.html": 10,
                "eda_missing_values.html": 11,
                "eda_numeric_distributions.html": 12,
                "eda_categorical_distributions.html": 13,
                "eda_suicidal_thoughts.html": 14,
                "eda_correlation_numeric.html": 15,
                "model_comparison.html": 20,
                "calibration_curves.html": 21,
                "decision_curves.html": 22,
                "gam_feature_effects.html": 30,
                "famd_clustering_report.html": 70,
                "famd_variance_explained.html": 71,
                "famd_sample_projection.html": 72,
                "famd_correlation_circle.html": 73,
            }

            def mtime(path: Path) -> float:
                try:
                    return path.stat().st_mtime
                except OSError:
                    return 0.0

            return sorted(
                paths,
                key=lambda path: (
                    group_order.get(self._html_artifact_group(path), 99),
                    title_order.get(path.name.lower(), 50),
                    -mtime(path),
                    path.name.lower(),
                ),
            )

        def _middle_truncate(self, value: Any, limit: int = 42) -> str:
            rendered = str(value)
            if len(rendered) <= limit:
                return rendered
            if limit <= 7:
                return rendered[:limit]
            head = (limit - 3) // 2
            tail = limit - 3 - head
            return f"{rendered[:head]}...{rendered[-tail:]}"

        def _path_brief(self, value: Any, limit: int = 44) -> str:
            if not value or value == Select.NULL:
                return "not selected"
            path = Path(str(value))
            if not path.name:
                return self._format_signal(value, limit)
            parent = path.parent.name or str(path.parent)
            return self._middle_truncate(f"{parent}/{path.name}", limit)

        def _build_artifact_hint(self) -> Panel:
            palette = self._palette()
            rows = [
                ("html", self._path_brief(self.query_one("#html_pick", Select).value)),
                ("json", self._path_brief(self.query_one("#history_pick", Select).value)),
                ("log", self._path_brief(self.query_one("#log_pick", Select).value)),
            ]
            table = Table(box=box.SIMPLE, show_header=False, expand=True, pad_edge=False)
            table.add_column("type", style=f"bold {palette['meta_label']}", width=5, no_wrap=True)
            table.add_column("selected", style=palette["meta_value"], ratio=1)
            for key, value in rows:
                table.add_row(key, value)
            hint = Text("2 latest  3 html  4 json  6 log", style=palette["hotkeys"])
            return Panel(
                Group(table, hint),
                title="[bold]SELECTED[/]",
                border_style=palette["artifact_border"],
                box=box.ROUNDED,
            )

        def _build_prediction_hint(self) -> Panel:
            palette = self._palette()
            model_path = Path(self.query_one("#pred_model_path", Input).value.strip() or "models/best_depression_model.joblib")
            state = "sẵn sàng" if model_path.exists() else "chưa có"
            rows = [
                ("model", self._path_brief(model_path)),
                ("trạng thái", state),
                ("chạy", "P hoặc :predict"),
            ]
            table = Table(box=box.SIMPLE, show_header=False, expand=True, pad_edge=False)
            table.add_column("field", style=f"bold {palette['meta_label']}", width=10, no_wrap=True)
            table.add_column("value", style=palette["meta_value"], ratio=1)
            for key, value in rows:
                table.add_row(key, value)
            hint = Text(
                "Nếu chưa có model, bấm T để train/export ngay trong TUI.",
                style=palette["hotkeys"] if model_path.exists() else f"bold {palette['skull']}",
            )
            return Panel(
                Group(table, hint),
                title="[bold]DỰ ĐOÁN[/]",
                border_style="#FFB86B" if model_path.exists() else palette["skull"],
                box=box.ROUNDED,
            )

        def _workflow_rows(self) -> list[tuple[str, str]]:
            spec = self._selected_spec()
            export_html = self.query_one("#export_html", Checkbox).value
            if export_html and spec.supports_export_html:
                html_state = "on"
            elif spec.supports_export_html:
                html_state = "available"
            else:
                html_state = "n/a"
            rows = [
                ("label", self._format_signal(spec.label, 30)),
                ("family", spec.family),
                ("desc", self._format_signal(spec.description, 38)),
                ("variant", "enabled" if spec.supports_variant else "fixed"),
                ("budget", "enabled" if spec.supports_budget else "fixed"),
                ("html", html_state),
            ]
            return rows

        def _operator_rows(self) -> list[tuple[str, str]]:
            return [
                ("run", "1 / :run"),
                ("train", "T / :train-best"),
                ("predict", "P / :predict"),
                ("review", "4 JSON, 6 LOG"),
                ("html", "2 latest, 3 selected"),
                ("refresh", "F5 artifact lists"),
                ("scroll", "wheel, PgUp/PgDn"),
                ("command", ":set workflow report"),
            ]

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
            text.append("> nhấn T để tạo artifact model dự đoán từ dataset hiện tại\n", style=palette["accent_soft"])
            text.append("> điền form dự đoán bên trái rồi nhấn P để sàng lọc một hồ sơ\n", style="#FFB86B")
            text.append("> press r to rerun previous workflow\n", style=palette["delta_border"])
            text.append("> press : to open command palette\n", style="#E0B8FF")
            text.append("> press F5 to refresh html/json/log artifact lists\n", style=palette["hotkeys"])
            text.append("> use PgUp / PgDn to scroll control, output and intel lanes together\n", style=palette["hotkeys"])
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
                        "!! attention mode :: review leakage, subgroup and robustness signals",
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

        def _prediction_record(self) -> dict[str, Any]:
            return {
                "Gender": str(self.query_one("#pred_gender", Select).value or "Female"),
                "Age": self._prediction_number("#pred_age", "Tuổi"),
                "City": self.query_one("#pred_city", Input).value.strip() or "Unknown",
                "Academic Pressure": self._prediction_number("#pred_academic_pressure", "Áp lực học tập", is_select=True),
                "CGPA": self._prediction_number("#pred_cgpa", "CGPA"),
                "Study Satisfaction": self._prediction_number("#pred_study_satisfaction", "Mức hài lòng học tập", is_select=True),
                "Sleep Duration": str(self.query_one("#pred_sleep_duration", Select).value or "5-6 hours"),
                "Dietary Habits": str(self.query_one("#pred_dietary_habits", Select).value or "Moderate"),
                "Degree": self.query_one("#pred_degree", Input).value.strip() or "Unknown",
                "Have you ever had suicidal thoughts ?": str(self.query_one("#pred_suicidal", Select).value or "No"),
                "Work/Study Hours": self._prediction_number("#pred_work_study_hours", "Số giờ học/làm mỗi ngày"),
                "Financial Stress": self._prediction_number("#pred_financial_stress", "Áp lực tài chính", is_select=True),
                "Family History of Mental Illness": str(self.query_one("#pred_family_history", Select).value or "No"),
            }

        def _prediction_number(self, selector: str, label: str, *, is_select: bool = False) -> float:
            if is_select:
                raw_value = self.query_one(selector, Select).value
            else:
                raw_value = self.query_one(selector, Input).value
            text = str(raw_value or "").strip()
            if not text:
                raise ValueError(f"{label} là bắt buộc.")
            try:
                return float(text)
            except ValueError as exc:
                raise ValueError(f"{label} phải là số, giá trị hiện tại: {text}") from exc

        def _prediction_result_output(self, prediction: dict[str, Any], record: dict[str, Any], metadata: dict[str, Any]) -> Group:
            palette = self._palette()
            flagged = int(prediction.get("prediction", 0)) == 1
            verdict = "CÓ DẤU HIỆU NGUY CƠ TRẦM CẢM" if flagged else "CHƯA VƯỢT NGƯỠNG SÀNG LỌC"
            verdict_style = f"bold {palette['skull']}" if flagged else f"bold {palette['accent_soft']}"

            result_table = Table(box=box.SIMPLE_HEAVY, show_header=False, expand=True)
            result_table.add_column("chỉ số", style=f"bold {palette['meta_label']}", width=18)
            result_table.add_column("giá trị", style=palette["meta_value"])
            result_table.add_row("kết luận", verdict)
            result_table.add_row("xác suất", self._format_signal(prediction.get("probability"), 18))
            result_table.add_row("ngưỡng", self._format_signal(prediction.get("threshold"), 18))
            result_table.add_row("chính sách ngưỡng", str(prediction.get("threshold_policy")))
            result_table.add_row("model", str(prediction.get("model")))
            result_table.add_row("profile", str(prediction.get("profile")))

            display_labels = {
                "Gender": "Giới tính",
                "Age": "Tuổi",
                "City": "Thành phố",
                "Academic Pressure": "Áp lực học tập",
                "CGPA": "CGPA / điểm TB",
                "Study Satisfaction": "Hài lòng học tập",
                "Sleep Duration": "Thời lượng ngủ",
                "Dietary Habits": "Thói quen ăn uống",
                "Degree": "Bậc học / bằng cấp",
                "Have you ever had suicidal thoughts ?": "Từng có ý nghĩ tự tử",
                "Work/Study Hours": "Giờ học/làm mỗi ngày",
                "Financial Stress": "Áp lực tài chính",
                "Family History of Mental Illness": "Tiền sử gia đình",
            }
            display_values = {
                "Gender": {"Female": "Nữ", "Male": "Nam"},
                "Sleep Duration": {
                    "Less than 5 hours": "Dưới 5 giờ",
                    "5-6 hours": "5-6 giờ",
                    "7-8 hours": "7-8 giờ",
                    "More than 8 hours": "Trên 8 giờ",
                },
                "Dietary Habits": {"Healthy": "Lành mạnh", "Moderate": "Trung bình", "Unhealthy": "Không lành mạnh"},
                "Have you ever had suicidal thoughts ?": {"No": "Không", "Yes": "Có"},
                "Family History of Mental Illness": {"No": "Không", "Yes": "Có"},
            }
            input_table = Table(title="Hồ Sơ Đã Nhập", box=box.SIMPLE, show_header=False, expand=True)
            input_table.add_column("trường", style=f"bold {palette['meta_label']}", width=34)
            input_table.add_column("giá trị", style=palette["meta_value"])
            selected_columns = metadata.get("selected_columns", [])
            for key, value in record.items():
                marker = "*" if key in selected_columns else ""
                display_value = display_values.get(key, {}).get(str(value), value)
                input_table.add_row(f"{display_labels.get(key, key)}{marker}", self._format_signal(display_value, 40))

            note = Text()
            note.append(verdict + "\n", style=verdict_style)
            note.append(
                "Đây là kết quả hỗ trợ sàng lọc, không phải chẩn đoán lâm sàng. "
                "Nếu có dấu hiệu nguy cơ, nên trao đổi với chuyên gia sức khỏe tâm thần.",
                style=palette["hotkeys"],
            )
            return self._wrap_with_scanlines(
                Panel(
                    Group(
                        Text(self._grid_line(96, 6), style=palette["grid"]),
                        Text(self._radar_sweep(96, 3), style="#FFB86B"),
                        note,
                        result_table,
                        input_table,
                        Text("* = cột được artifact triển khai sử dụng", style=palette["hotkeys"]),
                        Text(self._grid_line(96, 8), style=palette["grid"]),
                    ),
                    title="[bold]DỰ ĐOÁN SÀNG LỌC TRẦM CẢM[/]",
                    border_style="#FFB86B",
                    box=box.DOUBLE,
                )
            )

        def _prediction_error_output(self, message: str) -> Group:
            palette = self._palette()
            return self._wrap_with_scanlines(
                Panel(
                    Group(
                        Text("Không thể chạy dự đoán.\n", style=f"bold {palette['skull']}"),
                        Text(message, style=palette["meta_value"]),
                        Text(
                            "\nBấm T trong TUI để tạo artifact, hoặc chạy: robot train-best --dataset Student_Depression_Dataset.csv --preset research --budget auto",
                            style=palette["hotkeys"],
                        ),
                    ),
                    title="[bold]LỖI DỰ ĐOÁN[/]",
                    border_style=palette["skull"],
                    box=box.HEAVY,
                )
            )

        def _train_best_running_output(
            self,
            *,
            dataset_path: str,
            preset: str,
            training_budget_mode: str,
            model_path: str,
        ) -> Group:
            palette = self._palette()
            command = (
                "robot train-best "
                f"--dataset {dataset_path} "
                f"--preset {preset} "
                f"--budget {training_budget_mode} "
                f"--model-path {model_path}"
            )
            text = Text()
            text.append("Đang train và xuất best model artifact...\n\n", style=f"bold {palette['accent_soft']}")
            text.append("TUI đang chạy tương đương lệnh:\n", style=palette["meta_label"])
            text.append(command, style=palette["meta_value"])
            text.append(
                "\n\nSau khi hoàn tất, ô model trong form dự đoán sẽ trỏ tới artifact vừa tạo.",
                style=palette["hotkeys"],
            )
            return self._wrap_with_scanlines(
                Panel(
                    Group(
                        Text(self._grid_line(96, 2), style=palette["grid"]),
                        Text(self._radar_sweep(96, 4), style=palette["accent"]),
                        text,
                        Text(self._grid_line(96, 6), style=palette["grid"]),
                    ),
                    title="[bold]TRAIN / EXPORT MODEL[/]",
                    border_style=palette["accent"],
                    box=box.DOUBLE,
                )
            )

        def _train_best_result_output(self, result: Any) -> Group:
            palette = self._palette()
            selection = result.selection if isinstance(result.selection, dict) else {}
            holdout = selection.get("holdout", {}) if isinstance(selection.get("holdout"), dict) else {}
            metadata = result.metadata if isinstance(result.metadata, dict) else {}

            table = Table(box=box.SIMPLE_HEAVY, show_header=False, expand=True)
            table.add_column("trường", style=f"bold {palette['meta_label']}", width=22)
            table.add_column("giá trị", style=palette["meta_value"])
            table.add_row("model tốt nhất", str(selection.get("model", "n/a")))
            table.add_row("profile", str(selection.get("profile", "n/a")))
            table.add_row("ROC-AUC", self._format_signal(holdout.get("roc_auc"), 18))
            table.add_row("PR-AUC", self._format_signal(holdout.get("pr_auc"), 18))
            table.add_row("F1", self._format_signal(holdout.get("f1"), 18))
            table.add_row("ngưỡng", self._format_signal(selection.get("threshold"), 18))
            table.add_row("số cột dùng", str(len(metadata.get("selected_columns", []))))

            artifact_table = Table(title="Artifact Đã Tạo", box=box.SIMPLE, show_header=False, expand=True)
            artifact_table.add_column("loại", style=f"bold {palette['meta_label']}", width=18)
            artifact_table.add_column("đường dẫn", style=palette["meta_value"])
            artifact_table.add_row("model", str(result.model_path))
            artifact_table.add_row("metadata", str(result.metadata_path))
            if result.selection_path:
                artifact_table.add_row("best selection", str(result.selection_path))
            if result.comparison_path:
                artifact_table.add_row("comparison", str(result.comparison_path))

            note = Text()
            note.append("Model triển khai đã sẵn sàng cho form dự đoán.\n", style=f"bold {palette['accent_soft']}")
            note.append(
                "Bấm P để chạy sàng lọc trên hồ sơ đang nhập. Kết quả vẫn chỉ là hỗ trợ sàng lọc, không phải chẩn đoán lâm sàng.",
                style=palette["hotkeys"],
            )
            return self._wrap_with_scanlines(
                Panel(
                    Group(
                        Text(self._grid_line(96, 10), style=palette["grid"]),
                        Text(self._radar_sweep(96, 12), style=palette["accent"]),
                        note,
                        table,
                        artifact_table,
                        Text(self._grid_line(96, 14), style=palette["grid"]),
                    ),
                    title="[bold]BEST MODEL ARTIFACT READY[/]",
                    border_style=palette["accent"],
                    box=box.DOUBLE,
                )
            )

        def _train_best_error_output(self, message: str) -> Group:
            palette = self._palette()
            return self._wrap_with_scanlines(
                Panel(
                    Group(
                        Text("Không thể tạo model artifact.\n", style=f"bold {palette['skull']}"),
                        Text(message, style=palette["meta_value"]),
                        Text(
                            "\nKiểm tra lại dataset path, preset/budget và dependency của model trước khi chạy lại.",
                            style=palette["hotkeys"],
                        ),
                    ),
                    title="[bold]LỖI TRAIN / EXPORT MODEL[/]",
                    border_style=palette["skull"],
                    box=box.HEAVY,
                )
            )

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

        def _sorted_model_items(self, report: Any) -> list[tuple[str, Any]]:
            models = self._dict_field(report, "models")

            def sort_key(item: tuple[str, Any]) -> float:
                holdout = self._dict_field(item[1], "holdout")
                try:
                    return float(holdout.get("roc_auc"))
                except (TypeError, ValueError):
                    return -1.0

            return sorted(models.items(), key=sort_key, reverse=True)

        def _run_briefing_panel(self, report: Any) -> Panel:
            palette = self._palette()
            config = self._dict_field(report, "config")
            dataset = self._dict_field(report, "dataset")
            split = self._dict_field(report, "split")
            warnings_list = self._field(report, "warnings", [])
            models = list(self._dict_field(report, "models").keys())
            profile = str(config.get("profile", "safe"))
            profile_label = "A / safe" if profile == "safe" else "B / full"
            rust_engine = config.get("rust_engine")
            if isinstance(rust_engine, dict):
                rust_text = "available" if rust_engine.get("available") else self._format_signal(rust_engine.get("error"), 34)
            else:
                rust_text = self._format_signal(rust_engine, 34)
            return self._telemetry_frame(
                "RUN BRIEFING",
                [
                    ("profile", profile_label),
                    ("preset", self._format_signal(config.get("preset"))),
                    ("budget", self._format_signal(config.get("training_budget_mode", "default"))),
                    ("dataset", f"{self._format_signal(dataset.get('rows'))} rows, {self._format_signal(dataset.get('cols'))} cols"),
                    ("split", f"train={self._format_signal(split.get('train_size'))}, test={self._format_signal(split.get('test_size'))}"),
                    ("positive_rate", self._format_signal(split.get("test_positive_rate"))),
                    ("models", ", ".join(models) if models else "none"),
                    ("warnings", str(len(warnings_list)) if isinstance(warnings_list, list) else "0"),
                    ("rust_gam", rust_text),
                ],
                palette["signal_border"],
                skull=self._danger_workflow(),
            )

        def _model_note(self, model_name: str, model: Any) -> str:
            metadata = self._dict_field(model, "metadata")
            if model_name == "dummy":
                return f"baseline/{metadata.get('strategy', 'prior')}"
            if model_name == "logistic":
                return "balanced LR"
            if model_name == "gam":
                engine = metadata.get("engine", "pygam")
                splines = metadata.get("n_splines")
                return f"{engine}, splines={splines}"
            if model_name == "catboost":
                return "GPU" if metadata.get("used_gpu") else "CPU"
            return self._format_signal(metadata.get("evaluation"), 22)

        def _model_scoreboard_panel(self, report: Any) -> Panel:
            palette = self._palette()
            table = Table(
                box=box.SIMPLE_HEAVY,
                border_style=palette["signal_border"],
                show_header=True,
                header_style=f"bold {palette['meta_label']}",
                expand=True,
            )
            table.add_column("#", justify="right", width=3, no_wrap=True)
            table.add_column("model", style=f"bold {palette['meta_value']}", width=12, no_wrap=True)
            table.add_column("roc_auc", justify="right", no_wrap=True)
            table.add_column("pr_auc", justify="right", no_wrap=True)
            table.add_column("f1", justify="right", no_wrap=True)
            table.add_column("recall", justify="right", no_wrap=True)
            table.add_column("brier", justify="right", no_wrap=True)
            table.add_column("threshold", justify="right", no_wrap=True)
            table.add_column("note", ratio=1)

            model_items = self._sorted_model_items(report)
            if not model_items:
                table.add_row("-", "none", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "no model output")
            for rank, (model_name, model) in enumerate(model_items, start=1):
                holdout = self._dict_field(model, "holdout")
                thresholds = self._dict_field(model, "thresholds")
                best_f1 = thresholds.get("best_f1") if isinstance(thresholds.get("best_f1"), dict) else {}
                table.add_row(
                    str(rank),
                    model_name,
                    self._format_signal(holdout.get("roc_auc"), 10),
                    self._format_signal(holdout.get("pr_auc"), 10),
                    self._format_signal(holdout.get("f1"), 10),
                    self._format_signal(holdout.get("recall"), 10),
                    self._format_signal(holdout.get("brier_score"), 10),
                    self._format_signal(best_f1.get("threshold"), 10),
                    self._model_note(model_name, model),
                )

            caption = Text("Sorted by holdout ROC-AUC. Use this table as the opening slide for model results.", style=palette["hotkeys"])
            return Panel(
                Group(caption, table),
                title="[bold]MODEL SCOREBOARD[/]",
                border_style=palette["signal_border"],
                box=box.DOUBLE,
            )

        def _feature_evidence(self, model: Any) -> str:
            rows = self._field(model, "feature_importance", [])
            if not isinstance(rows, list) or not rows:
                return "baseline/no feature ranking"
            snippets: list[str] = []
            for item in rows[:3]:
                if not isinstance(item, dict):
                    continue
                feature = self._format_signal(item.get("feature"), 24)
                if "odds_ratio" in item:
                    snippets.append(f"{feature} OR={self._format_signal(item.get('odds_ratio'), 8)}")
                elif "importance" in item:
                    snippets.append(f"{feature} imp={self._format_signal(item.get('importance'), 8)}")
                elif "variance_importance" in item:
                    snippets.append(f"{feature} var={self._format_signal(item.get('variance_importance'), 8)}")
                else:
                    snippets.append(feature)
            return "; ".join(snippets) if snippets else "feature ranking unavailable"

        def _model_evidence_panel(self, report: Any) -> Panel:
            palette = self._palette()
            table = Table(
                box=box.SIMPLE_HEAVY,
                border_style=palette["delta_border"],
                show_header=True,
                header_style=f"bold {palette['meta_label']}",
                expand=True,
            )
            table.add_column("model", style=f"bold {palette['meta_value']}", width=12, no_wrap=True)
            table.add_column("talk track", ratio=1)
            for model_name, model in self._sorted_model_items(report):
                holdout = self._dict_field(model, "holdout")
                thresholds = self._dict_field(model, "thresholds")
                best_f1 = thresholds.get("best_f1") if isinstance(thresholds.get("best_f1"), dict) else {}
                talk_track = (
                    f"ROC-AUC {self._format_signal(holdout.get('roc_auc'), 8)}, "
                    f"F1 {self._format_signal(holdout.get('f1'), 8)}, "
                    f"best-F1 threshold {self._format_signal(best_f1.get('threshold'), 8)}. "
                    f"Evidence: {self._feature_evidence(model)}"
                )
                table.add_row(model_name, talk_track)
            if not self._sorted_model_items(report):
                table.add_row("none", "No model evidence available.")
            return Panel(
                table,
                title="[bold]KEY EVIDENCE[/]",
                border_style=palette["delta_border"],
                box=box.DOUBLE,
            )

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
                        "RUN SUMMARY",
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
                        "CONFIG",
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
                        "DIAGNOSTICS",
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
                        "PROFILE SUMMARY",
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
                        "COLUMN SETS",
                        [
                            ("warnings", str(len(warnings_list))),
                            ("safe_columns", self._format_signal(safe_columns)),
                            ("full_columns", self._format_signal(full_columns)),
                        ],
                        palette["signal_border"],
                    ),
                    self._telemetry_frame(
                        "PROFILE DELTA",
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
                return self._wrap_with_scanlines(
                    self._run_briefing_panel(report),
                    self._model_scoreboard_panel(report),
                    self._model_evidence_panel(report),
                    self._telemetry_frame("MODEL SPREAD", self._risk_spread_rows(report), palette["delta_border"]),
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
                        "COMPARE SUMMARY",
                        [
                            ("preset", self._format_signal(self._field(report, "preset", "quick"))),
                            ("rows", self._format_signal(dataset.get("rows"))),
                            ("train_size", self._format_signal(split.get("train_size"))),
                            ("test_size", self._format_signal(split.get("test_size"))),
                        ],
                        palette["signal_border"],
                        skull=self._danger_workflow(),
                    ),
                    self._telemetry_frame("A/B SCOREBOARD", signal_rows or [("status", "no compare signal")], palette["signal_border"]),
                    self._telemetry_frame("A/B DELTA", self._compare_delta_rows(report), palette["delta_border"]),
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
                    "RUN SUMMARY",
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
                    "KEY SIGNALS",
                    [(str(key), self._format_signal(value)) for key, value in list(result.summary.items())[:6]]
                    or [("status", "legacy transcript captured")],
                    palette["signal_border"],
                ),
                self._telemetry_frame(
                    "DIAGNOSTICS",
                    [
                        ("artifacts", str(len(result.artifacts))),
                        ("transcript_chars", str(len(result.transcript))),
                        ("transcript_lines", str(len(result.transcript.splitlines()))),
                    ],
                    palette["delta_border"],
                ),
                self._artifact_panel(result),
                self._channel_frame(
                    "DETAIL TRACE",
                    text or Text("No transcript captured.", style=palette["meta_value"]),
                    palette["output_border"],
                ),
                *json_renderables,
            )

        def _artifact_panel(self, result: WorkflowResult) -> Panel:
            palette = self._palette()
            table = Table(
                box=box.SIMPLE_HEAVY,
                border_style=palette["artifact_border"],
                show_header=True,
                header_style=f"bold {palette['meta_label']}",
                expand=True,
            )
            table.add_column("type", style="bold #67D5FF", width=12, no_wrap=True)
            table.add_column("name", style="#D8FEE3", ratio=1)
            table.add_column("folder", style=palette["hotkeys"], ratio=1)
            for key, value in sorted(result.artifacts.items()):
                path = Path(str(value))
                if path.name:
                    table.add_row(str(key), path.name, str(path.parent))
                else:
                    table.add_row(str(key), self._format_signal(value, 48), "")
            for html_path in result.html_artifacts:
                path = Path(html_path)
                table.add_row("html", path.name, str(path.parent))
            if not result.artifacts and not result.html_artifacts:
                table.add_row("status", "no artifact channel data yet", "")
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
            ranked_html = self._rank_html_artifacts(html_files)
            self._last_html = [str(path) for path in ranked_html[:120]]
            select = self.query_one("#html_pick", Select)
            options = [("select html artifact", "")]
            options.extend((self._artifact_select_label(path, "html"), path) for path in self._last_html)
            select.set_options(options)
            if self._last_html:
                select.value = self._last_html[0]

        def _refresh_history_options(self) -> None:
            history_files = scan_json_artifacts()
            self._last_json = [str(path) for path in history_files[:25]]
            select = self.query_one("#history_pick", Select)
            options = [("select json history", "")]
            options.extend((self._artifact_select_label(path, "json"), path) for path in self._last_json)
            select.set_options(options)
            if self._last_json:
                select.value = self._last_json[0]

        def _refresh_log_options(self) -> None:
            log_files = scan_log_artifacts()
            self._last_logs = [str(path) for path in log_files[:25]]
            select = self.query_one("#log_pick", Select)
            options = [("select console log", "")]
            options.extend((self._artifact_select_label(path, "log"), path) for path in self._last_logs)
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

        def action_train_best_model(self) -> None:
            dataset_path = self.query_one("#dataset", Input).value.strip()
            if not dataset_path:
                self._last_action = "train-best:error"
                self._set_output(self._train_best_error_output("Dataset path đang trống."))
                self._set_status("error", "dataset path đang trống")
                return
            if not Path(dataset_path).exists():
                self._last_action = "train-best:error"
                self._set_output(self._train_best_error_output(f"Không tìm thấy dataset: {dataset_path}"))
                self._set_status("error", f"không tìm thấy dataset: {dataset_path}")
                return

            preset = str(self.query_one("#preset", Select).value or "research")
            training_budget_mode = str(self.query_one("#budget", Select).value or "auto")
            model_path = self.query_one("#pred_model_path", Input).value.strip() or "models/best_depression_model.joblib"

            self._last_action = "train-best"
            self._set_status("running", "đang train/export best model")
            self._set_output(
                self._train_best_running_output(
                    dataset_path=dataset_path,
                    preset=preset,
                    training_budget_mode=training_budget_mode,
                    model_path=model_path,
                )
            )
            self._train_best_worker(dataset_path, preset, training_budget_mode, model_path)

        def action_predict_current(self) -> None:
            model_path = Path(self.query_one("#pred_model_path", Input).value.strip() or "models/best_depression_model.joblib")
            try:
                from src.app import load_deployment

                record = self._prediction_record()
                deployment = load_deployment(model_path)
                prediction = deployment.predict(record)[0]
            except Exception as exc:
                self._last_action = "predict:error"
                self._set_output(self._prediction_error_output(str(exc)))
                self._set_status("error", f"dự đoán lỗi: {exc}")
                return

            self._last_action = "predict"
            self._set_output(self._prediction_result_output(prediction, record, deployment.metadata))
            label = "có nguy cơ" if int(prediction.get("prediction", 0)) == 1 else "chưa vượt ngưỡng"
            probability = prediction.get("probability")
            self._set_status("success", f"{label}; xác suất={probability}")

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
            elif event.button.id == "train_best_btn":
                self.action_train_best_model()
            elif event.button.id == "predict_btn":
                self.action_predict_current()

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
            if normalized in {"train-best", "train best", "export-model", "export model", "model", "t"}:
                self.action_train_best_model()
                return
            if normalized in {"predict", "risk", "screen", "p"}:
                self.action_predict_current()
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
            if event.select.id in {"html_pick", "history_pick", "log_pick"}:
                self.query_one("#artifact_hint", Static).update(self._build_artifact_hint())
                return
            if event.select.id not in {"workflow", "variant", "preset", "budget"}:
                return
            self._render_static_panels()
            if event.select.id == "workflow" and self._last_result is not None and self._last_action != "idle":
                self._set_output(self._build_result_output(self._last_result))
            elif event.select.id == "workflow":
                self._set_output(self._build_idle_output())

        def on_key(self, event: events.Key) -> None:
            sidebar_scroll = self.query_one("#sidebar_scroll", VerticalScroll)
            output_scroll = self.query_one("#output_scroll", VerticalScroll)
            intel_scroll = self.query_one("#intel_scroll", VerticalScroll)

            if event.key == "pageup":
                sidebar_scroll.scroll_page_up(animate=False)
                output_scroll.scroll_page_up(animate=False)
                intel_scroll.scroll_page_up(animate=False)
                event.stop()
            elif event.key == "pagedown":
                sidebar_scroll.scroll_page_down(animate=False)
                output_scroll.scroll_page_down(animate=False)
                intel_scroll.scroll_page_down(animate=False)
                event.stop()
            elif event.key == "home":
                sidebar_scroll.scroll_home(animate=False)
                output_scroll.scroll_home(animate=False)
                intel_scroll.scroll_home(animate=False)
                event.stop()
            elif event.key == "end":
                sidebar_scroll.scroll_end(animate=False)
                output_scroll.scroll_end(animate=False)
                intel_scroll.scroll_end(animate=False)
                event.stop()

        @work(thread=True, exclusive=True)
        def _train_best_worker(
            self,
            dataset_path: str,
            preset: str,
            training_budget_mode: str,
            model_path: str,
        ) -> None:
            try:
                from src.app import train_best_deployment

                result = train_best_deployment(
                    dataset_path=dataset_path,
                    preset=preset,
                    training_budget_mode=training_budget_mode,
                    output_dir="results/app",
                    model_path=model_path,
                    threshold_policy="screening",
                )
                self.call_from_thread(self._train_best_done, result)
            except Exception as exc:  # pragma: no cover - interactive path
                self.call_from_thread(self._train_best_failed, exc)

        def _train_best_done(self, result: Any) -> None:
            self.query_one("#pred_model_path", Input).value = str(result.model_path)
            self._json_cache.clear()
            self._refresh_history_options()
            self._refresh_log_options()
            self._last_action = "train-best:done"
            self._set_output(self._train_best_result_output(result))
            self._set_status("success", f"model artifact ready: {Path(result.model_path).name}")

        def _train_best_failed(self, exc: Exception) -> None:
            self._last_action = "train-best:error"
            self._set_output(self._train_best_error_output(str(exc)))
            self._set_status("error", f"train/export model lỗi: {exc}")

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
