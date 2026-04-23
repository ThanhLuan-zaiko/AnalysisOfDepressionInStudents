from __future__ import annotations

from pathlib import Path

from src.app import ArtifactPolicy, RunPreset, RunProfile, compare_profiles, load_dataset, profile_dataset, run_pipeline


def launch_tui(default_dataset: Path) -> None:
    from textual.app import App, ComposeResult
    from textual.containers import Vertical
    from textual.widgets import Button, Checkbox, Footer, Header, Input, Select, Static

    class DepressionTUI(App[None]):
        CSS = """
        Screen {
            background: #F6F4EF;
            color: #1E3A5F;
        }

        #root {
            padding: 1 2;
        }

        #banner {
            color: #0F766E;
            content-align: center middle;
            height: 8;
            border: round #D4A373;
            margin-bottom: 1;
        }

        Input, Select, Checkbox, Button {
            margin-bottom: 1;
        }

        #output {
            border: round #1E3A5F;
            padding: 1;
            min-height: 16;
        }
        """

        BINDINGS = [("q", "quit", "Quit")]

        def compose(self) -> ComposeResult:
            yield Header(show_clock=False)
            with Vertical(id="root"):
                yield Static(
                    "Sen Analytics\nHoldout-first depression analysis\nlotus tone for Vietnamese research workflows",
                    id="banner",
                )
                yield Input(value=str(default_dataset), placeholder="Dataset path", id="dataset")
                yield Select(
                    [(label.title(), label) for label in (RunProfile.SAFE.value, RunProfile.FULL.value)],
                    value=RunProfile.SAFE.value,
                    prompt="Profile",
                    id="profile",
                )
                yield Select(
                    [(label.title(), label) for label in (RunPreset.QUICK.value, RunPreset.RESEARCH.value)],
                    value=RunPreset.QUICK.value,
                    prompt="Preset",
                    id="preset",
                )
                yield Checkbox("Export HTML EDA", value=False, id="export_html")
                yield Button("Profile Dataset", id="profile_btn", variant="primary")
                yield Button("Run Pipeline", id="run_btn", variant="success")
                yield Button("Compare Profiles", id="compare_btn")
                yield Static("Ready.", id="output")
            yield Footer()

        def _output(self) -> Static:
            return self.query_one("#output", Static)

        def _dataset(self) -> str:
            return self.query_one("#dataset", Input).value

        def _profile(self) -> RunProfile:
            return RunProfile(self.query_one("#profile", Select).value)

        def _preset(self) -> RunPreset:
            return RunPreset(self.query_one("#preset", Select).value)

        def _export_html(self) -> bool:
            return self.query_one("#export_html", Checkbox).value

        def on_button_pressed(self, event: Button.Pressed) -> None:
            output = self._output()
            output.update("Running...")
            try:
                bundle = load_dataset(self._dataset())
                if event.button.id == "profile_btn":
                    report = profile_dataset(
                        bundle=bundle,
                        artifact_policy=ArtifactPolicy.FULL_EXPORT if self._export_html() else ArtifactPolicy.JSON,
                        export_html=self._export_html(),
                    )
                    output.update(
                        f"Rows: {report.summary['rows']}\n"
                        f"Cols: {report.summary['cols']}\n"
                        f"Positive rate: {report.summary['target_positive_rate']}\n"
                        f"Warnings:\n- " + "\n- ".join(report.warnings if report.warnings else ["None"])
                    )
                    return

                if event.button.id == "compare_btn":
                    comparison = compare_profiles(
                        bundle=bundle,
                        preset=self._preset(),
                        artifact_policy=ArtifactPolicy.JSON,
                    )
                    lines = ["Profile comparison:"]
                    for model_name, summary in comparison.summary.items():
                        lines.append(
                            f"{model_name}: safe={summary['safe_roc_auc']} full={summary['full_roc_auc']} "
                            f"delta={summary['roc_auc_delta_full_minus_safe']}"
                        )
                    output.update("\n".join(lines))
                    return

                report = run_pipeline(
                    bundle=bundle,
                    profile=self._profile(),
                    preset=self._preset(),
                    artifact_policy=ArtifactPolicy.JSON,
                )
                lines = [f"Profile={report.config['profile']} preset={report.config['preset']}"]
                for model_name, result in report.models.items():
                    lines.append(
                        f"{model_name}: holdout roc_auc={result.holdout.get('roc_auc')} "
                        f"f1={result.holdout.get('f1')}"
                    )
                if report.warnings:
                    lines.append("Warnings:")
                    lines.extend(f"- {warning}" for warning in report.warnings)
                output.update("\n".join(lines))
            except Exception as exc:  # pragma: no cover - interactive path
                output.update(f"Error: {exc}")

    DepressionTUI().run()
