from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli.entrypoint import _request_from_args
from src.cli.workflows import (
    WorkflowRequest,
    analyze_console_log,
    describe_json_artifact,
    describe_log_artifact,
    execute_workflow,
    latest_json_artifact,
    latest_log_artifact,
    latest_html_artifact,
    load_console_log_result,
    load_history_result,
    open_html_artifact,
    scan_html_artifacts,
    scan_json_artifacts,
    scan_log_artifacts,
)
from src.training_budget import resolve_training_budget


class WorkflowHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        fixture_root = Path("results/test-fixtures")
        fixture_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(dir=fixture_root))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_scan_html_artifacts_returns_newest_first(self) -> None:
        older = self.temp_dir / "older.html"
        newer = self.temp_dir / "newer.html"
        older.write_text("<html>older</html>", encoding="utf-8")
        newer.write_text("<html>newer</html>", encoding="utf-8")
        os.utime(older, (older.stat().st_atime, older.stat().st_mtime - 10))

        scanned = scan_html_artifacts((self.temp_dir,))

        self.assertEqual(scanned[0].name, "newer.html")
        self.assertEqual(scanned[1].name, "older.html")

    def test_latest_html_artifact_prefers_existing_candidates(self) -> None:
        html_path = self.temp_dir / "report.html"
        html_path.write_text("<html>report</html>", encoding="utf-8")

        picked = latest_html_artifact([str(html_path), str(self.temp_dir / "missing.html")])

        self.assertEqual(Path(picked), html_path.resolve())

    def test_open_html_artifact_prefers_windows_startfile(self) -> None:
        html_path = self.temp_dir / "report.html"
        html_path.write_text("<html>report</html>", encoding="utf-8")

        with (
            patch("src.cli.workflows.os.startfile", create=True) as mocked_startfile,
            patch("src.cli.workflows.webbrowser.open", return_value=True) as mocked_open,
        ):
            opened = open_html_artifact(html_path)

        if os.name == "nt":
            mocked_startfile.assert_called_once_with(str(html_path.resolve()))
            mocked_open.assert_not_called()
        else:
            mocked_open.assert_called_once()
        self.assertEqual(opened, html_path.resolve())

    def test_open_html_artifact_falls_back_to_browser_when_startfile_fails(self) -> None:
        html_path = self.temp_dir / "report.html"
        html_path.write_text("<html>report</html>", encoding="utf-8")

        with (
            patch("src.cli.workflows.os.startfile", side_effect=OSError("blocked"), create=True) as mocked_startfile,
            patch("src.cli.workflows.webbrowser.open", return_value=True) as mocked_open,
        ):
            opened = open_html_artifact(html_path)

        if os.name == "nt":
            mocked_startfile.assert_called_once_with(str(html_path.resolve()))
            mocked_open.assert_called_once()
        else:
            mocked_startfile.assert_not_called()
            mocked_open.assert_called_once()
        self.assertEqual(opened, html_path.resolve())

    def test_scan_json_artifacts_returns_newest_first(self) -> None:
        older = self.temp_dir / "older.json"
        newer = self.temp_dir / "newer.json"
        older.write_text("{}", encoding="utf-8")
        newer.write_text("{}", encoding="utf-8")
        os.utime(older, (older.stat().st_atime, older.stat().st_mtime - 10))

        scanned = scan_json_artifacts((self.temp_dir,), exclude_parts=())

        self.assertEqual(scanned[0].name, "newer.json")
        self.assertEqual(scanned[1].name, "older.json")

    def test_latest_json_artifact_prefers_existing_candidates(self) -> None:
        json_path = self.temp_dir / "run_safe_quick.json"
        json_path.write_text("{}", encoding="utf-8")

        picked = latest_json_artifact([str(json_path), str(self.temp_dir / "missing.json")])

        self.assertEqual(Path(picked), json_path.resolve())

    def test_scan_log_artifacts_returns_newest_first(self) -> None:
        older = self.temp_dir / "older.log"
        newer = self.temp_dir / "newer.log"
        older.write_text("older", encoding="utf-8")
        newer.write_text("newer", encoding="utf-8")
        os.utime(older, (older.stat().st_atime, older.stat().st_mtime - 10))

        scanned = scan_log_artifacts((self.temp_dir,), exclude_parts=())

        self.assertEqual(scanned[0].name, "newer.log")
        self.assertEqual(scanned[1].name, "older.log")

    def test_latest_log_artifact_prefers_existing_candidates(self) -> None:
        log_path = self.temp_dir / "eda.log"
        log_path.write_text("log", encoding="utf-8")

        picked = latest_log_artifact([str(log_path), str(self.temp_dir / "missing.log")])

        self.assertEqual(Path(picked), log_path.resolve())

    def test_load_history_result_reads_run_report_json(self) -> None:
        html_path = self.temp_dir / "report.html"
        html_path.write_text("<html>history</html>", encoding="utf-8")
        console_log = self.temp_dir / "console.log"
        console_log.write_text("console history", encoding="utf-8")
        json_path = self.temp_dir / "run_safe_quick.json"
        json_path.write_text(
            json.dumps(
                {
                    "config": {
                        "profile": "safe",
                        "preset": "quick",
                        "training_budget_mode": "auto",
                        "resolved_training_params": {"logistic": {"max_iter": 900}},
                        "models": ["logistic"],
                    },
                    "dataset": {"path": "sample.csv", "rows": 72, "cols": 18},
                    "split": {"train_size": 50, "test_size": 22, "test_positive_rate": 0.5},
                    "models": {
                        "logistic": {
                            "holdout": {"roc_auc": 0.81, "f1": 0.77},
                            "oof": {"roc_auc": 0.8},
                            "metadata": {"engine": "python"},
                            "feature_importance": [],
                            "artifacts": {"report_html": str(html_path)},
                        }
                    },
                    "timings": {"fit": 1.2},
                    "warnings": [],
                    "artifacts": {"run_json": str(json_path), "report_html": str(html_path), "console_log": str(console_log)},
                }
            ),
            encoding="utf-8",
        )

        result = load_history_result(json_path)

        self.assertEqual(result.workflow_id, "run")
        self.assertIsInstance(result.payload, dict)
        self.assertEqual(result.summary["profile"], "safe")
        self.assertIn(str(html_path.resolve()), result.html_artifacts)
        self.assertEqual(result.artifacts["history_json"], str(json_path.resolve()))
        self.assertEqual(result.artifacts["console_log"], str(console_log))

    def test_describe_json_artifact_includes_workflow_and_filename(self) -> None:
        json_path = self.temp_dir / "compare_profiles_quick.json"
        json_path.write_text("{}", encoding="utf-8")

        label = describe_json_artifact(json_path)

        self.assertIn("compare", label)
        self.assertIn("compare_profiles_quick.json", label)

    def test_load_console_log_result_reads_metadata_and_body(self) -> None:
        log_path = self.temp_dir / "20260424_091911_eda_A_quick_default.log"
        log_path.write_text(
            "\n".join(
                [
                    "workflow: eda",
                    "family: legacy",
                    "dataset: Student_Depression_Dataset.csv",
                    "variant: A / safe / conservative",
                    "preset: quick",
                    "training_budget_mode: default",
                    "timestamp: 2026-04-24T09:19:11",
                    "",
                    "=== console result ===",
                    "EDA line 1",
                    "EDA line 2",
                ]
            ),
            encoding="utf-8",
        )

        result = load_console_log_result(log_path)

        self.assertEqual(result.family, "log")
        self.assertEqual(result.workflow_id, "eda")
        self.assertIn("EDA line 1", result.transcript)
        self.assertEqual(result.summary["dataset"], "Student_Depression_Dataset.csv")
        self.assertEqual(result.artifacts["console_log"], str(log_path.resolve()))
        self.assertIn("assessment_rows", result.summary)
        self.assertIn("benchmark_rows", result.summary)

    def test_describe_log_artifact_includes_workflow_and_filename(self) -> None:
        log_path = self.temp_dir / "20260424_091911_eda_A_quick_default.log"
        log_path.write_text("workflow: eda\n\n=== console result ===\nbody", encoding="utf-8")

        label = describe_log_artifact(log_path)

        self.assertIn("eda", label)
        self.assertIn("20260424_091911_eda_A_quick_default.log", label)

    def test_analyze_console_log_builds_eda_benchmark(self) -> None:
        transcript = "\n".join(
            [
                "Shape: 27,901 rows × 18 cols",
                "Có trầm cảm (1): 16,336 (58.55%)",
                "⚠️  [Quality Gate] Tỷ lệ 'Suicidal thoughts = Yes' = 63.3% — cảnh báo leakage.",
                "📊 [Quality Gate] 1 cột có tổng 3 giá trị thiếu",
                "🔍 [Quality Gate] 2 cột có category hiếm (<1%)",
                "6 visualizations saved to: results\\visualizations",
                "Exploratory Data Analysis completed in 1.10s",
                "Data Review completed in 0.03s",
            ]
        )

        analysis = analyze_console_log("eda", transcript, html_artifacts=["a.html", "b.html"])

        self.assertIn(("kết luận", "EDA đã hoàn tất"), analysis["assessment_rows"])
        self.assertIn(("số dòng", "27,901"), analysis["benchmark_rows"])
        self.assertIn(("số HTML", "6"), analysis["benchmark_rows"])

    def test_analyze_console_log_builds_profile_summary(self) -> None:
        transcript = "\n".join(
            [
                "{'rows': 27901, 'cols': 18, 'target_positive_rate': 58.55, 'selected_columns_safe': ['Age', 'CGPA'], 'selected_columns_full': ['Age', 'CGPA', 'Have you ever had suicidal thoughts ?']}",
                "Warnings:",
                "- leakage risk from suicidal thoughts",
            ]
        )

        analysis = analyze_console_log("profile", transcript, html_artifacts=["profile.html"])

        self.assertIn(("kết luận", "Hồ sơ dữ liệu đã được dựng lại từ log"), analysis["assessment_rows"])
        self.assertIn(("số cột", "18"), analysis["benchmark_rows"])
        self.assertIn(("HTML", "1"), analysis["benchmark_rows"])

    def test_analyze_console_log_builds_compare_summary(self) -> None:
        transcript = (
            "{'logistic': {'safe_roc_auc': 0.8123, 'full_roc_auc': 0.8465, "
            "'roc_auc_delta_full_minus_safe': 0.0342, 'safe_f1': 0.701, 'full_f1': 0.744, "
            "'f1_delta_full_minus_safe': 0.043}, "
            "'catboost': {'safe_roc_auc': 0.834, 'full_roc_auc': 0.841, "
            "'roc_auc_delta_full_minus_safe': 0.007, 'safe_f1': 0.721, 'full_f1': 0.733, "
            "'f1_delta_full_minus_safe': 0.012}}"
        )

        analysis = analyze_console_log("compare", transcript, html_artifacts=["compare.html"])

        self.assertIn(("kết luận", "So sánh A/B trên cùng split đã hoàn tất"), analysis["assessment_rows"])
        self.assertIn(("số mô hình", "2"), analysis["benchmark_rows"])
        self.assertIn(("HTML", "1"), analysis["benchmark_rows"])

    def test_analyze_console_log_builds_advanced_summary(self) -> None:
        transcript = "\n".join(
            [
                "Analyzing fairness for logistic...",
                "Analyzing fairness for catboost...",
                "results/fairness_logistic.json",
                "results/fairness_catboost.json",
                "Fairness analysis completed in 2.50s",
                "⚠ warning: disparate impact below threshold",
            ]
        )

        analysis = analyze_console_log("fairness", transcript, html_artifacts=["fairness_dashboard_logistic.html"])

        self.assertIn(("kết luận", "Phân tích fairness đã hoàn tất"), analysis["assessment_rows"])
        self.assertIn(("số mô hình", "2"), analysis["benchmark_rows"])
        self.assertIn(("JSON", "2"), analysis["benchmark_rows"])

    def test_main_import_handles_stream_without_buffer(self) -> None:
        code = (
            "import io, sys;"
            "C=type('C',(io.StringIO,),{});"
            "sys.stdout=C();"
            "sys.stderr=C();"
            "import main;"
            "sys.__stdout__.write('ok\\n')"
        )

        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("ok", completed.stdout)

    def test_request_from_args_maps_task_command(self) -> None:
        args = argparse.Namespace(
            command="task",
            workflow="analysis",
            dataset="Student_Depression_Dataset.csv",
            variant="B",
            preset="research",
            export_html=True,
            output_dir="results/app",
            console_only=False,
            budget="auto",
        )

        request = _request_from_args(args)

        self.assertEqual(request.workflow_id, "analysis")
        self.assertEqual(request.variant, "B")
        self.assertEqual(request.training_budget_mode, "auto")

    def test_execute_legacy_workflow_captures_transcript(self) -> None:
        class FakeMainModule:
            @staticmethod
            def main(**kwargs):
                print(f"legacy run {kwargs['dataset_path']} budget={kwargs['training_budget_mode']}")

        with patch("src.cli.workflows.importlib.import_module", return_value=FakeMainModule):
            result = execute_workflow(
                WorkflowRequest(
                    workflow_id="report",
                    dataset_path="sample.csv",
                    training_budget_mode="auto",
                    output_dir=str(self.temp_dir),
                )
            )

        self.assertEqual(result.family, "legacy")
        self.assertIn("legacy run sample.csv budget=auto", result.transcript)
        console_log = Path(result.artifacts["console_log"])
        self.assertTrue(console_log.exists())
        self.assertIn("legacy run sample.csv budget=auto", console_log.read_text(encoding="utf-8"))


class TrainingBudgetResolverTest(unittest.TestCase):
    def test_resolve_training_budget_auto_changes_modern_defaults(self) -> None:
        resolved = resolve_training_budget(mode="auto", family="modern", preset="research", train_rows=1800)

        self.assertGreaterEqual(resolved["logistic"]["max_iter"], 1300)
        self.assertGreaterEqual(resolved["catboost"]["iterations"], 320)
        self.assertGreaterEqual(resolved["gam"]["n_splines"], 12)


if __name__ == "__main__":
    unittest.main()
