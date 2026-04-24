from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli.entrypoint import _request_from_args
from src.cli.workflows import (
    WorkflowRequest,
    describe_json_artifact,
    execute_workflow,
    latest_json_artifact,
    latest_html_artifact,
    load_history_result,
    open_html_artifact,
    scan_html_artifacts,
    scan_json_artifacts,
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

    def test_open_html_artifact_uses_browser(self) -> None:
        html_path = self.temp_dir / "report.html"
        html_path.write_text("<html>report</html>", encoding="utf-8")

        with patch("src.cli.workflows.webbrowser.open", return_value=True) as mocked_open:
            opened = open_html_artifact(html_path)

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

    def test_load_history_result_reads_run_report_json(self) -> None:
        html_path = self.temp_dir / "report.html"
        html_path.write_text("<html>history</html>", encoding="utf-8")
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
                    "artifacts": {"run_json": str(json_path), "report_html": str(html_path)},
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

    def test_describe_json_artifact_includes_workflow_and_filename(self) -> None:
        json_path = self.temp_dir / "compare_profiles_quick.json"
        json_path.write_text("{}", encoding="utf-8")

        label = describe_json_artifact(json_path)

        self.assertIn("compare", label)
        self.assertIn("compare_profiles_quick.json", label)

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
                )
            )

        self.assertEqual(result.family, "legacy")
        self.assertIn("legacy run sample.csv budget=auto", result.transcript)


class TrainingBudgetResolverTest(unittest.TestCase):
    def test_resolve_training_budget_auto_changes_modern_defaults(self) -> None:
        resolved = resolve_training_budget(mode="auto", family="modern", preset="research", train_rows=1800)

        self.assertGreaterEqual(resolved["logistic"]["max_iter"], 1300)
        self.assertGreaterEqual(resolved["catboost"]["iterations"], 320)
        self.assertGreaterEqual(resolved["gam"]["n_splines"], 12)


if __name__ == "__main__":
    unittest.main()
