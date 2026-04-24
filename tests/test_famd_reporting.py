from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import ReportGenerator
from src.ml_models.famd import FAMDAnalyzer


class FAMDOutputTest(unittest.TestCase):
    def setUp(self) -> None:
        fixture_root = Path("results/test-fixtures")
        fixture_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(dir=fixture_root))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_adjacent_component_plots_and_clustering_outputs(self) -> None:
        analyzer = FAMDAnalyzer(random_state=42)
        coords = pd.DataFrame(
            {
                "F1": [-2.0, -1.8, -1.5, 1.4, 1.7, 2.0, 0.1, -0.1],
                "F2": [0.2, 0.1, 0.3, -0.1, -0.3, -0.2, 1.8, 1.6],
                "F3": [0.0, 0.2, -0.1, 0.1, -0.2, 0.0, -1.5, -1.6],
                "F4": [1.0, 0.9, 1.1, -0.9, -1.0, -1.1, 0.0, 0.1],
                "Depression": [0, 0, 0, 1, 1, 1, 1, 0],
            }
        )
        correlations = {
            "Age": {
                "type": "numeric",
                "correlations": [0.8, 0.1, 0.0, 0.3],
                "cos2": [0.64, 0.01, 0.0, 0.09],
                "contrib": [0.64, 0.01, 0.0, 0.09],
            },
            "City": {
                "type": "categorical",
                "correlations": [0.1, 0.7, 0.2, 0.0],
                "cos2": [0.01, 0.49, 0.04, 0.0],
                "contrib": [0.01, 0.49, 0.04, 0.0],
            },
        }
        analyzer.results = {
            "n_components": 4,
            "n_samples": len(coords),
            "n_numeric": 1,
            "n_categorical": 1,
            "eigenvalues": np.array([4.0, 3.0, 2.0, 1.0]),
            "explained_variance_ratio": [0.4, 0.3, 0.2, 0.1],
            "cumulative_variance": [0.4, 0.7, 0.9, 1.0],
            "coordinates": coords,
            "correlations": correlations,
            "top_contributions": {
                f"F{i + 1}": [
                    {
                        "variable": name,
                        "type": data["type"],
                        "contribution": data["contrib"][i],
                        "cos2": data["cos2"][i],
                    }
                    for name, data in correlations.items()
                ]
                for i in range(4)
            },
            "numeric_cols": ["Age"],
            "categorical_cols": ["City"],
        }

        saved = analyzer.save_all_plots(str(self.temp_dir), max_components=4)
        clustering = analyzer.save_clustering_outputs(str(self.temp_dir), n_dims=4)

        self.assertIn("sample_projection_F1_F2", saved)
        self.assertIn("sample_projection_F2_F3", saved)
        self.assertIn("sample_projection_F3_F4", saved)
        self.assertIn("contributions_F4", saved)
        self.assertTrue(Path(clustering["clustering_json"]).exists())
        self.assertTrue(Path(clustering["clustering_report"]).exists())


class FinalReportGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        fixture_root = Path("results/test-fixtures")
        fixture_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(dir=fixture_root))
        app_dir = self.temp_dir / "app"
        app_dir.mkdir()
        compare = {
            "preset": "research",
            "dataset": {"rows": 100, "cols": 18},
            "split": {"test_size": 20},
            "summary": {
                "dummy": {"safe_roc_auc": 0.5, "full_roc_auc": 0.5, "roc_auc_delta_full_minus_safe": 0.0, "safe_f1": 0.6, "full_f1": 0.6},
                "logistic": {"safe_roc_auc": 0.82, "full_roc_auc": 0.85, "roc_auc_delta_full_minus_safe": 0.03, "safe_f1": 0.78, "full_f1": 0.8},
                "gam": {"safe_roc_auc": 0.6, "full_roc_auc": 0.62, "roc_auc_delta_full_minus_safe": 0.02, "safe_f1": 0.62, "full_f1": 0.63},
                "catboost": {"safe_roc_auc": 0.9, "full_roc_auc": 0.92, "roc_auc_delta_full_minus_safe": 0.02, "safe_f1": 0.84, "full_f1": 0.85},
            },
            "profiles": {
                "safe": {"models": self._models()},
                "full": {"models": self._models(full=True)},
            },
            "artifacts": {},
        }
        (app_dir / "compare_profiles_research.json").write_text(
            json.dumps(compare, ensure_ascii=False),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_report_selects_catboost_from_safe_profile(self) -> None:
        generator = ReportGenerator(results_dir=str(self.temp_dir))
        generator.load_all_results()
        html_path = generator.generate_html_report(
            output_path=str(self.temp_dir / "final_report.html")
        )

        selection = json.loads((self.temp_dir / "best_model_selection.json").read_text(encoding="utf-8"))
        self.assertEqual(selection["profile"], "safe")
        self.assertEqual(selection["model"], "catboost")
        self.assertTrue(Path(html_path).exists())

    def _models(self, full: bool = False) -> dict:
        bump = 0.02 if full else 0.0
        base = {
            "dummy": (0.5, 0.58, 0.6, 0.24),
            "logistic": (0.82 + bump, 0.86 + bump, 0.78 + bump, 0.14),
            "gam": (0.6 + bump, 0.62 + bump, 0.62 + bump, 0.3),
            "catboost": (0.9 + bump, 0.93 + bump, 0.84 + bump, 0.12),
        }
        models = {}
        for name, (auc, pr, f1, brier) in base.items():
            models[name] = {
                "holdout": {
                    "roc_auc": auc,
                    "pr_auc": pr,
                    "f1": f1,
                    "recall": f1,
                    "precision": f1,
                    "brier_score": brier,
                    "confusion_matrix": {"tn": 8, "fp": 2, "fn": 1, "tp": 9},
                },
                "thresholds": {
                    "best_f1": {"threshold": 0.4, "f1": f1, "recall": f1},
                    "screening": {"threshold": 0.2, "recall": 0.9, "flagged_pct": 60.0},
                },
                "fairness": [],
                "feature_importance": [{"feature": "Age", "importance": 1.0}],
                "metadata": {},
            }
        return models


if __name__ == "__main__":
    unittest.main()
