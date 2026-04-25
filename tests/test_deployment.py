from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import polars as pl

from src.app import DatasetBundle, RunPreset
from src.app.deployment import (
    feature_columns_for_profile,
    fit_final_deployment_model,
    load_deployment,
    save_deployment,
    select_best_deployable_model,
)


def build_sample_frame(n_rows: int = 80) -> pl.DataFrame:
    rows = []
    for idx in range(n_rows):
        positive = 1 if idx % 4 in (1, 2, 3) else 0
        rows.append(
            {
                "id": idx + 1,
                "Gender": "Male" if idx % 2 == 0 else "Female",
                "Age": 18 + (idx % 10),
                "City": f"City-{idx % 4}",
                "Profession": "Student",
                "Academic Pressure": 2 + (idx % 3) + positive,
                "Work Pressure": 0,
                "CGPA": 6.2 + (idx % 5) * 0.4,
                "Study Satisfaction": 4 - (idx % 3),
                "Job Satisfaction": 0,
                "Sleep Duration": ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"][idx % 4],
                "Dietary Habits": ["Healthy", "Moderate", "Unhealthy"][idx % 3],
                "Degree": ["Bachelor", "Master", "PhD"][idx % 3],
                "Have you ever had suicidal thoughts ?": "Yes" if idx % 5 != 0 else "No",
                "Work/Study Hours": 3 + (idx % 6),
                "Financial Stress": 1 + (idx % 4) + positive,
                "Family History of Mental Illness": "Yes" if idx % 3 == 0 else "No",
                "Depression": positive,
            }
        )
    return pl.DataFrame(rows)


class DeploymentSelectionTest(unittest.TestCase):
    def test_select_best_deployable_model_across_profiles(self) -> None:
        comparison = {
            "profiles": {
                "safe": {
                    "config": {"selected_columns": ["Age"]},
                    "models": {
                        "catboost": {
                            "holdout": {"roc_auc": 0.84, "pr_auc": 0.80, "f1": 0.72},
                            "thresholds": {"screening": {"threshold": 0.31}},
                        }
                    },
                },
                "full": {
                    "config": {"selected_columns": ["Age", "Have you ever had suicidal thoughts ?"]},
                    "models": {
                        "gam": {
                            "holdout": {"roc_auc": 0.86, "pr_auc": 0.79, "f1": 0.70},
                            "thresholds": {"screening": {"threshold": 0.27}},
                        }
                    },
                },
            }
        }

        selection = select_best_deployable_model(comparison)

        self.assertEqual(selection["model"], "gam")
        self.assertEqual(selection["profile"], "full")
        self.assertEqual(selection["threshold"], 0.27)
        self.assertEqual(selection["roc_auc"], 0.86)
        self.assertIn("profile_warning", selection)

    def test_select_best_deployable_model_tie_prefers_catboost(self) -> None:
        comparison = {
            "profiles": {
                "safe": {
                    "config": {"selected_columns": ["Age"]},
                    "models": {
                        "logistic": {
                            "holdout": {"roc_auc": 0.84, "pr_auc": 0.80, "f1": 0.72},
                            "thresholds": {"screening": {"threshold": 0.40}},
                        },
                        "catboost": {
                            "holdout": {"roc_auc": 0.84, "pr_auc": 0.80, "f1": 0.72},
                            "thresholds": {"screening": {"threshold": 0.35}},
                        },
                    },
                }
            }
        }

        selection = select_best_deployable_model(comparison)

        self.assertEqual(selection["model"], "catboost")
        self.assertEqual(selection["threshold"], 0.35)


class DeploymentArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        fixture_root = Path("results/test-fixtures")
        fixture_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(dir=fixture_root))
        self.frame = build_sample_frame()
        self.dataset_path = self.temp_dir / "sample_dataset.csv"
        self.frame.write_csv(self.dataset_path)
        self.bundle = DatasetBundle(
            path=self.dataset_path,
            frame=self.frame,
            cache_path=None,
            loaded_from_cache=False,
            load_seconds=0.0,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_load_and_predict_logistic_deployment(self) -> None:
        selected_columns = feature_columns_for_profile(self.frame, "safe")
        deployment = fit_final_deployment_model(
            self.bundle,
            {
                "profile": "safe",
                "model": "logistic",
                "selected_columns": selected_columns,
                "threshold": 0.4,
                "threshold_policy": "screening",
                "threshold_source": "screening",
                "holdout": {"roc_auc": 0.8, "pr_auc": 0.75, "f1": 0.7},
            },
            preset=RunPreset.QUICK,
            training_budget_mode="default",
        )
        model_path = self.temp_dir / "best_model.joblib"
        metadata_path = self.temp_dir / "best_model.json"
        save_deployment(deployment, model_path=model_path, metadata_path=metadata_path)

        loaded = load_deployment(model_path)
        record = self.frame.drop("Depression").head(1).to_pandas()
        predictions = loaded.predict(record)

        self.assertTrue(model_path.exists())
        self.assertTrue(metadata_path.exists())
        self.assertEqual(len(predictions), 1)
        self.assertIn("probability", predictions[0])
        self.assertIn("not a clinical diagnosis", predictions[0]["disclaimer"])

    def test_predict_missing_required_column_fails_clearly(self) -> None:
        selected_columns = feature_columns_for_profile(self.frame, "safe")
        deployment = fit_final_deployment_model(
            self.bundle,
            {
                "profile": "safe",
                "model": "logistic",
                "selected_columns": selected_columns,
                "threshold": 0.4,
                "threshold_policy": "screening",
                "threshold_source": "screening",
                "holdout": {"roc_auc": 0.8, "pr_auc": 0.75, "f1": 0.7},
            },
            preset=RunPreset.QUICK,
            training_budget_mode="default",
        )
        record = self.frame.drop(["Depression", selected_columns[0]]).head(1).to_pandas()

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            deployment.predict(record)


if __name__ == "__main__":
    unittest.main()
