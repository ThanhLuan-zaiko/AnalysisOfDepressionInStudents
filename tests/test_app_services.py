from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import polars as pl

from src.app import ArtifactPolicy, RunConfig, RunPreset, RunProfile, compare_profiles, load_dataset, run_pipeline


def build_sample_frame(n_rows: int = 72) -> pl.DataFrame:
    rows = []
    for idx in range(n_rows):
        positive = 1 if idx % 4 in (1, 2, 3) else 0
        suicidal = "Yes" if idx % 5 != 0 else "No"
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
                "Have you ever had suicidal thoughts ?": suicidal,
                "Work/Study Hours": 3 + (idx % 6),
                "Financial Stress": 1 + (idx % 4) + positive,
                "Family History of Mental Illness": "Yes" if idx % 3 == 0 else "No",
                "Depression": positive,
            }
        )

    frame = pl.DataFrame(rows)
    return frame.with_columns(
        pl.when(pl.col("id") == 3)
        .then(None)
        .otherwise(pl.col("Financial Stress"))
        .alias("Financial Stress")
    )


class AppServicesTest(unittest.TestCase):
    def setUp(self) -> None:
        fixture_root = Path("results/test-fixtures")
        fixture_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(dir=fixture_root))
        self.dataset_path = self.temp_dir / "sample_dataset.csv"
        build_sample_frame().write_csv(self.dataset_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_dataset_uses_cache_on_second_read(self) -> None:
        first = load_dataset(self.dataset_path)
        second = load_dataset(self.dataset_path)

        self.assertFalse(first.loaded_from_cache)
        self.assertTrue(second.loaded_from_cache)
        self.assertIsNotNone(second.cache_path)

    def test_run_pipeline_safe_profile_excludes_sensitive_feature(self) -> None:
        bundle = load_dataset(self.dataset_path)
        config = RunConfig(
            profile=RunProfile.SAFE,
            preset=RunPreset.QUICK,
            artifact_policy=ArtifactPolicy.CONSOLE_ONLY,
            test_size=0.25,
            cv_splits=3,
            models=("logistic",),
        )
        report = run_pipeline(
            bundle=bundle,
            profile=RunProfile.SAFE,
            preset=RunPreset.QUICK,
            artifact_policy=ArtifactPolicy.CONSOLE_ONLY,
            config=config,
        )

        self.assertNotIn("Have you ever had suicidal thoughts ?", report.config["selected_columns"])
        self.assertIn("logistic", report.models)
        self.assertIn("roc_auc", report.models["logistic"].oof)
        self.assertIn("roc_auc", report.models["logistic"].holdout)
        self.assertEqual(report.config["training_budget_mode"], "default")
        self.assertIn("resolved_training_params", report.config)

    def test_run_pipeline_full_profile_keeps_sensitive_feature(self) -> None:
        bundle = load_dataset(self.dataset_path)
        config = RunConfig(
            profile=RunProfile.FULL,
            preset=RunPreset.QUICK,
            artifact_policy=ArtifactPolicy.CONSOLE_ONLY,
            test_size=0.25,
            cv_splits=3,
            models=("logistic",),
        )
        report = run_pipeline(
            bundle=bundle,
            profile=RunProfile.FULL,
            preset=RunPreset.QUICK,
            artifact_policy=ArtifactPolicy.CONSOLE_ONLY,
            config=config,
        )

        self.assertIn("Have you ever had suicidal thoughts ?", report.config["selected_columns"])
        self.assertGreaterEqual(report.split["test_size"], 1)

    def test_run_pipeline_auto_budget_records_resolved_training_params(self) -> None:
        bundle = load_dataset(self.dataset_path)
        config = RunConfig(
            profile=RunProfile.SAFE,
            preset=RunPreset.QUICK,
            artifact_policy=ArtifactPolicy.CONSOLE_ONLY,
            test_size=0.25,
            cv_splits=3,
            models=("logistic", "catboost"),
            training_budget_mode="auto",
        )
        report = run_pipeline(
            bundle=bundle,
            profile=RunProfile.SAFE,
            preset=RunPreset.QUICK,
            artifact_policy=ArtifactPolicy.CONSOLE_ONLY,
            config=config,
        )

        self.assertEqual(report.config["training_budget_mode"], "auto")
        self.assertGreater(report.config["resolved_training_params"]["logistic"]["max_iter"], 0)
        self.assertGreater(report.config["resolved_training_params"]["catboost"]["iterations"], 0)

    def test_compare_profiles_returns_same_split(self) -> None:
        bundle = load_dataset(self.dataset_path)
        comparison = compare_profiles(
            bundle=bundle,
            preset=RunPreset.QUICK,
            artifact_policy=ArtifactPolicy.CONSOLE_ONLY,
        )

        self.assertIn("safe", comparison.profiles)
        self.assertIn("full", comparison.profiles)
        self.assertEqual(
            comparison.profiles["safe"].split["test_size"],
            comparison.profiles["full"].split["test_size"],
        )

    def test_run_pipeline_research_supports_gam(self) -> None:
        bundle = load_dataset(self.dataset_path)
        config = RunConfig(
            profile=RunProfile.SAFE,
            preset=RunPreset.RESEARCH,
            artifact_policy=ArtifactPolicy.CONSOLE_ONLY,
            test_size=0.25,
            cv_splits=3,
            models=("gam",),
        )

        try:
            report = run_pipeline(
                bundle=bundle,
                profile=RunProfile.SAFE,
                preset=RunPreset.RESEARCH,
                artifact_policy=ArtifactPolicy.CONSOLE_ONLY,
                config=config,
            )
        except ImportError as exc:
            self.skipTest(f"GAM dependencies unavailable: {exc}")

        self.assertIn("gam", report.models)
        self.assertIn("roc_auc", report.models["gam"].oof)
        self.assertIn("roc_auc", report.models["gam"].holdout)


if __name__ == "__main__":
    unittest.main()
