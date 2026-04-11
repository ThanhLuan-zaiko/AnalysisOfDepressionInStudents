"""
Robustness Analysis Module
==========================
Bootstrap confidence intervals, noise injection, feature ablation,
cross-validation stability, adversarial testing, population shift simulation.

Output: HTML reports + JSON results
"""

import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from src.utils.helpers import save_json, Timer

import logging
logger = logging.getLogger(__name__)


class RobustnessAnalyzer:
    """
    Comprehensive robustness analysis for classification models.
    """

    def __init__(self, n_bootstrap: int = 1000, n_cv_folds: int = 5, random_state: int = 42):
        """
        Args:
            n_bootstrap: Number of bootstrap samples for CI estimation
            n_cv_folds: Number of CV folds for stability analysis
            random_state: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.n_cv_folds = n_cv_folds
        self.random_state = random_state
        self.results = {}

        np.random.seed(random_state)

    def analyze(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_proba: np.ndarray,
        y_pred: np.ndarray,
        feature_names: List[str],
        model_trainer: Callable,  # Function to retrain model
        model_name: str = "model",
    ) -> Dict:
        """
        Run comprehensive robustness analysis.

        Args:
            X: Feature matrix
            y: Ground truth labels
            y_proba: Predicted probabilities (baseline)
            y_pred: Predicted labels (baseline)
            feature_names: List of feature names
            model_trainer: Function that takes (X, y) and returns (model, y_proba, y_pred)
            model_name: Name of the model

        Returns:
            Dict with all robustness results.
        """
        timer = Timer("Robustness Analysis")
        logger.info(f"🛡️  Starting robustness analysis for {model_name}...")

        with timer:
            results = {
                "model": model_name,
                "baseline": {
                    "roc_auc": float(roc_auc_score(y, y_proba)),
                    "f1": float(f1_score(y, y_pred, zero_division=0)),
                    "n_samples": len(y),
                },
                "bootstrap_ci": self._bootstrap_ci(y, y_proba, y_pred),
                "cv_stability": self._cv_stability(X, y, model_trainer),
                "noise_injection": self._noise_injection(X, y, model_trainer, feature_names),
                "feature_ablation": self._feature_ablation(X, y, feature_names, model_trainer),
                "adversarial_label_flip": self._adversarial_label_flip(X, y, model_trainer),
            }

        self.results[model_name] = results
        logger.info(f"✅ Robustness analysis completed in {timer.elapsed:.2f}s")

        return results

    # ==========================================
    # BOOTSTRAP CONFIDENCE INTERVALS
    # ==========================================

    def _bootstrap_ci(
        self,
        y: np.ndarray,
        y_proba: np.ndarray,
        y_pred: np.ndarray,
        ci_level: float = 0.95,
    ) -> Dict:
        """
        Bootstrap confidence intervals for key metrics.
        """
        logger.info("  📊 Computing bootstrap confidence intervals...")

        n = len(y)
        alpha = 1 - ci_level
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100

        auc_scores = []
        f1_scores = []
        precision_scores = []
        recall_scores = []

        for i in range(self.n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            y_boot = y[indices]
            yp_boot = y_proba[indices]
            yd_boot = y_pred[indices]

            if len(np.unique(y_boot)) < 2:
                continue

            auc_scores.append(roc_auc_score(y_boot, yp_boot))
            f1_scores.append(f1_score(y_boot, yd_boot, zero_division=0))
            precision_scores.append(precision_score(y_boot, yd_boot, zero_division=0))
            recall_scores.append(recall_score(y_boot, yd_boot, zero_division=0))

        def compute_ci(scores):
            if not scores:
                return {"mean": 0, "std": 0, "lower": 0, "upper": 0, "ci_width": 0}
            return {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "lower": float(np.percentile(scores, lower_pct)),
                "upper": float(np.percentile(scores, upper_pct)),
                "ci_width": float(np.percentile(scores, upper_pct) - np.percentile(scores, lower_pct)),
            }

        return {
            "n_bootstrap": self.n_bootstrap,
            "ci_level": ci_level,
            "roc_auc": compute_ci(auc_scores),
            "f1": compute_ci(f1_scores),
            "precision": compute_ci(precision_scores),
            "recall": compute_ci(recall_scores),
        }

    # ==========================================
    # CROSS-VALIDATION STABILITY
    # ==========================================

    def _cv_stability(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_trainer: Callable,
    ) -> Dict:
        """
        Assess model stability across CV folds.
        """
        logger.info(f"  🔄 Running {self.n_cv_folds}-fold CV stability analysis...")

        skf = StratifiedKFold(
            n_splits=self.n_cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        fold_aucs = []
        fold_f1s = []
        fold_precisions = []
        fold_recalls = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                _, y_proba_fold, y_pred_fold = model_trainer(X_train, y_train, X_test, y_test)

                if len(np.unique(y_test)) < 2:
                    continue

                fold_aucs.append(roc_auc_score(y_test, y_proba_fold))
                fold_f1s.append(f1_score(y_test, y_pred_fold, zero_division=0))
                fold_precisions.append(precision_score(y_test, y_pred_fold, zero_division=0))
                fold_recalls.append(recall_score(y_test, y_pred_fold, zero_division=0))

            except Exception as e:
                logger.warning(f"    Fold {fold_idx + 1} failed: {e}")
                continue

        def summarize(scores):
            if not scores:
                return {"mean": 0, "std": 0, "min": 0, "max": 0, "range": 0}
            return {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "range": float(np.max(scores) - np.min(scores)),
                "scores": [float(s) for s in scores],
            }

        # Stability assessment
        auc_range = (np.max(fold_aucs) - np.min(fold_aucs)) if fold_aucs else 0
        f1_range = (np.max(fold_f1s) - np.min(fold_f1s)) if fold_f1s else 0

        return {
            "n_folds": self.n_cv_folds,
            "n_successful_folds": len(fold_aucs),
            "roc_auc": summarize(fold_aucs),
            "f1": summarize(fold_f1s),
            "precision": summarize(fold_precisions),
            "recall": summarize(fold_recalls),
            "stability_assessment": {
                "auc_range": auc_range,
                "f1_range": f1_range,
                "stable": auc_range < 0.05 and f1_range < 0.05,
                "interpretation": (
                    "✅ Stable (ΔAUC < 0.05, ΔF1 < 0.05)"
                    if auc_range < 0.05 and f1_range < 0.05
                    else f"⚠️  Moderate variance (ΔAUC={auc_range:.3f}, ΔF1={f1_range:.3f})"
                ),
            },
        }

    # ==========================================
    # NOISE INJECTION
    # ==========================================

    def _noise_injection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_trainer: Callable,
        feature_names: List[str],
        noise_levels: List[float] = None,
    ) -> Dict:
        """
        Test robustness to noise injection.
        Adds Gaussian noise at increasing levels to features.
        """
        logger.info("  🔊 Testing noise injection robustness...")

        if noise_levels is None:
            noise_levels = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

        baseline_auc = roc_auc_score(y, self._predict_proba_with_noise(X, y, X, model_trainer, 0))
        baseline_f1 = f1_score(y, self._predict_with_noise(X, y, X, model_trainer, 0), zero_division=0)

        results = []
        for noise_level in noise_levels:
            try:
                y_proba_noisy = self._predict_proba_with_noise(
                    X, y, X, model_trainer, noise_level
                )
                y_pred_noisy = (y_proba_noisy >= 0.5).astype(int)

                auc_noisy = roc_auc_score(y, y_proba_noisy)
                f1_noisy = f1_score(y, y_pred_noisy, zero_division=0)

                results.append({
                    "noise_level": noise_level,
                    "roc_auc": float(auc_noisy),
                    "f1": float(f1_noisy),
                    "auc_drop": float(baseline_auc - auc_noisy),
                    "f1_drop": float(baseline_f1 - f1_noisy),
                    "auc_relative_drop": float((baseline_auc - auc_noisy) / baseline_auc) if baseline_auc > 0 else 0,
                })

            except Exception as e:
                logger.warning(f"    Noise level {noise_level} failed: {e}")
                continue

        # Find breaking point (where AUC drops below 0.70)
        breaking_point = None
        for r in results:
            if r["roc_auc"] < 0.70:
                breaking_point = r["noise_level"]
                break

        return {
            "noise_levels_tested": noise_levels,
            "baseline_auc": float(baseline_auc),
            "baseline_f1": float(baseline_f1),
            "results": results,
            "breaking_point": breaking_point,
            "robustness_score": self._compute_robustness_score(results),
        }

    def _predict_proba_with_noise(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        model_trainer: Callable,
        noise_level: float,
    ) -> np.ndarray:
        """Train with noise and predict."""
        noise = np.random.normal(0, noise_level, X_train.shape)
        X_train_noisy = X_train + noise

        _, y_proba, _ = model_trainer(X_train_noisy, y_train, X_test, y_train)
        return y_proba

    def _predict_with_noise(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        model_trainer: Callable,
        noise_level: float,
    ) -> np.ndarray:
        """Train with noise and predict with threshold."""
        y_proba = self._predict_proba_with_noise(
            X_train, y_train, X_test, model_trainer, noise_level
        )
        return (y_proba >= 0.5).astype(int)

    @staticmethod
    def _compute_robustness_score(results: List[Dict]) -> Dict:
        """
        Compute overall robustness score.
        Higher = more robust.
        """
        if not results:
            return {"score": 0, "grade": "F", "interpretation": "No results"}

        # Average relative AUC drop
        avg_drop = np.mean([r["auc_relative_drop"] for r in results])

        # Grade
        if avg_drop < 0.05:
            grade = "A"
        elif avg_drop < 0.10:
            grade = "B"
        elif avg_drop < 0.20:
            grade = "C"
        elif avg_drop < 0.30:
            grade = "D"
        else:
            grade = "F"

        return {
            "score": float(1 - avg_drop),
            "avg_relative_drop": float(avg_drop),
            "grade": grade,
            "interpretation": (
                f"{'Highly robust' if grade in ['A', 'B'] else 'Moderately robust' if grade == 'C' else 'Not robust'} "
                f"(avg ΔAUC={avg_drop:.1%})"
            ),
        }

    # ==========================================
    # FEATURE ABLATION
    # ==========================================

    def _feature_ablation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_trainer: Callable,
    ) -> Dict:
        """
        Test impact of removing feature groups.
        Groups features by semantic meaning.
        """
        logger.info("  🔬 Running feature ablation analysis...")

        # Baseline
        _, y_proba_baseline, y_pred_baseline = model_trainer(X, y, X, y)
        baseline_auc = roc_auc_score(y, y_proba_baseline)
        baseline_f1 = f1_score(y, y_pred_baseline, zero_division=0)

        # Define feature groups (heuristic: by prefix)
        groups = self._group_features(feature_names)

        results = []
        for group_name, feature_indices in groups.items():
            try:
                # Remove features in this group
                keep_indices = [i for i in range(X.shape[1]) if i not in feature_indices]
                X_ablated = X[:, keep_indices]

                _, y_proba_ablated, y_pred_ablated = model_trainer(
                    X_ablated, y, X_ablated, y
                )

                auc_ablated = roc_auc_score(y, y_proba_ablated)
                f1_ablated = f1_score(y, y_pred_ablated, zero_division=0)

                results.append({
                    "group": group_name,
                    "n_features_removed": len(feature_indices),
                    "n_features_remaining": len(keep_indices),
                    "roc_auc": float(auc_ablated),
                    "f1": float(f1_ablated),
                    "auc_drop": float(baseline_auc - auc_ablated),
                    "f1_drop": float(baseline_f1 - f1_ablated),
                    "auc_relative_drop": float((baseline_auc - auc_ablated) / baseline_auc) if baseline_auc > 0 else 0,
                })

            except Exception as e:
                logger.warning(f"    Group '{group_name}' ablation failed: {e}")
                continue

        # Sort by AUC drop (descending)
        results.sort(key=lambda x: x["auc_drop"], reverse=True)

        return {
            "baseline_auc": float(baseline_auc),
            "baseline_f1": float(baseline_f1),
            "n_total_features": X.shape[1],
            "feature_groups": {k: len(v) for k, v in groups.items()},
            "ablation_results": results,
        }

    @staticmethod
    def _group_features(feature_names: List[str]) -> Dict[str, List[int]]:
        """Group features by semantic prefix."""
        groups = {}

        for idx, name in enumerate(feature_names):
            # Determine group by prefix
            if name.startswith("City_"):
                group = "City"
            elif name.startswith("Degree_"):
                group = "Degree"
            elif name.startswith("Gender_"):
                group = "Gender"
            elif name.startswith("Sleep Duration_"):
                group = "Sleep"
            elif name.startswith("Dietary Habits_"):
                group = "Dietary"
            elif name.startswith("Family History"):
                group = "Family_History"
            elif "Academic Pressure" in name or "Academic_Pressure" in name:
                group = "Academic"
            elif "Financial" in name or "financial" in name.lower():
                group = "Financial"
            elif "Work" in name or "Hours" in name:
                group = "Work_Study"
            elif name in ["Age", "CGPA", "Work/Study Hours", "Financial Stress", "Academic Pressure", "Study Satisfaction"]:
                group = "Numeric"
            else:
                group = "Other"

            if group not in groups:
                groups[group] = []
            groups[group].append(idx)

        return groups

    # ==========================================
    # ADVERSARIAL LABEL FLIP
    # ==========================================

    def _adversarial_label_flip(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_trainer: Callable,
        flip_ratios: List[float] = None,
    ) -> Dict:
        """
        Test robustness to label noise (adversarial label flipping).
        """
        logger.info("  ⚡ Testing adversarial label flip robustness...")

        if flip_ratios is None:
            flip_ratios = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]

        results = []
        n = len(y)

        for ratio in flip_ratios:
            n_flip = int(n * ratio)
            flip_indices = np.random.choice(n, size=n_flip, replace=False)

            y_flipped = y.copy()
            y_flipped[flip_indices] = 1 - y_flipped[flip_indices]

            try:
                _, y_proba_flipped, y_pred_flipped = model_trainer(
                    X, y_flipped, X, y
                )

                auc_flipped = roc_auc_score(y, y_proba_flipped)
                f1_flipped = f1_score(y, y_pred_flipped, zero_division=0)

                results.append({
                    "flip_ratio": ratio,
                    "n_flipped": n_flip,
                    "roc_auc": float(auc_flipped),
                    "f1": float(f1_flipped),
                    "auc_drop": float(results[0]["roc_auc"] - auc_flipped) if results else 0,
                })

            except Exception as e:
                logger.warning(f"    Flip ratio {ratio} failed: {e}")
                continue

        # Find breaking point
        breaking_point = None
        for r in results:
            if r["roc_auc"] < 0.70:
                breaking_point = r["flip_ratio"]
                break

        return {
            "flip_ratios_tested": flip_ratios,
            "results": results,
            "breaking_point": breaking_point,
        }

    # ==========================================
    # VISUALIZATION
    # ==========================================

    def plot_robustness_dashboard(
        self,
        results: Dict,
        output_path: str = "results/robustness_dashboard.html",
    ):
        """Create comprehensive robustness dashboard."""
        model_name = results.get("model", "model")

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Bootstrap Confidence Intervals",
                "CV Stability (Fold AUC)",
                "Noise Injection Robustness",
                "Feature Ablation Impact",
                "Adversarial Label Flip",
                "Robustness Summary",
            ),
            specs=[
                [{"type": "bar"}, {"type": "box"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "table"}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.10,
        )

        # 1. Bootstrap CI
        bootstrap = results.get("bootstrap_ci", {})
        if bootstrap:
            metrics = ["roc_auc", "f1", "precision", "recall"]
            means = []
            lowers = []
            uppers = []
            labels = []

            for metric in metrics:
                if metric in bootstrap:
                    data = bootstrap[metric]
                    means.append(data.get("mean", 0))
                    lowers.append(data.get("lower", 0))
                    uppers.append(data.get("upper", 0))
                    labels.append(metric.upper())

            fig.add_trace(
                go.Bar(
                    x=labels, y=means,
                    name="Mean",
                    marker_color="#3498db",
                    error_y=dict(
                        type="data",
                        array=[u - m for u, m in zip(uppers, means)],
                        arrayminus=[m - l for m, l in zip(means, lowers)],
                        visible=True,
                    ),
                ),
                row=1, col=1,
            )

        # 2. CV Stability
        cv = results.get("cv_stability", {})
        if cv and "roc_auc" in cv:
            auc_scores = cv["roc_auc"].get("scores", [])
            if auc_scores:
                fig.add_trace(
                    go.Box(
                        y=auc_scores,
                        name="CV AUC",
                        marker_color="#2ecc71",
                        boxmean=True,
                    ),
                    row=1, col=2,
                )

        # 3. Noise Injection
        noise = results.get("noise_injection", {})
        if noise and "results" in noise:
            noise_results = noise["results"]
            noise_levels = [r["noise_level"] for r in noise_results]
            auc_drops = [r["roc_auc"] for r in noise_results]
            f1_drops = [r["f1"] for r in noise_results]

            fig.add_trace(
                go.Scatter(
                    x=noise_levels, y=auc_drops,
                    mode="lines+markers",
                    name="AUC",
                    line=dict(color="#3498db"),
                    marker=dict(size=8),
                ),
                row=2, col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=noise_levels, y=f1_drops,
                    mode="lines+markers",
                    name="F1",
                    line=dict(color="#e74c3c"),
                    marker=dict(size=8),
                ),
                row=2, col=1,
            )

        # 4. Feature Ablation
        ablation = results.get("feature_ablation", {})
        if ablation and "ablation_results" in ablation:
            ablation_results = ablation["ablation_results"]
            groups = [r["group"] for r in ablation_results]
            auc_drops = [r["auc_drop"] for r in ablation_results]

            fig.add_trace(
                go.Bar(
                    x=groups, y=auc_drops,
                    name="AUC Drop",
                    marker_color="coral",
                ),
                row=2, col=2,
            )

        # 5. Adversarial Label Flip
        adversarial = results.get("adversarial_label_flip", {})
        if adversarial and "results" in adversarial:
            adv_results = adversarial["results"]
            flip_ratios = [r["flip_ratio"] for r in adv_results]
            adv_aucs = [r["roc_auc"] for r in adv_results]

            fig.add_trace(
                go.Scatter(
                    x=flip_ratios, y=adv_aucs,
                    mode="lines+markers",
                    name="AUC",
                    line=dict(color="#9b59b6"),
                    marker=dict(size=8),
                ),
                row=3, col=1,
            )

        # 6. Robustness Summary Table
        summary_data = []

        # Bootstrap CI
        if bootstrap and "roc_auc" in bootstrap:
            auc_ci = bootstrap["roc_auc"]
            summary_data.append([
                "Bootstrap CI (AUC)",
                f"{auc_ci.get('lower', 0):.3f} — {auc_ci.get('upper', 0):.3f}",
                f"±{auc_ci.get('std', 0):.3f}",
            ])

        # CV Stability
        if cv and "stability_assessment" in cv:
            stability = cv["stability_assessment"]
            summary_data.append([
                "CV Stability",
                stability.get("interpretation", "N/A"),
                f"Δ={cv['roc_auc'].get('range', 0):.3f}",
            ])

        # Noise Robustness
        if noise and "robustness_score" in noise:
            rs = noise["robustness_score"]
            summary_data.append([
                "Noise Robustness",
                rs.get("interpretation", "N/A"),
                f"Grade: {rs.get('grade', 'N/A')}",
            ])

        # Breaking points
        breaking_noise = noise.get("breaking_point", "N/A")
        breaking_adv = adversarial.get("breaking_point", None)
        if breaking_noise != "N/A" and breaking_noise is not None:
            summary_data.append([
                "Noise Breaking Point",
                f"σ={breaking_noise}",
                "AUC < 0.70",
            ])
        if breaking_adv is not None:
            summary_data.append([
                "Label Flip Breaking Point",
                f"{breaking_adv:.0%} labels",
                "AUC < 0.70",
            ])

        if summary_data:
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=["Metric", "Value", "Detail"],
                        fill_color="paleturquoise",
                        align="center",
                    ),
                    cells=dict(
                        values=list(zip(*summary_data)),
                        fill_color="lavender",
                        align="center",
                    ),
                ),
                row=3, col=2,
            )

        fig.update_layout(
            title=f"🛡️  Robustness Analysis Dashboard — {model_name}",
            height=1100,
            width=1200,
            showlegend=True,
            margin=dict(l=60, r=60, t=60, b=60),
        )

        fig.update_xaxes(title_text="Metric", row=1, col=1)
        fig.update_yaxes(title_text="Value ± CI", row=1, col=1)
        fig.update_xaxes(title_text="Noise Level (σ)", row=2, col=1)
        fig.update_yaxes(title_text="Score", row=2, col=1)
        fig.update_xaxes(title_text="Feature Group", row=2, col=2)
        fig.update_yaxes(title_text="AUC Drop", row=2, col=2)
        fig.update_xaxes(title_text="Label Flip Ratio", row=3, col=1)
        fig.update_yaxes(title_text="AUC", row=3, col=1)

        fig.write_html(output_path, include_plotlyjs=True, full_html=True)
        logger.info(f"✅ Saved robustness dashboard: {output_path}")
        return output_path

    # ==========================================
    # REPORT PRINTING
    # ==========================================

    def print_report(self, results: Dict):
        """Print formatted robustness report."""
        model = results.get("model", "model")
        print(f"\n{'='*70}")
        print(f" 🛡️  ROBUSTNESS ANALYSIS — {model.upper()}")
        print(f"{'='*70}\n")

        # Baseline
        baseline = results.get("baseline", {})
        print(f" 📊 Baseline Performance:")
        print(f"    ROC-AUC: {baseline.get('roc_auc', 0):.4f}")
        print(f"    F1 Score: {baseline.get('f1', 0):.4f}")
        print()

        # Bootstrap CI
        bootstrap = results.get("bootstrap_ci", {})
        if bootstrap:
            print(f" 📐 Bootstrap Confidence Intervals ({bootstrap.get('n_bootstrap', 0)} samples):")
            for metric in ["roc_auc", "f1", "precision", "recall"]:
                if metric in bootstrap:
                    data = bootstrap[metric]
                    print(f"    {metric.upper():12s}: {data['lower']:.3f} — {data['upper']:.3f} "
                          f"(μ={data['mean']:.3f}, σ={data['std']:.3f})")
            print()

        # CV Stability
        cv = results.get("cv_stability", {})
        if cv:
            print(f" 🔄 CV Stability ({cv.get('n_folds', 0)} folds):")
            stability = cv.get("stability_assessment", {})
            print(f"    {stability.get('interpretation', 'N/A')}")
            print(f"    AUC: μ={cv['roc_auc'].get('mean', 0):.3f} ± {cv['roc_auc'].get('std', 0):.3f} "
                  f"(range: {cv['roc_auc'].get('min', 0):.3f} — {cv['roc_auc'].get('max', 0):.3f})")
            print(f"    F1:  μ={cv['f1'].get('mean', 0):.3f} ± {cv['f1'].get('std', 0):.3f} "
                  f"(range: {cv['f1'].get('min', 0):.3f} — {cv['f1'].get('max', 0):.3f})")
            print()

        # Noise Injection
        noise = results.get("noise_injection", {})
        if noise:
            print(f" 🔊 Noise Injection Robustness:")
            rs = noise.get("robustness_score", {})
            print(f"    {rs.get('interpretation', 'N/A')}")
            if noise.get("breaking_point"):
                print(f"    ⚠️  Breaking point at σ={noise['breaking_point']}")
            print()

        # Feature Ablation
        ablation = results.get("feature_ablation", {})
        if ablation and "ablation_results" in ablation:
            print(f" 🔬 Feature Ablation (Top 5):")
            for r in ablation["ablation_results"][:5]:
                print(f"    {r['group']:20s}: ΔAUC={r['auc_drop']:.3f} "
                      f"({r['n_features_removed']} features removed)")
            print()

        # Adversarial Label Flip
        adversarial = results.get("adversarial_label_flip", {})
        if adversarial and "results" in adversarial:
            print(f" ⚡ Adversarial Label Flip:")
            for r in adversarial["results"]:
                print(f"    Flip {r['flip_ratio']:.0%}: AUC={r['roc_auc']:.3f} "
                      f"(Δ={r.get('auc_drop', 0):.3f})")
            breaking_adv = adversarial.get("breaking_point", None)
            if breaking_adv is not None:
                print(f"    ⚠️  Breaking point at {breaking_adv:.0%} label flip")
            print()
