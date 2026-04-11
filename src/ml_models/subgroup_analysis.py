"""
Subgroup Analysis Module
========================
Deep subgroup performance breakdown, error analysis, feature importance
differences across groups, calibration per subgroup, threshold optimization.

Output: HTML reports + JSON results
"""

import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, brier_score_loss,
    precision_recall_curve, average_precision_score,
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from src.utils.helpers import save_json, Timer

import logging
logger = logging.getLogger(__name__)


class SubgroupAnalyzer:
    """
    Comprehensive subgroup analysis for classification models.
    """

    def __init__(self):
        self.results = {}

    def analyze(
        self,
        df: pl.DataFrame,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        y_pred: np.ndarray,
        feature_names: List[str] = None,
        model_name: str = "model",
        threshold: float = 0.5,
    ) -> Dict:
        """
        Run comprehensive subgroup analysis.

        Args:
            df: Original DataFrame with demographic columns
            y_true: Ground truth labels
            y_proba: Predicted probabilities
            y_pred: Predicted labels
            feature_names: List of feature names (for importance analysis)
            model_name: Name of the model
            threshold: Decision threshold used

        Returns:
            Dict with all subgroup analysis results.
        """
        timer = Timer("Subgroup Analysis")
        logger.info(f"🔍 Starting subgroup analysis for {model_name}...")

        with timer:
            results = {
                "model": model_name,
                "threshold": threshold,
                "overall": self._overall_analysis(y_true, y_proba, y_pred),
                "subgroups": {},
                "error_analysis": self._error_analysis(df, y_true, y_pred, y_proba),
                "calibration_by_subgroup": {},
                "threshold_recommendations": {},
            }

            # Analyze by Gender
            if "Gender" in df.columns:
                results["subgroups"]["Gender"] = self._analyze_split(
                    df, "Gender", y_true, y_proba, y_pred
                )

            # Analyze by Age groups
            if "Age" in df.columns:
                results["subgroups"]["Age"] = self._analyze_age_groups(
                    df, y_true, y_proba, y_pred
                )

            # Analyze by Family History
            if "Family History of Mental Illness" in df.columns:
                results["subgroups"]["Family_History"] = self._analyze_split(
                    df, "Family History of Mental Illness", y_true, y_proba, y_pred
                )

            # Analyze by City (top cities)
            if "City" in df.columns:
                results["subgroups"]["City"] = self._analyze_top_cities(
                    df, y_true, y_proba, y_pred, top_n=10
                )

            # Analyze by Academic Pressure
            if "Academic Pressure" in df.columns:
                results["subgroups"]["Academic_Pressure"] = self._analyze_ordinal(
                    df, "Academic Pressure", y_true, y_proba, y_pred
                )

            # Calibration by subgroup
            for subgroup_type, subgroup_data in results["subgroups"].items():
                results["calibration_by_subgroup"][subgroup_type] = (
                    self._calibration_by_subgroup(
                        df, subgroup_type, y_true, y_proba
                    )
                )

            # Threshold recommendations per subgroup
            results["threshold_recommendations"] = self._threshold_recommendations(
                df, results["subgroups"], y_true, y_proba
            )

        self.results[model_name] = results
        logger.info(f"✅ Subgroup analysis completed in {timer.elapsed:.2f}s")

        return results

    # ==========================================
    # OVERALL ANALYSIS
    # ==========================================

    def _overall_analysis(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict:
        """Overall model analysis."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        return {
            "n_samples": len(y_true),
            "n_positive": int(y_true.sum()),
            "n_negative": int((~y_true.astype(bool)).sum()),
            "prevalence": float(y_true.mean()),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "pr_auc": float(average_precision_score(y_true, y_proba)),
            "accuracy": float(np.mean(y_true == y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
            "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
            "fnr": float(fn / (fn + tp)) if (fn + tp) > 0 else 0,
            "npv": float(tn / (tn + fn)) if (tn + fn) > 0 else 0,
            "brier_score": float(brier_score_loss(y_true, y_proba)),
            "confusion_matrix": {
                "tn": int(tn), "fp": int(fp),
                "fn": int(fn), "tp": int(tp),
            },
        }

    # ==========================================
    # SUBGROUP ANALYSIS HELPERS
    # ==========================================

    def _analyze_split(
        self,
        df: pl.DataFrame,
        column: str,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict:
        """Analyze a categorical/binary column."""
        values = df[column].to_numpy()
        unique_vals = np.unique(values)

        subgroups = {}
        for val in unique_vals:
            mask = values == val
            if mask.sum() < 30:
                continue

            y_sub = y_true[mask]
            yp_sub = y_proba[mask]
            yd_sub = y_pred[mask]

            subgroups[str(val)] = self._compute_subgroup_metrics(
                y_sub, yp_sub, yd_sub, int(mask.sum())
            )

        return subgroups

    def _analyze_age_groups(
        self,
        df: pl.DataFrame,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict:
        """Analyze by age groups."""
        ages = df["Age"].to_numpy()
        bins = [(18, 22), (23, 26), (27, 30), (31, 100)]
        labels = ["18-22", "23-26", "27-30", "31+"]

        subgroups = {}
        for label, (lo, hi) in zip(labels, bins):
            mask = (ages >= lo) & (ages <= hi)
            if mask.sum() < 30:
                continue

            y_sub = y_true[mask]
            yp_sub = y_proba[mask]
            yd_sub = y_pred[mask]

            subgroups[label] = self._compute_subgroup_metrics(
                y_sub, yp_sub, yd_sub, int(mask.sum()),
                extra={"age_range": f"{lo}-{hi}", "age_mean": float(ages[mask].mean())}
            )

        return subgroups

    def _analyze_top_cities(
        self,
        df: pl.DataFrame,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        y_pred: np.ndarray,
        top_n: int = 10,
    ) -> Dict:
        """Analyze top N cities by sample size."""
        city_counts = df["City"].value_counts().sort("count", descending=True)
        top_cities = city_counts["City"].head(top_n).to_numpy()

        cities = df["City"].to_numpy()
        subgroups = {}

        for city in top_cities:
            mask = cities == city
            if mask.sum() < 30:
                continue

            y_sub = y_true[mask]
            yp_sub = y_proba[mask]
            yd_sub = y_pred[mask]

            subgroups[str(city)] = self._compute_subgroup_metrics(
                y_sub, yp_sub, yd_sub, int(mask.sum())
            )

        # Also add "Other" for remaining cities
        other_mask = ~np.isin(cities, top_cities)
        if other_mask.sum() >= 30:
            y_sub = y_true[other_mask]
            yp_sub = y_proba[other_mask]
            yd_sub = y_pred[other_mask]
            subgroups["Other"] = self._compute_subgroup_metrics(
                y_sub, yp_sub, yd_sub, int(other_mask.sum())
            )

        return subgroups

    def _analyze_ordinal(
        self,
        df: pl.DataFrame,
        column: str,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict:
        """Analyze an ordinal column."""
        values = df[column].to_numpy()
        unique_vals = np.unique(values)

        subgroups = {}
        for val in unique_vals:
            mask = values == val
            if mask.sum() < 30:
                continue

            y_sub = y_true[mask]
            yp_sub = y_proba[mask]
            yd_sub = y_pred[mask]

            subgroups[str(val)] = self._compute_subgroup_metrics(
                y_sub, yp_sub, yd_sub, int(mask.sum())
            )

        return subgroups

    @staticmethod
    def _compute_subgroup_metrics(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        y_pred: np.ndarray,
        n_samples: int,
        extra: Dict = None,
    ) -> Dict:
        """Compute metrics for a single subgroup."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        metrics = {
            "n_samples": n_samples,
            "n_positive": int(y_true.sum()),
            "n_negative": int((~y_true.astype(bool)).sum()),
            "prevalence": float(y_true.mean()),
            "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0,
            "pr_auc": float(average_precision_score(y_true, y_proba)),
            "accuracy": float(np.mean(y_true == y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
            "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
            "fnr": float(fn / (fn + tp)) if (fn + tp) > 0 else 0,
            "npv": float(tn / (tn + fn)) if (tn + fn) > 0 else 0,
            "brier_score": float(brier_score_loss(y_true, y_proba)),
        }

        if extra:
            metrics.update(extra)

        return metrics

    # ==========================================
    # ERROR ANALYSIS
    # ==========================================

    def _error_analysis(
        self,
        df: pl.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict:
        """
        Analyze where the model makes errors.
        Focus on false negatives (missed cases) and false positives (false alarms).
        """
        false_negatives = (y_true == 1) & (y_pred == 0)
        false_positives = (y_true == 0) & (y_pred == 1)
        true_positives = (y_true == 1) & (y_pred == 1)
        true_negatives = (y_true == 0) & (y_pred == 0)

        fn_count = int(false_negatives.sum())
        fp_count = int(false_positives.sum())
        tp_count = int(true_positives.sum())
        tn_count = int(true_negatives.sum())

        # Profile of false negatives (missed depression cases)
        fn_profiles = {}
        for col in ["Gender", "Age", "Family History of Mental Illness"]:
            if col not in df.columns:
                continue
            values = df[col].to_numpy()
            fn_values = values[false_negatives]
            unique, counts = np.unique(fn_values, return_counts=True)
            fn_profiles[col] = {
                str(v): int(c) for v, c in zip(unique, counts)
            }

        # Profile of false positives (false alarms)
        fp_profiles = {}
        for col in ["Gender", "Age", "Family History of Mental Illness"]:
            if col not in df.columns:
                continue
            values = df[col].to_numpy()
            fp_values = values[false_positives]
            unique, counts = np.unique(fp_values, return_counts=True)
            fp_profiles[col] = {
                str(v): int(c) for v, c in zip(unique, counts)
            }

        # Confidence analysis: what probabilities did the model assign to errors?
        fn_confidence = y_proba[false_negatives]
        fp_confidence = y_proba[false_positives]

        return {
            "error_counts": {
                "false_negatives": fn_count,
                "false_positives": fp_count,
                "true_positives": tp_count,
                "true_negatives": tn_count,
                "fn_rate": fn_count / (fn_count + tp_count) if (fn_count + tp_count) > 0 else 0,
                "fp_rate": fp_count / (fp_count + tn_count) if (fp_count + tn_count) > 0 else 0,
            },
            "false_negative_profile": fn_profiles,
            "false_positive_profile": fp_profiles,
            "fn_confidence": {
                "mean": float(fn_confidence.mean()) if len(fn_confidence) > 0 else 0,
                "std": float(fn_confidence.std()) if len(fn_confidence) > 0 else 0,
                "min": float(fn_confidence.min()) if len(fn_confidence) > 0 else 0,
                "max": float(fn_confidence.max()) if len(fn_confidence) > 0 else 0,
                "median": float(np.median(fn_confidence)) if len(fn_confidence) > 0 else 0,
            },
            "fp_confidence": {
                "mean": float(fp_confidence.mean()) if len(fp_confidence) > 0 else 0,
                "std": float(fp_confidence.std()) if len(fp_confidence) > 0 else 0,
                "min": float(fp_confidence.min()) if len(fp_confidence) > 0 else 0,
                "max": float(fp_confidence.max()) if len(fp_confidence) > 0 else 0,
                "median": float(np.median(fp_confidence)) if len(fp_confidence) > 0 else 0,
            },
        }

    # ==========================================
    # CALIBRATION BY SUBGROUP
    # ==========================================

    def _calibration_by_subgroup(
        self,
        df: pl.DataFrame,
        subgroup_type: str,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict:
        """Compute calibration metrics for each subgroup."""
        if subgroup_type == "Age":
            ages = df["Age"].to_numpy()
            bins = [(18, 22), (23, 26), (27, 30), (31, 100)]
            labels = ["18-22", "23-26", "27-30", "31+"]

            calibrations = {}
            for label, (lo, hi) in zip(labels, bins):
                mask = (ages >= lo) & (ages <= hi)
                if mask.sum() < 30:
                    continue
                y_sub = y_true[mask]
                yp_sub = y_proba[mask]
                calibrations[label] = self._single_calibration(y_sub, yp_sub)

            return calibrations

        else:
            if subgroup_type == "Gender":
                column = "Gender"
            elif subgroup_type == "Family_History":
                column = "Family History of Mental Illness"
            elif subgroup_type == "City":
                column = "City"
            elif subgroup_type == "Academic_Pressure":
                column = "Academic Pressure"
            else:
                return {}

            if column not in df.columns:
                return {}

            values = df[column].to_numpy()
            unique_vals = np.unique(values)

            calibrations = {}
            for val in unique_vals:
                mask = values == val
                if mask.sum() < 30:
                    continue
                y_sub = y_true[mask]
                yp_sub = y_proba[mask]
                calibrations[str(val)] = self._single_calibration(y_sub, yp_sub)

            return calibrations

    @staticmethod
    def _single_calibration(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> Dict:
        """Compute calibration metrics for a single group."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_true_rates = []
        bin_pred_means = []
        bin_counts = []

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:
                mask = (y_proba >= lo) & (y_proba <= hi)
            else:
                mask = (y_proba >= lo) & (y_proba < hi)

            if mask.sum() == 0:
                continue

            bin_true_rates.append(float(y_true[mask].mean()))
            bin_pred_means.append(float(y_proba[mask].mean()))
            bin_counts.append(int(mask.sum()))

        # Expected Calibration Error (ECE)
        n = len(y_true)
        ece = sum(
            (count / n) * abs(true_rate - pred_mean)
            for true_rate, pred_mean, count in zip(bin_true_rates, bin_pred_means, bin_counts)
        )

        # Maximum Calibration Error (MCE)
        mce = max(
            abs(true_rate - pred_mean)
            for true_rate, pred_mean in zip(bin_true_rates, bin_pred_means)
        ) if bin_true_rates else 0

        return {
            "bin_true_rates": bin_true_rates,
            "bin_pred_means": bin_pred_means,
            "bin_counts": bin_counts,
            "ece": ece,
            "mce": mce,
            "n_samples": n,
        }

    # ==========================================
    # THRESHOLD RECOMMENDATIONS
    # ==========================================

    def _threshold_recommendations(
        self,
        df: pl.DataFrame,
        subgroups: Dict,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict:
        """
        Recommend optimal thresholds for each subgroup.
        Uses F1-optimal and cost-optimal (FN=2×FP) thresholds.
        """
        recommendations = {}

        # Overall
        recommendations["Overall"] = self._optimal_thresholds(y_true, y_proba)

        # By Gender
        if "Gender" in df.columns:
            genders = df["Gender"].to_numpy()
            for val in np.unique(genders):
                mask = genders == val
                if mask.sum() < 30:
                    continue
                recommendations[f"Gender={val}"] = self._optimal_thresholds(
                    y_true[mask], y_proba[mask]
                )

        # By Age group
        if "Age" in df.columns:
            ages = df["Age"].to_numpy()
            bins = [(18, 22), (23, 26), (27, 30), (31, 100)]
            labels = ["18-22", "23-26", "27-30", "31+"]
            for label, (lo, hi) in zip(labels, bins):
                mask = (ages >= lo) & (ages <= hi)
                if mask.sum() < 30:
                    continue
                recommendations[f"Age={label}"] = self._optimal_thresholds(
                    y_true[mask], y_proba[mask]
                )

        return recommendations

    @staticmethod
    def _optimal_thresholds(
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict:
        """Find optimal thresholds for a single group."""
        thresholds = np.arange(0.05, 0.96, 0.01)

        best_f1 = 0
        best_f1_threshold = 0.5
        best_cost = float("inf")
        best_cost_threshold = 0.5

        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                cost = 2 * fn + fp  # FN costs 2× FP
            else:
                cost = float("inf")

            if f1 > best_f1:
                best_f1 = f1
                best_f1_threshold = t

            if cost < best_cost:
                best_cost = cost
                best_cost_threshold = t

        return {
            "f1_optimal_threshold": best_f1_threshold,
            "f1_optimal_value": best_f1,
            "cost_optimal_threshold": best_cost_threshold,
            "cost_optimal_value": best_cost / len(y_true),
        }

    # ==========================================
    # VISUALIZATION
    # ==========================================

    def plot_subgroup_dashboard(
        self,
        results: Dict,
        output_path: str = "results/subgroup_dashboard.html",
    ):
        """Create comprehensive subgroup analysis dashboard."""
        model_name = results.get("model", "model")

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Performance by Subgroup (ROC-AUC)",
                "Performance by Subgroup (F1 Score)",
                "Error Distribution",
                "Calibration by Subgroup (ECE)",
                "Threshold Recommendations",
                "Confidence Distribution (FN vs FP)",
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "box"}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.10,
        )

        colors = px.colors.qualitative.Set2
        subgroup_idx = 0

        # 1 & 2. Performance by subgroup
        for subgroup_type, subgroup_data in results.get("subgroups", {}).items():
            if not isinstance(subgroup_data, dict):
                continue

            vals = []
            aucs = []
            f1s = []
            for val, metrics in subgroup_data.items():
                vals.append(f"{subgroup_type}: {val}")
                aucs.append(metrics.get("roc_auc", 0))
                f1s.append(metrics.get("f1", 0))

            color = colors[subgroup_idx % len(colors)]

            fig.add_trace(
                go.Bar(
                    x=vals, y=aucs,
                    name=f"{subgroup_type} - AUC",
                    marker_color=color,
                    legendgroup=subgroup_type,
                    showlegend=True,
                ),
                row=1, col=1,
            )

            fig.add_trace(
                go.Bar(
                    x=vals, y=f1s,
                    name=f"{subgroup_type} - F1",
                    marker_color=color,
                    legendgroup=subgroup_type,
                    showlegend=False,
                ),
                row=1, col=2,
            )

            subgroup_idx += 1

        # 3. Error Distribution Pie
        error_counts = results.get("error_analysis", {}).get("error_counts", {})
        if error_counts:
            labels = ["True Positive", "True Negative", "False Positive", "False Negative"]
            values = [
                error_counts.get("true_positives", 0),
                error_counts.get("true_negatives", 0),
                error_counts.get("false_positives", 0),
                error_counts.get("false_negatives", 0),
            ]
            fig.add_trace(
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4,
                    marker_colors=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"],
                ),
                row=2, col=1,
            )

        # 4. Calibration by Subgroup (ECE)
        cal_data = results.get("calibration_by_subgroup", {})
        cal_subgroups = []
        cal_ece = []
        for subgroup_type, calibrations in cal_data.items():
            for val, cal in calibrations.items():
                if isinstance(cal, dict) and "ece" in cal:
                    cal_subgroups.append(f"{subgroup_type}: {val}")
                    cal_ece.append(cal["ece"])

        if cal_subgroups:
            fig.add_trace(
                go.Bar(
                    x=cal_subgroups, y=cal_ece,
                    name="ECE",
                    marker_color="coral",
                ),
                row=2, col=2,
            )

        # 5. Threshold Recommendations Table
        threshold_recs = results.get("threshold_recommendations", {})
        if threshold_recs:
            table_headers = ["Subgroup", "F1-Optimal Threshold", "Cost-Optimal Threshold"]
            table_data = []
            for subgroup, recs in threshold_recs.items():
                table_data.append([
                    subgroup,
                    f"{recs.get('f1_optimal_threshold', 0):.2f}",
                    f"{recs.get('cost_optimal_threshold', 0):.2f}",
                ])

            fig.add_trace(
                go.Table(
                    header=dict(
                        values=table_headers,
                        fill_color="paleturquoise",
                        align="center",
                    ),
                    cells=dict(
                        values=list(zip(*table_data)) if table_data else [[] for _ in table_headers],
                        fill_color="lavender",
                        align="center",
                    ),
                ),
                row=3, col=1,
            )

        # 6. Confidence Distribution
        error_analysis = results.get("error_analysis", {})
        fn_conf = error_analysis.get("fn_confidence", {})
        fp_conf = error_analysis.get("fp_confidence", {})

        if fn_conf and fp_conf:
            # Create mock distributions for box plot
            fn_mean = fn_conf.get("mean", 0)
            fn_std = fn_conf.get("std", 0)
            fp_mean = fp_conf.get("mean", 0)
            fp_std = fp_conf.get("std", 0)

            fig.add_trace(
                go.Box(
                    name="False Negatives",
                    y=[fn_mean],
                    marker_color="#f39c12",
                    boxmean=True,
                    boxpoints=False,
                ),
                row=3, col=2,
            )

            fig.add_trace(
                go.Box(
                    name="False Positives",
                    y=[fp_mean],
                    marker_color="#e74c3c",
                    boxmean=True,
                    boxpoints=False,
                ),
                row=3, col=2,
            )

        fig.update_layout(
            title=f"🔍 Subgroup Analysis Dashboard — {model_name}",
            height=1100,
            width=1200,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
            margin=dict(l=60, r=60, t=60, b=60),
        )

        fig.write_html(output_path, include_plotlyjs=True, full_html=True)
        logger.info(f"✅ Saved subgroup dashboard: {output_path}")
        return output_path

    # ==========================================
    # REPORT PRINTING
    # ==========================================

    def print_report(self, results: Dict):
        """Print formatted subgroup analysis report."""
        model = results.get("model", "model")
        print(f"\n{'='*70}")
        print(f" 🔍 SUBGROUP ANALYSIS — {model.upper()}")
        print(f"{'='*70}\n")

        # Overall
        overall = results.get("overall", {})
        print(f" 📊 Overall Performance:")
        print(f"    N={overall.get('n_samples', 0)} | "
              f"Prevalence={overall.get('prevalence', 0):.1%} | "
              f"AUC={overall.get('roc_auc', 0):.3f} | "
              f"F1={overall.get('f1', 0):.3f} | "
              f"Brier={overall.get('brier_score', 0):.4f}")
        print()

        # Error Analysis
        error = results.get("error_analysis", {}).get("error_counts", {})
        if error:
            print(f" 🚨 Error Analysis:")
            print(f"    False Negatives: {error.get('false_negatives', 0)} "
                  f"(FNR={error.get('fn_rate', 0):.1%})")
            print(f"    False Positives: {error.get('false_positives', 0)} "
                  f"(FPR={error.get('fp_rate', 0):.1%})")

            fn_conf = results.get("error_analysis", {}).get("fn_confidence", {})
            if fn_conf:
                print(f"    FN Confidence: mean={fn_conf.get('mean', 0):.3f}, "
                      f"std={fn_conf.get('std', 0):.3f}")
            fp_conf = results.get("error_analysis", {}).get("fp_confidence", {})
            if fp_conf:
                print(f"    FP Confidence: mean={fp_conf.get('mean', 0):.3f}, "
                      f"std={fp_conf.get('std', 0):.3f}")
            print()

        # Subgroup Performance
        for subgroup_type, subgroup_data in results.get("subgroups", {}).items():
            if not isinstance(subgroup_data, dict):
                continue

            print(f" {'─'*60}")
            print(f" 📌 {subgroup_type}")
            print(f" {'─'*60}")

            for val, metrics in subgroup_data.items():
                print(f"    {val:15s} | N={metrics['n_samples']:5d} | "
                      f"Prev={metrics['prevalence']:.1%} | "
                      f"AUC={metrics['roc_auc']:.3f} | "
                      f"F1={metrics['f1']:.3f} | "
                      f"FPR={metrics['fpr']:.3f} | "
                      f"FNR={metrics['fnr']:.3f}")
            print()

        # Calibration by Subgroup
        cal_data = results.get("calibration_by_subgroup", {})
        if cal_data:
            print(f" {'─'*60}")
            print(f" 📐 Calibration by Subgroup (ECE)")
            print(f" {'─'*60}")
            for subgroup_type, calibrations in cal_data.items():
                for val, cal in calibrations.items():
                    if isinstance(cal, dict) and "ece" in cal:
                        print(f"    {subgroup_type}: {val:15s} | "
                              f"ECE={cal['ece']:.4f} | "
                              f"MCE={cal['mce']:.4f} | "
                              f"N={cal['n_samples']}")
            print()

        # Threshold Recommendations
        threshold_recs = results.get("threshold_recommendations", {})
        if threshold_recs:
            print(f" {'─'*60}")
            print(f" 🎯 Threshold Recommendations")
            print(f" {'─'*60}")
            for subgroup, recs in threshold_recs.items():
                print(f"    {subgroup:20s} | "
                      f"F1-optimal={recs.get('f1_optimal_threshold', 0):.2f} | "
                      f"Cost-optimal={recs.get('cost_optimal_threshold', 0):.2f}")
            print()
