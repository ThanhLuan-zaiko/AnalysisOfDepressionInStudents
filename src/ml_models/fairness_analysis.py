"""
Fairness Analysis Module
========================
Comprehensive fairness metrics: Demographic Parity, Equalized Odds,
Predictive Rate Parity, Disparate Impact, Theil Index.

Output: HTML reports + JSON results
"""

import numpy as np
import polars as pl
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from src.utils.helpers import save_json, Timer

import logging
logger = logging.getLogger(__name__)


class FairnessAnalyzer:
    """
    Comprehensive fairness analysis for classification models.
    """

    # Fairness thresholds
    DISPARATE_IMPACT_ACCEPTABLE = (0.8, 1.25)  # 4/5 rule
    FPR_DIFF_THRESHOLD = 0.10
    FNR_DIFF_THRESHOLD = 0.10
    PRECISION_DIFF_THRESHOLD = 0.10

    def __init__(self, sensitive_attributes: List[str] = None):
        """
        Args:
            sensitive_attributes: List of column names to analyze for fairness.
                                  Default: ["Gender", "Age", "Family History of Mental Illness"]
        """
        self.sensitive_attributes = sensitive_attributes or [
            "Gender",
            "Age",
            "Family History of Mental Illness",
        ]
        self.results = {}

    def analyze(
        self,
        df: pl.DataFrame,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model",
    ) -> Dict:
        """
        Run comprehensive fairness analysis.

        Returns dict with all fairness metrics.
        """
        timer = Timer("Fairness Analysis")
        logger.info(f"🔍 Starting fairness analysis for {model_name}...")

        with timer:
            results = {
                "model": model_name,
                "overall_metrics": self._overall_metrics(y_true, y_pred),
                "subgroup_metrics": {},
                "demographic_parity": {},
                "equalized_odds": {},
                "predictive_parity": {},
                "disparate_impact": {},
                "theil_index": {},
                "warnings": [],
            }

            for attr in self.sensitive_attributes:
                if attr not in df.columns:
                    logger.warning(f"⚠️  Column '{attr}' not found, skipping.")
                    continue

                logger.info(f"  📊 Analyzing {attr}...")

                subgroup_results = self._analyze_attribute(
                    df, attr, y_true, y_proba, y_pred
                )

                results["subgroup_metrics"][attr] = subgroup_results
                results["demographic_parity"][attr] = self._demographic_parity(
                    df, attr, y_pred
                )
                results["equalized_odds"][attr] = self._equalized_odds(
                    df, attr, y_true, y_pred
                )
                results["predictive_parity"][attr] = self._predictive_parity(
                    df, attr, y_true, y_pred
                )
                results["disparate_impact"][attr] = self._disparate_impact(
                    df, attr, y_pred
                )
                results["theil_index"][attr] = self._theil_index(
                    y_true, y_proba, df, attr
                )

            # Collect warnings
            results["warnings"] = self._collect_warnings(results)

        self.results[model_name] = results
        logger.info(f"✅ Fairness analysis completed in {timer.elapsed:.2f}s")

        return results

    # ==========================================
    # CORE ANALYSIS
    # ==========================================

    def _analyze_attribute(
        self,
        df: pl.DataFrame,
        attr: str,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict:
        """Analyze a single sensitive attribute."""
        values = df[attr].to_numpy()
        unique_vals = np.unique(values)

        subgroup_metrics = {}
        for val in unique_vals:
            mask = values == val
            if mask.sum() < 30:
                continue

            y_sub = y_true[mask]
            yp_sub = y_proba[mask]
            yd_sub = y_pred[mask]

            n_pos = int(y_sub.sum())
            n_neg = int((~y_sub.astype(bool)).sum())

            metrics = {
                "n_samples": int(mask.sum()),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "prevalence": float(y_sub.mean()),
                "roc_auc": float(roc_auc_score(y_sub, yp_sub)) if len(np.unique(y_sub)) > 1 else 0.0,
                "accuracy": float(np.mean(y_sub == yd_sub)),
                "precision": float(precision_score(y_sub, yd_sub, zero_division=0)),
                "recall": float(recall_score(y_sub, yd_sub, zero_division=0)),
                "f1": float(f1_score(y_sub, yd_sub, zero_division=0)),
                "specificity": self._specificity(y_sub, yd_sub),
                "fpr": self._fpr(y_sub, yd_sub),
                "fnr": self._fnr(y_sub, yd_sub),
                "npv": self._npv(y_sub, yd_sub),
                "positive_rate": float(yd_sub.mean()),
            }
            subgroup_metrics[str(val)] = metrics

        return subgroup_metrics

    def _demographic_parity(
        self,
        df: pl.DataFrame,
        attr: str,
        y_pred: np.ndarray,
    ) -> Dict:
        """
        Demographic Parity: P(Ŷ=1|A=a) should be similar across groups.
        Also called Statistical Parity.
        """
        values = df[attr].to_numpy()
        unique_vals = np.unique(values)

        rates = {}
        for val in unique_vals:
            mask = values == val
            if mask.sum() < 30:
                continue
            rates[str(val)] = float(y_pred[mask].mean())

        rate_values = list(rates.values())
        max_diff = max(rate_values) - min(rate_values) if rate_values else 0
        std_diff = float(np.std(rate_values)) if rate_values else 0

        return {
            "positive_prediction_rates": rates,
            "max_difference": max_diff,
            "std_difference": std_diff,
            "passes": max_diff < 0.10,  # Tolerance threshold
            "interpretation": (
                "✅ Pass" if max_diff < 0.10 else
                f"⚠️  Fail (Δ={max_diff:.3f} > 0.10)"
            ),
        }

    def _equalized_odds(
        self,
        df: pl.DataFrame,
        attr: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict:
        """
        Equalized Odds: TPR and FPR should be similar across groups.
        """
        values = df[attr].to_numpy()
        unique_vals = np.unique(values)

        tpr_by_group = {}
        fpr_by_group = {}

        for val in unique_vals:
            mask = values == val
            if mask.sum() < 30:
                continue

            y_sub = y_true[mask]
            yd_sub = y_pred[mask]

            tpr_by_group[str(val)] = self._tpr(y_sub, yd_sub)
            fpr_by_group[str(val)] = self._fpr(y_sub, yd_sub)

        tpr_diff = max(tpr_by_group.values()) - min(tpr_by_group.values()) if tpr_by_group else 0
        fpr_diff = max(fpr_by_group.values()) - min(fpr_by_group.values()) if fpr_by_group else 0

        return {
            "true_positive_rates": tpr_by_group,
            "false_positive_rates": fpr_by_group,
            "tpr_max_diff": tpr_diff,
            "fpr_max_diff": fpr_diff,
            "passes_tpr": tpr_diff < self.FNR_DIFF_THRESHOLD,
            "passes_fpr": fpr_diff < self.FPR_DIFF_THRESHOLD,
            "interpretation": (
                f"TPR Δ={tpr_diff:.3f}, FPR Δ={fpr_diff:.3f}"
            ),
        }

    def _predictive_parity(
        self,
        df: pl.DataFrame,
        attr: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict:
        """
        Predictive Rate Parity: Precision should be similar across groups.
        """
        values = df[attr].to_numpy()
        unique_vals = np.unique(values)

        precision_by_group = {}
        for val in unique_vals:
            mask = values == val
            if mask.sum() < 30:
                continue

            y_sub = y_true[mask]
            yd_sub = y_pred[mask]
            precision_by_group[str(val)] = precision_score(
                y_sub, yd_sub, zero_division=0
            )

        precision_diff = (
            max(precision_by_group.values()) - min(precision_by_group.values())
            if precision_by_group
            else 0
        )

        return {
            "precision_by_group": {k: float(v) for k, v in precision_by_group.items()},
            "max_difference": precision_diff,
            "passes": precision_diff < self.PRECISION_DIFF_THRESHOLD,
            "interpretation": (
                f"Precision Δ={precision_diff:.3f}"
            ),
        }

    def _disparate_impact(
        self,
        df: pl.DataFrame,
        attr: str,
        y_pred: np.ndarray,
    ) -> Dict:
        """
        Disparate Impact Ratio:
        DI = min(P(Ŷ=1|A=a)) / max(P(Ŷ=1|A=a))
        Should be >= 0.8 (4/5 rule).
        """
        values = df[attr].to_numpy()
        unique_vals = np.unique(values)

        rates = {}
        for val in unique_vals:
            mask = values == val
            if mask.sum() < 30:
                continue
            rates[str(val)] = float(y_pred[mask].mean())

        if not rates:
            return {"ratio": 1.0, "passes": True, "interpretation": "✅ Pass"}

        max_rate = max(rates.values())
        min_rate = min(rates.values())
        ratio = min_rate / max_rate if max_rate > 0 else 0

        passes = ratio >= 0.8

        return {
            "rates": rates,
            "ratio": ratio,
            "max_rate_group": max(rates, key=rates.get),
            "min_rate_group": min(rates, key=rates.get),
            "passes": passes,
            "interpretation": (
                f"✅ Pass (DI={ratio:.3f} >= 0.80)" if passes
                else f"❌ Fail (DI={ratio:.3f} < 0.80) — 4/5 rule violation"
            ),
        }

    def _theil_index(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        df: pl.DataFrame,
        attr: str,
    ) -> Dict:
        """
        Theil Index: Measures inequality in prediction confidence.
        T = (1/n) Σ (p_i / μ) ln(p_i / μ)
        where μ = mean probability.
        T = 0 means perfect equality, T > 0 means inequality.
        """
        # Overall Theil
        mu = y_proba.mean()
        if mu == 0:
            overall_theil = 0
        else:
            ratio = y_proba / mu
            # Handle zero probabilities
            ratio_safe = np.where(ratio > 0, ratio, 1e-10)
            overall_theil = float(np.mean(ratio_safe * np.log(ratio_safe)))

        # By subgroup
        values = df[attr].to_numpy()
        unique_vals = np.unique(values)
        subgroup_theil = {}
        subgroup_weights = {}

        for val in unique_vals:
            mask = values == val
            if mask.sum() < 30:
                continue

            p_sub = y_proba[mask]
            mu_sub = p_sub.mean()
            if mu_sub == 0:
                theil_sub = 0
            else:
                ratio_sub = p_sub / mu_sub
                ratio_safe = np.where(ratio_sub > 0, ratio_sub, 1e-10)
                theil_sub = float(np.mean(ratio_safe * np.log(ratio_safe)))

            subgroup_theil[str(val)] = theil_sub
            subgroup_weights[str(val)] = int(mask.sum())

        return {
            "overall_theil": overall_theil,
            "subgroup_theil": subgroup_theil,
            "subgroup_weights": subgroup_weights,
            "interpretation": (
                f"{'Low' if overall_theil < 0.1 else 'Moderate' if overall_theil < 0.3 else 'High'} "
                f"inequality (T={overall_theil:.4f})"
            ),
        }

    # ==========================================
    # HELPER METRICS
    # ==========================================

    def _overall_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Overall model metrics."""
        return {
            "accuracy": float(np.mean(y_true == y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

    @staticmethod
    def _tpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """True Positive Rate (Recall)."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    @staticmethod
    def _fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """False Positive Rate."""
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    @staticmethod
    def _fnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """False Negative Rate."""
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        return float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    @staticmethod
    def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Specificity (True Negative Rate)."""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    @staticmethod
    def _npv(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Negative Predictive Value."""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    # ==========================================
    # WARNINGS
    # ==========================================

    def _collect_warnings(self, results: Dict) -> List[Dict]:
        """Collect all fairness warnings."""
        warnings = []

        for attr, di in results.get("disparate_impact", {}).items():
            if not di.get("passes", True):
                warnings.append({
                    "type": "disparate_impact",
                    "attribute": attr,
                    "severity": "high",
                    "message": f"{attr}: {di['interpretation']}",
                })

        for attr, eo in results.get("equalized_odds", {}).items():
            if not eo.get("passes_fpr", True):
                warnings.append({
                    "type": "equalized_odds_fpr",
                    "attribute": attr,
                    "severity": "medium",
                    "message": f"{attr}: FPR disparity Δ={eo['fpr_max_diff']:.3f}",
                })
            if not eo.get("passes_tpr", True):
                warnings.append({
                    "type": "equalized_odds_tpr",
                    "attribute": attr,
                    "severity": "medium",
                    "message": f"{attr}: TPR disparity Δ={eo['tpr_max_diff']:.3f}",
                })

        for attr, dp in results.get("demographic_parity", {}).items():
            if not dp.get("passes", True):
                warnings.append({
                    "type": "demographic_parity",
                    "attribute": attr,
                    "severity": "low",
                    "message": f"{attr}: {dp['interpretation']}",
                })

        return warnings

    # ==========================================
    # VISUALIZATION
    # ==========================================

    def plot_fairness_dashboard(
        self,
        results: Dict,
        output_path: str = "results/fairness_dashboard.html",
    ):
        """Create comprehensive fairness dashboard HTML."""
        model_name = results.get("model", "model")

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Demographic Parity (Positive Prediction Rate)",
                "Equalized Odds (TPR vs FPR)",
                "Disparate Impact Ratio",
                "Predictive Parity (Precision)",
                "Theil Index (Inequality)",
                "Subgroup Performance Overview",
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "table"}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.10,
        )

        # Collect all attributes
        attrs = list(results.get("subgroup_metrics", {}).keys())
        colors = px.colors.qualitative.Set2

        for idx, attr in enumerate(attrs):
            color = colors[idx % len(colors)]

            # 1. Demographic Parity
            subgroup_metrics = results.get("subgroup_metrics", {}).get(attr, {})
            if subgroup_metrics:
                vals = list(subgroup_metrics.keys())
                pos_rates = [m["positive_rate"] for m in subgroup_metrics.values()]

                fig.add_trace(
                    go.Bar(
                        x=vals,
                        y=pos_rates,
                        name=f"{attr} - DP",
                        marker_color=color,
                        legendgroup=attr,
                        showlegend=(idx == 0),
                    ),
                    row=1, col=1,
                )

            # 2. Equalized Odds
            eo = results.get("equalized_odds", {}).get(attr, {})
            if eo:
                tpr_vals = list(eo.get("true_positive_rates", {}).values())
                fpr_vals = list(eo.get("false_positive_rates", {}).values())
                labels = list(eo.get("true_positive_rates", {}).keys())

                fig.add_trace(
                    go.Scatter(
                        x=fpr_vals,
                        y=tpr_vals,
                        mode="markers+text",
                        text=labels,
                        textposition="top center",
                        name=f"{attr} - EO",
                        marker=dict(color=color, size=12),
                        legendgroup=attr,
                        showlegend=False,
                    ),
                    row=1, col=2,
                )

            # 3. Disparate Impact
            di = results.get("disparate_impact", {}).get(attr, {})
            if di and "rates" in di:
                di_vals = list(di["rates"].keys())
                di_rates = list(di["rates"].values())

                fig.add_trace(
                    go.Bar(
                        x=di_vals,
                        y=di_rates,
                        name=f"{attr} - DI",
                        marker_color=color,
                        legendgroup=attr,
                        showlegend=False,
                    ),
                    row=2, col=1,
                )

                # Add threshold line
                fig.add_trace(
                    go.Scatter(
                        x=di_vals,
                        y=[0.8] * len(di_vals),
                        mode="lines",
                        name="DI Threshold (0.80)",
                        line=dict(color="red", dash="dash"),
                        showlegend=(idx == 0),
                    ),
                    row=2, col=1,
                )

            # 4. Predictive Parity
            pp = results.get("predictive_parity", {}).get(attr, {})
            if pp and "precision_by_group" in pp:
                pp_vals = list(pp["precision_by_group"].keys())
                pp_rates = list(pp["precision_by_group"].values())

                fig.add_trace(
                    go.Bar(
                        x=pp_vals,
                        y=pp_rates,
                        name=f"{attr} - PP",
                        marker_color=color,
                        legendgroup=attr,
                        showlegend=False,
                    ),
                    row=2, col=2,
                )

            # 5. Theil Index
            ti = results.get("theil_index", {}).get(attr, {})
            if ti and "subgroup_theil" in ti:
                ti_vals = list(ti["subgroup_theil"].keys())
                ti_rates = list(ti["subgroup_theil"].values())

                fig.add_trace(
                    go.Bar(
                        x=ti_vals,
                        y=ti_rates,
                        name=f"{attr} - Theil",
                        marker_color=color,
                        legendgroup=attr,
                        showlegend=False,
                    ),
                    row=3, col=1,
                )

        # 6. Subgroup Performance Table
        if attrs:
            table_data = []
            headers = ["Subgroup", "N", "AUC", "F1", "Precision", "Recall", "FPR", "FNR"]
            for attr in attrs:
                subgroup_metrics = results.get("subgroup_metrics", {}).get(attr, {})
                for val, metrics in subgroup_metrics.items():
                    table_data.append([
                        f"{attr}: {val}",
                        metrics.get("n_samples", "-"),
                        f"{metrics.get('roc_auc', 0):.3f}",
                        f"{metrics.get('f1', 0):.3f}",
                        f"{metrics.get('precision', 0):.3f}",
                        f"{metrics.get('recall', 0):.3f}",
                        f"{metrics.get('fpr', 0):.3f}",
                        f"{metrics.get('fnr', 0):.3f}",
                    ])

            fig.add_trace(
                go.Table(
                    header=dict(
                        values=headers,
                        fill_color="paleturquoise",
                        align="center",
                        font=dict(size=11),
                    ),
                    cells=dict(
                        values=list(zip(*table_data)) if table_data else [[] for _ in headers],
                        fill_color="lavender",
                        align="center",
                        font=dict(size=10),
                    ),
                ),
                row=3, col=2,
            )

        fig.update_layout(
            title=f"⚖️  Fairness Dashboard — {model_name}",
            height=1000,
            width=1200,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05),
            margin=dict(l=60, r=60, t=60, b=60),
        )

        # Axis labels
        fig.update_xaxes(title_text="Subgroup", row=1, col=1)
        fig.update_yaxes(title_text="Positive Prediction Rate", row=1, col=1)
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)
        fig.update_xaxes(title_text="Subgroup", row=2, col=1)
        fig.update_yaxes(title_text="Positive Prediction Rate", row=2, col=1)
        fig.update_xaxes(title_text="Subgroup", row=2, col=2)
        fig.update_yaxes(title_text="Precision", row=2, col=2)
        fig.update_xaxes(title_text="Subgroup", row=3, col=1)
        fig.update_yaxes(title_text="Theil Index", row=3, col=1)

        fig.write_html(output_path, include_plotlyjs=True, full_html=True)
        logger.info(f"✅ Saved fairness dashboard: {output_path}")
        return output_path

    # ==========================================
    # REPORT PRINTING
    # ==========================================

    def print_report(self, results: Dict):
        """Print formatted fairness report to console."""
        model = results.get("model", "model")
        print(f"\n{'='*70}")
        print(f" ⚖️  FAIRNESS REPORT — {model.upper()}")
        print(f"{'='*70}\n")

        # Overall metrics
        overall = results.get("overall_metrics", {})
        print(f" 📊 Overall Performance:")
        print(f"    Accuracy:  {overall.get('accuracy', 0):.4f}")
        print(f"    Precision: {overall.get('precision', 0):.4f}")
        print(f"    Recall:    {overall.get('recall', 0):.4f}")
        print(f"    F1 Score:  {overall.get('f1', 0):.4f}")
        print()

        # Warnings
        warnings = results.get("warnings", [])
        if warnings:
            print(f" 🚨 FAIRNESS WARNINGS ({len(warnings)}):")
            for w in warnings:
                severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(w["severity"], "⚪")
                print(f"    {severity_icon} [{w['severity'].upper()}] {w['message']}")
            print()
        else:
            print(" ✅ No fairness warnings detected!\n")

        # Per-attribute analysis
        for attr in results.get("subgroup_metrics", {}):
            print(f" {'─'*60}")
            print(f" 📌 {attr}")
            print(f" {'─'*60}")

            # Subgroup metrics
            subgroups = results["subgroup_metrics"][attr]
            for val, metrics in subgroups.items():
                print(f"    {val:15s} | N={metrics['n_samples']:5d} | "
                      f"AUC={metrics['roc_auc']:.3f} | "
                      f"F1={metrics['f1']:.3f} | "
                      f"Precision={metrics['precision']:.3f} | "
                      f"Recall={metrics['recall']:.3f} | "
                      f"FPR={metrics['fpr']:.3f} | "
                      f"FNR={metrics['fnr']:.3f}")

            print()

            # Demographic Parity
            dp = results.get("demographic_parity", {}).get(attr, {})
            if dp:
                print(f"    Demographic Parity: {dp['interpretation']}")
                print(f"      Max Δ={dp['max_difference']:.3f}")

            # Disparate Impact
            di = results.get("disparate_impact", {}).get(attr, {})
            if di:
                print(f"    Disparate Impact: {di['interpretation']}")

            # Equalized Odds
            eo = results.get("equalized_odds", {}).get(attr, {})
            if eo:
                print(f"    Equalized Odds: {eo['interpretation']}")

            print()
