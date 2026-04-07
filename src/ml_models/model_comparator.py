"""
Model Comparator - So sánh toàn diện các mô hình Depression Analysis

So sánh Baseline → Logistic → GAM → CatBoost về:
- Performance metrics (ROC-AUC, PR-AUC, F1, Recall, Precision, Brier)
- Statistical significance tests (McNemar's, DeLong's)
- Calibration curves
- Decision curve analysis (clinical utility)
- Feature importance consistency
- Runtime complexity

Usage:
    from src.ml_models.model_comparator import ModelComparator
    
    comparator = ModelComparator()
    comparator.add_model("logistic", y_true, y_proba, y_pred)
    comparator.add_model("gam", y_true, y_proba, y_pred)
    results = comparator.compare_all()
    comparator.plot_calibration_curves(save_path)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    recall_score, precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelComparator:
    """
    So sánh toàn diện nhiều mô hình classification.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.models = {}  # name -> {y_true, y_proba, y_pred}
        self.results = {}
        logger.info(f"ModelComparator initialized (confidence={confidence_level})")
    
    def add_model(
        self,
        name: str,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        y_pred: np.ndarray,
        metrics: Optional[Dict] = None,
    ):
        """
        Thêm một mô hình vào danh sách so sánh.
        
        Args:
            name: Tên mô hình
            y_true: Ground truth labels
            y_proba: Predicted probabilities (positive class)
            y_pred: Predicted labels
            metrics: Optional pre-computed metrics
        """
        assert len(y_true) == len(y_proba) == len(y_pred), \
            f"Length mismatch for model {name}"
        
        self.models[name] = {
            "y_true": y_true,
            "y_proba": y_proba,
            "y_pred": y_pred,
            "metrics": metrics or {},
        }
        
        logger.info(f"Added model: {name} (n={len(y_true)})")
    
    def compute_basic_metrics(self) -> pd.DataFrame:
        """
        Tính toán cơ bản metrics cho tất cả mô hình.
        
        Returns:
            DataFrame với rows là models, columns là metrics
        """
        rows = []
        
        for name, data in self.models.items():
            y_true = data["y_true"]
            y_proba = data["y_proba"]
            y_pred = data["y_pred"]
            
            # Classification metrics
            roc_auc = roc_auc_score(y_true, y_proba)
            pr_auc = average_precision_score(y_true, y_proba)
            brier = brier_score_loss(y_true, y_proba)
            f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
            
            # Rates
            sensitivity = recall  # Same thing
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
            
            # Accuracy
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            # Prevalence
            prevalence = y_true.mean()
            
            rows.append({
                "model": name,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "brier_score": brier,
                "f1": f1,
                "recall": recall,
                "precision": precision,
                "accuracy": accuracy,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "fpr": fpr,
                "fnr": fnr,
                "ppv": ppv,
                "npv": npv,
                "prevalence": prevalence,
                "n_samples": len(y_true),
                "n_positive": int(y_true.sum()),
            })
        
        df_metrics = pd.DataFrame(rows)
        self.results["basic_metrics"] = df_metrics
        
        return df_metrics
    
    def mcnemar_test(
        self,
        model1_name: str,
        model2_name: str,
    ) -> Dict:
        """
        McNemar's test: So sánh hai mô hình có significant difference không.
        
        Test xem disagreement giữa 2 models có systematic không.
        Null hypothesis: Hai models có cùng performance.
        
        Returns:
            Dict với test statistic, p-value, interpretation
        """
        from scipy.stats import chi2
        
        if model1_name not in self.models or model2_name not in self.models:
            raise ValueError(f"Models not found: {model1_name}, {model2_name}")
        
        y_pred1 = self.models[model1_name]["y_pred"]
        y_pred2 = self.models[model2_name]["y_pred"]
        y_true = self.models[model1_name]["y_true"]  # Same for both
        
        # Contingency table
        both_correct = ((y_pred1 == y_true) & (y_pred2 == y_true)).sum()
        both_wrong = ((y_pred1 != y_true) & (y_pred2 != y_true)).sum()
        model1_correct_only = ((y_pred1 == y_true) & (y_pred2 != y_true)).sum()
        model2_correct_only = ((y_pred1 != y_true) & (y_pred2 == y_true)).sum()
        
        # McNemar's chi-square statistic (with continuity correction)
        b = model1_correct_only  # Model 1 đúng, Model 2 sai
        c = model2_correct_only  # Model 2 đúng, Model 1 sai
        
        if b + c == 0:
            return {
                "model1": model1_name,
                "model2": model2_name,
                "chi2_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "interpretation": "Hai mô hình giống hệt nhau",
                "contingency": {
                    "both_correct": int(both_correct),
                    "both_wrong": int(both_wrong),
                    f"{model1_name}_only_correct": int(b),
                    f"{model2_name}_only_correct": int(c),
                },
            }
        
        # Chi-square với continuity correction
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(chi2_stat, df=1)
        
        significant = p_value < (1 - self.confidence_level)
        
        interpretation = (
            f"Có sự khác biệt significant giữa {model1_name} và {model2_name} (p={p_value:.4f})"
            if significant
            else f"Không có sự khác biệt significant giữa {model1_name} và {model2_name} (p={p_value:.4f})"
        )
        
        result = {
            "model1": model1_name,
            "model2": model2_name,
            "chi2_statistic": float(chi2_stat),
            "p_value": float(p_value),
            "significant": significant,
            "interpretation": interpretation,
            "contingency": {
                "both_correct": int(both_correct),
                "both_wrong": int(both_wrong),
                f"{model1_name}_only_correct": int(b),
                f"{model2_name}_only_correct": int(c),
            },
        }
        
        return result
    
    def delong_test(
        self,
        model1_name: str,
        model2_name: str,
    ) -> Dict:
        """
        DeLong's test: So sánh hai ROC-AUC curves có significant difference không.
        
        Implemented using bootstrap approximation (DeLong exact requires full covariance).
        """
        if model1_name not in self.models or model2_name not in self.models:
            raise ValueError(f"Models not found: {model1_name}, {model2_name}")
        
        y_true = self.models[model1_name]["y_true"]
        y_proba1 = self.models[model1_name]["y_proba"]
        y_proba2 = self.models[model2_name]["y_proba"]
        
        # Bootstrap test
        n_bootstrap = 2000
        rng = np.random.RandomState(42)
        auc_diffs = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = rng.choice(len(y_true), size=len(y_true), replace=True)
            y_true_boot = y_true[indices]
            y_proba1_boot = y_proba1[indices]
            y_proba2_boot = y_proba2[indices]
            
            # Check both classes present
            if len(np.unique(y_true_boot)) < 2:
                continue
            
            auc1 = roc_auc_score(y_true_boot, y_proba1_boot)
            auc2 = roc_auc_score(y_true_boot, y_proba2_boot)
            auc_diffs.append(auc1 - auc2)
        
        auc_diffs = np.array(auc_diffs)
        mean_diff = auc_diffs.mean()
        std_diff = auc_diffs.std()
        
        # 95% CI
        ci_lower = np.percentile(auc_diffs, 2.5)
        ci_upper = np.percentile(auc_diffs, 97.5)
        
        # P-value (two-tailed)
        p_value = 2 * min(
            (auc_diffs > 0).mean(),
            (auc_diffs < 0).mean(),
        )
        
        significant = (ci_lower > 0) or (ci_upper < 0)
        
        auc1 = roc_auc_score(y_true, y_proba1)
        auc2 = roc_auc_score(y_true, y_proba2)
        
        result = {
            "model1": model1_name,
            "model2": model2_name,
            "auc1": auc1,
            "auc2": auc2,
            "auc_diff": mean_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": float(p_value),
            "significant": significant,
            "interpretation": (
                f"ROC-AUC của {model1_name} {'cao hơn' if mean_diff > 0 else 'thấp hơn'} "
                f"{model2_name} một cách significant (p={p_value:.4f})"
                if significant
                else f"Không có sự khác biệt significant về ROC-AUC giữa {model1_name} và {model2_name} (p={p_value:.4f})"
            ),
        }
        
        return result
    
    def run_statistical_tests(self) -> Dict:
        """
        Chạy tất cả statistical tests cho tất cả cặp models.
        """
        model_names = list(self.models.keys())
        
        if len(model_names) < 2:
            return {"error": "Cần ít nhất 2 models để so sánh"}
        
        tests = {
            "mcnemar": [],
            "delong": [],
        }
        
        # Pairwise comparisons
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                m1, m2 = model_names[i], model_names[j]
                
                # McNemar's test
                try:
                    mcnemar_result = self.mcnemar_test(m1, m2)
                    tests["mcnemar"].append(mcnemar_result)
                except Exception as e:
                    logger.warning(f"McNemar test failed for {m1} vs {m2}: {e}")
                
                # DeLong's test
                try:
                    delong_result = self.delong_test(m1, m2)
                    tests["delong"].append(delong_result)
                except Exception as e:
                    logger.warning(f"DeLong test failed for {m1} vs {m2}: {e}")
        
        self.results["statistical_tests"] = tests
        return tests
    
    def calibration_analysis(self) -> Dict:
        """
        Phân tích calibration cho từng mô hình.
        """
        from sklearn.calibration import calibration_curve
        
        calibration_results = {}
        
        for name, data in self.models.items():
            y_true = data["y_true"]
            y_proba = data["y_proba"]
            
            # Calibration curve
            prob_true, prob_pred = calibration_curve(
                y_true, y_proba, n_bins=10, strategy="uniform"
            )
            
            # Brier score
            brier = brier_score_loss(y_true, y_proba)
            
            # ECE (Expected Calibration Error)
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            
            for i in range(n_bins):
                mask = (y_proba > bin_boundaries[i]) & (y_proba <= bin_boundaries[i + 1])
                if mask.sum() == 0:
                    continue
                bin_acc = y_true[mask].mean()
                bin_conf = y_proba[mask].mean()
                ece += mask.sum() * abs(bin_acc - bin_conf)
            
            ece /= len(y_true)
            
            calibration_results[name] = {
                "prob_true": prob_true.tolist(),
                "prob_pred": prob_pred.tolist(),
                "brier_score": brier,
                "ece": ece,
            }
        
        self.results["calibration"] = calibration_results
        return calibration_results
    
    def decision_curve_analysis(
        self,
        threshold_range: Tuple[float, float] = (0.01, 0.99),
        n_thresholds: int = 100,
    ) -> pd.DataFrame:
        """
        Decision Curve Analysis (DCA): Đánh giá clinical utility.
        
        Net benefit = (TP / N) - (FP / N) * (pt / (1 - pt))
        trong đó pt là threshold probability.
        
        Returns:
            DataFrame với threshold, net_benefit cho từng model
        """
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
        
        rows = []
        n_total = len(list(self.models.values())[0]["y_true"])
        
        for thresh in thresholds:
            row = {"threshold": thresh}
            
            # Treat-all strategy
            prevalence = np.mean([d["y_true"].mean() for d in self.models.values()])
            net_benefit_all = prevalence - (1 - prevalence) * (thresh / (1 - thresh))
            row["treat_all"] = net_benefit_all
            row["treat_none"] = 0.0  # Baseline
            
            # Each model
            for name, data in self.models.items():
                y_true = data["y_true"]
                y_proba = data["y_proba"]
                
                y_pred = (y_proba >= thresh).astype(int)
                tp = ((y_pred == 1) & (y_true == 1)).sum()
                fp = ((y_pred == 1) & (y_true == 0)).sum()
                
                net_benefit = (tp / n_total) - (fp / n_total) * (thresh / (1 - thresh))
                row[f"{name}_net_benefit"] = net_benefit
            
            rows.append(row)
        
        df_dca = pd.DataFrame(rows)
        self.results["decision_curve"] = df_dca
        
        return df_dca
    
    def compare_all(self) -> Dict:
        """
        Chạy tất cả comparisons và trả về comprehensive report.
        """
        logger.info("Running comprehensive model comparison...")
        
        # Basic metrics
        df_metrics = self.compute_basic_metrics()
        
        # Statistical tests
        tests = self.run_statistical_tests()
        
        # Calibration
        calibration = self.calibration_analysis()
        
        # Decision curve
        df_dca = self.decision_curve_analysis()
        
        # Ranking
        df_ranked = df_metrics.sort_values("roc_auc", ascending=False).reset_index(drop=True)
        df_ranked["rank"] = range(1, len(df_ranked) + 1)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "n_models": len(self.models),
            "models": list(self.models.keys()),
            "basic_metrics": df_metrics.to_dict(orient="records"),
            "ranking": df_ranked[["rank", "model", "roc_auc", "pr_auc", "f1", "brier_score"]].to_dict(orient="records"),
            "statistical_tests": tests,
            "calibration": calibration,
            "decision_curve_summary": {
                "optimal_thresholds": self._find_optimal_thresholds(),
            },
        }
        
        self.results["full_report"] = report
        return report
    
    def _find_optimal_thresholds(self) -> Dict[str, float]:
        """
        Tìm optimal threshold cho từng model (maximize F1).
        """
        optimal = {}
        
        for name, data in self.models.items():
            y_true = data["y_true"]
            y_proba = data["y_proba"]
            
            thresholds = np.arange(0.1, 0.9, 0.01)
            best_f1 = -1
            best_thresh = 0.5
            
            for thresh in thresholds:
                y_pred = (y_proba >= thresh).astype(int)
                f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            
            optimal[name] = round(float(best_thresh), 2)
        
        return optimal
    
    def plot_comparison_chart(
        self,
        save_path: str = "results/model_comparison.html",
    ) -> Path:
        """
        Vẽ biểu đồ so sánh tổng hợp các models.
        """
        import plotly.graph_objects as go
        import plotly.io as pio
        
        if "basic_metrics" not in self.results:
            self.compute_basic_metrics()
        
        df = self.results["basic_metrics"]
        
        # Subplot: ROC-AUC, PR-AUC, F1, Brier
        fig = go.Figure()
        
        # Bar chart cho ROC-AUC
        fig.add_trace(go.Bar(
            name="ROC-AUC",
            x=df["model"],
            y=df["roc_auc"],
            marker_color="rgb(50, 100, 200)",
        ))
        
        fig.add_trace(go.Bar(
            name="PR-AUC",
            x=df["model"],
            y=df["pr_auc"],
            marker_color="rgb(100, 150, 250)",
        ))
        
        fig.add_trace(go.Bar(
            name="F1 Score",
            x=df["model"],
            y=df["f1"],
            marker_color="rgb(200, 100, 50)",
        ))
        
        fig.update_layout(
            title="Model Comparison: ROC-AUC, PR-AUC, F1",
            xaxis_title="Model",
            yaxis_title="Score",
            template="plotly_white",
            barmode="group",
            height=500,
            width=900,
        )
        
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pio.write_html(fig, str(output_path), full_html=True, include_plotlyjs=True)
        logger.info(f"Saved model comparison chart: {output_path}")
        
        return output_path
    
    def plot_calibration_curves(
        self,
        save_path: str = "results/calibration_curves.html",
    ) -> Path:
        """
        Vẽ calibration curves cho tất cả models.
        """
        import plotly.graph_objects as go
        import plotly.io as pio
        
        if "calibration" not in self.results:
            self.calibration_analysis()
        
        cal = self.results["calibration"]
        
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="gray", width=2, dash="dash"),
        ))
        
        # Each model
        colors = ["rgb(50, 100, 200)", "rgb(200, 100, 50)", "rgb(100, 200, 100)", "rgb(200, 50, 100)"]
        for idx, (name, cal_data) in enumerate(cal.items()):
            color = colors[idx % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=cal_data["prob_pred"],
                y=cal_data["prob_true"],
                mode="lines+markers",
                name=f"{name} (Brier={cal_data['brier_score']:.4f})",
                line=dict(color=color, width=3),
                marker=dict(size=8),
            ))
        
        fig.update_layout(
            title="Calibration Curves",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            template="plotly_white",
            height=600,
            width=800,
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
        )
        
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pio.write_html(fig, str(output_path), full_html=True, include_plotlyjs=True)
        logger.info(f"Saved calibration curves: {output_path}")
        
        return output_path
    
    def plot_decision_curves(
        self,
        save_path: str = "results/decision_curves.html",
    ) -> Path:
        """
        Vẽ decision curves cho tất cả models.
        """
        import plotly.graph_objects as go
        import plotly.io as pio
        
        if "decision_curve" not in self.results:
            self.decision_curve_analysis()
        
        df_dca = self.results["decision_curve"]
        
        fig = go.Figure()
        
        # Treat-all line
        fig.add_trace(go.Scatter(
            x=df_dca["threshold"],
            y=df_dca["treat_all"],
            mode="lines",
            name="Treat All",
            line=dict(color="gray", width=2, dash="dash"),
        ))
        
        # Treat-none line
        fig.add_trace(go.Scatter(
            x=df_dca["threshold"],
            y=df_dca["treat_none"],
            mode="lines",
            name="Treat None",
            line=dict(color="gray", width=2, dash="dot"),
        ))
        
        # Each model
        colors = ["rgb(50, 100, 200)", "rgb(200, 100, 50)", "rgb(100, 200, 100)", "rgb(200, 50, 100)"]
        for idx, name in enumerate(self.models.keys()):
            color = colors[idx % len(colors)]
            net_benefit_col = f"{name}_net_benefit"
            
            if net_benefit_col in df_dca.columns:
                fig.add_trace(go.Scatter(
                    x=df_dca["threshold"],
                    y=df_dca[net_benefit_col],
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=3),
                ))
        
        fig.update_layout(
            title="Decision Curve Analysis (Clinical Utility)",
            xaxis_title="Threshold Probability",
            yaxis_title="Net Benefit",
            template="plotly_white",
            height=600,
            width=900,
        )
        
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pio.write_html(fig, str(output_path), full_html=True, include_plotlyjs=True)
        logger.info(f"Saved decision curves: {output_path}")
        
        return output_path
    
    def print_comparison_report(self) -> str:
        """
        In báo cáo so sánh dạng text.
        """
        if "basic_metrics" not in self.results:
            self.compute_basic_metrics()
        
        df = self.results["basic_metrics"]
        
        print()
        print("=" * 100)
        print(" 📊 SO SÁNH TOÀN DIỆN CÁC MÔ HÌNH")
        print("=" * 100)
        
        print(f"\n  {'Model':<20s} | {'ROC-AUC':>8s} | {'PR-AUC':>8s} | {'F1':>8s} | {'Recall':>8s} | {'Precision':>8s} | {'Brier':>8s}")
        print("  " + "-" * 96)
        
        for _, row in df.iterrows():
            print(
                f"  {row['model']:<20s} | {row['roc_auc']:>8.4f} | {row['pr_auc']:>8.4f} | "
                f"{row['f1']:>8.4f} | {row['recall']:>8.4f} | {row['precision']:>8.4f} | {row['brier_score']:>8.4f}"
            )
        
        # Statistical tests
        if "statistical_tests" in self.results:
            tests = self.results["statistical_tests"]
            
            print(f"\n  🔬 STATISTICAL SIGNIFICANCE TESTS:")
            
            if tests.get("mcnemar"):
                print(f"\n  McNemar's Test:")
                for test in tests["mcnemar"]:
                    sig_marker = "✓" if test["significant"] else "✗"
                    print(f"    {test['model1']} vs {test['model2']}: χ²={test['chi2_statistic']:.3f}, p={test['p_value']:.4f} {sig_marker}")
            
            if tests.get("delong"):
                print(f"\n  DeLong's Test (ROC-AUC):")
                for test in tests["delong"]:
                    sig_marker = "✓" if test["significant"] else "✗"
                    print(f"    {test['model1']} vs {test['model2']}: ΔAUC={test['auc_diff']:.4f}, p={test['p_value']:.4f} {sig_marker}")
        
        # Best model
        best = df.loc[df["roc_auc"].idxmax()]
        print(f"\n  🏆 Model tốt nhất (theo ROC-AUC): {best['model']} (ROC-AUC={best['roc_auc']:.4f})")
        
        print()
        
        return df.to_string(index=False)
