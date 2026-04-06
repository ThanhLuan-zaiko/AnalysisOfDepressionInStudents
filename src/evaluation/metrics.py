"""
Evaluation Module - Comprehensive Metrics
Evaluate and compare models with multiple metrics

Usage:
    from src.evaluation import ModelEvaluator

    evaluator = ModelEvaluator()
    report = evaluator.generate_report(y_true, y_pred, model_name="XGBoost")
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation toolkit.
    """
    
    def __init__(self):
        self.results = {}
        logger.info("ModelEvaluator initialized")
    
    # ==========================================
    # 📊 CLASSIFICATION METRICS
    # ==========================================
    
    def classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        average: str = 'weighted'
    ) -> Dict:
        """
        Calculate all metrics for classification

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for ROC AUC)
            average: Averaging method ('micro', 'macro', 'weighted')

        Returns:
            Dict with full metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # ROC AUC if probabilities available
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except:
                metrics['roc_auc'] = 0.0
        
        # Details from confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_positive': int(tp),
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0
            })
        
        return metrics
    
    def detailed_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Detailed classification report per class
        """
        report = classification_report(
            y_true, y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        
        return report
    
    # ==========================================
    # 📈 REGRESSION METRICS
    # ==========================================
    
    def regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Calculate all metrics for regression
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        # Residuals analysis
        residuals = y_true - y_pred
        metrics.update({
            'residual_mean': float(np.mean(residuals)),
            'residual_std': float(np.std(residuals)),
            'residual_min': float(np.min(residuals)),
            'residual_max': float(np.max(residuals))
        })
        
        return metrics
    
    # ==========================================
    # 🏆 MODEL COMPARISON
    # ==========================================
    
    def compare_models(
        self,
        models_metrics: Dict[str, Dict]
    ) -> pl.DataFrame:
        """
        Compare multiple models

        Args:
            models_metrics: Dict {model_name: metrics_dict}

        Returns:
            Comparison DataFrame

        Usage:
            comparison = evaluator.compare_models({
                'XGBoost': xgb_metrics,
                'LightGBM': lgb_metrics,
                'Neural Network': nn_metrics
            })
        """
        rows = []
        for model_name, metrics in models_metrics.items():
            row = {'model': model_name}
            row.update(metrics)
            rows.append(row)
        
        df_comparison = pl.DataFrame(rows)
        
        # Log best model for each metric
        logger.info("Model Comparison:")
        print(df_comparison)
        
        return df_comparison
    
    def rank_models(
        self,
        models_metrics: Dict[str, Dict],
        primary_metric: str = 'f1',
        higher_is_better: bool = True
    ) -> List[str]:
        """
        Rank models by primary metric

        Returns:
            List of model names sorted by performance
        """
        model_scores = []
        for model_name, metrics in models_metrics.items():
            if primary_metric in metrics:
                model_scores.append((model_name, metrics[primary_metric]))
        
        # Sort
        model_scores.sort(key=lambda x: x[1], reverse=higher_is_better)
        
        ranked_models = [name for name, score in model_scores]
        
        logger.info(f"Model Ranking (by {primary_metric}):")
        for i, name in enumerate(ranked_models, 1):
            score = dict(model_scores)[name]
            logger.info(f"  {i}. {name}: {score:.4f}")
        
        return ranked_models
    
    # ==========================================
    # 📊 STATISTICAL SIGNIFICANCE
    # ==========================================
    
    def mcnemar_test(
        self,
        y_true: np.ndarray,
        y_pred_1: np.ndarray,
        y_pred_2: np.ndarray
    ) -> Dict:
        """
        McNemar's test: Check if 2 models have significant difference

        Returns:
            Dict with test statistic and p-value
        """
        from scipy import stats
        
        # Contingency table
        both_correct = np.sum((y_pred_1 == y_true) & (y_pred_2 == y_true))
        both_wrong = np.sum((y_pred_1 != y_true) & (y_pred_2 != y_true))
        only_1_correct = np.sum((y_pred_1 == y_true) & (y_pred_2 != y_true))
        only_2_correct = np.sum((y_pred_1 != y_true) & (y_pred_2 == y_true))
        
        # McNemar's test
        contingency = np.array([[only_1_correct, only_2_correct],
                                [only_2_correct, only_1_correct]])
        
        chi2, p_value = stats.chi2_contingency(contingency, correction=True)
        
        result = {
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'contingency_table': contingency.tolist()
        }
        
        logger.info(f"McNemar's Test:")
        logger.info(f"  Chi2: {chi2:.4f}")
        logger.info(f"  p-value: {p_value:.6f}")
        logger.info(f"  Significant: {p_value < 0.05}")
        
        return result
    
    # ==========================================
    # 📝 REPORT GENERATION
    # ==========================================
    
    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        y_proba: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate model evaluation report

        Returns:
            Report string
        """
        # Calculate metrics
        metrics = self.classification_metrics(y_true, y_pred, y_proba)

        # Create report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f" 📊 MODEL EVALUATION REPORT: {model_name}")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        report_lines.append("🎯 OVERALL METRICS:")
        report_lines.append(f"  Accuracy:  {metrics['accuracy']:.4f}")
        report_lines.append(f"  Precision: {metrics['precision']:.4f}")
        report_lines.append(f"  Recall:    {metrics['recall']:.4f}")
        report_lines.append(f"  F1 Score:  {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            report_lines.append(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        report_lines.append("")
        
        if 'confusion_matrix' in metrics:
            report_lines.append("📋 CONFUSION MATRIX:")
            cm = metrics['confusion_matrix']
            for row in cm:
                report_lines.append(f"  {row}")
            report_lines.append("")
        
        if 'sensitivity' in metrics:
            report_lines.append("🔍 BINARY CLASSIFICATION DETAILS:")
            report_lines.append(f"  TP: {metrics['true_positive']}, TN: {metrics['true_negative']}")
            report_lines.append(f"  FP: {metrics['false_positive']}, FN: {metrics['false_negative']}")
            report_lines.append(f"  Sensitivity: {metrics['sensitivity']:.4f}")
            report_lines.append(f"  Specificity: {metrics['specificity']:.4f}")
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        # Log and save file
        logger.info(f"Report generated for {model_name}")
        print(report)
        
        if save_path:
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved: {save_path}")
        
        return report
    
    # ==========================================
    # 💾 SAVE/LOAD RESULTS
    # ==========================================
    
    def save_results(self, model_name: str, metrics: Dict, filepath: str):
        """
        Save evaluation results
        """
        from pathlib import Path
        import json
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'model_name': model_name,
            'metrics': metrics
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved: {filepath}")
    
    def load_results(self, filepath: str) -> Dict:
        """
        Load evaluation results from file
        """
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.info(f"Results loaded: {filepath}")
        
        return results
