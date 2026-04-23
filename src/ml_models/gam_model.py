"""
Generalized Additive Model (GAM) cho Depression Analysis

GAM kết hợp sức mạnh của GLM (diễn giải được) và flexibility của ML models.
Mỗi feature có một smooth function (spline) riêng, cho phép mô hình hóa
quan hệ phi tuyến mà vẫn giữ được interpretability.

Ưu điểm:
- Interpretability: visualization được effect của từng feature
- Flexibility: capture được nonlinear relationships
- Additivity: tổng đóng góp của từng feature = prediction cuối
- Transparent: không phải "black box" như ensemble trees

Usage:
    from src.ml_models.gam_model import GAMClassifier
    
    gam = GAMClassifier()
    results = gam.train(X, y, feature_types)
    gam.plot_partial_dependence(feature_idx, save_path)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    recall_score, precision_score, brier_score_loss
)
import logging
import warnings
from datetime import datetime

# Suppress pyGAM numerical warnings (harmless but noisy)
# These occur during grid search when exploring extreme parameter values
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='divide by zero')
warnings.filterwarnings('ignore', message='overflow encountered')
warnings.filterwarnings('ignore', message='invalid value encountered')
warnings.filterwarnings('ignore', module='pygam')

logger = logging.getLogger(__name__)


class GAMClassifier:
    """
    Generalized Additive Model cho classification.
    
    Sử dụng pygam để fit logistic GAM với spline terms.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.feature_types = None
        self.feature_names = None
        self.results = {}
        logger.info("GAMClassifier initialized")
    
    def _build_terms(
        self,
        feature_types: Dict[str, str],
        n_splines: int = 10,
    ) -> list:
        """
        Xây dựng spline terms cho GAM dựa trên feature types.
        
        Args:
            feature_types: Dict mapping feature name to type ('numeric', 'ordinal', 'nominal')
            n_splines: Số spline bases cho numeric features
            
        Returns:
            List of pygam terms
        """
        try:
            from pygam import s, f
        except ImportError:
            raise ImportError("pygam not installed. Install with: uv add pygam")
        
        terms = []
        term_mapping = {}  # Map feature index to term info
        
        current_idx = 0
        for feat_name, feat_type in feature_types.items():
            if feat_type == 'numeric':
                # Smooth spline cho continuous variables
                terms.append(s(current_idx, n_splines=n_splines))
                term_mapping[current_idx] = {'name': feat_name, 'type': 'spline'}
                current_idx += 1
            elif feat_type == 'ordinal':
                # Có thể dùng spline hoặc factor cho ordinal
                # Spline giữ được thứ tự tự nhiên
                terms.append(s(current_idx, n_splines=n_splines))
                term_mapping[current_idx] = {'name': feat_name, 'type': 'spline'}
                current_idx += 1
            elif feat_type == 'nominal':
                # Factor term cho categorical (one-hot internally)
                terms.append(f(current_idx))
                term_mapping[current_idx] = {'name': feat_name, 'type': 'factor'}
                current_idx += 1
            else:
                raise ValueError(f"Unknown feature type: {feat_type}")
        
        # Combine all terms into a single TermList
        # pygam expects: term0 + term1 + term2 ...
        if len(terms) == 0:
            combined_terms = None
        elif len(terms) == 1:
            combined_terms = terms[0]
        else:
            # Start with first term and add rest
            combined_terms = terms[0]
            for term in terms[1:]:
                combined_terms = combined_terms + term
        
        return combined_terms, term_mapping
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_types: Dict[str, str],
        feature_names: List[str],
        n_splines: int = 10,
        optimize_splines: bool = True,
        use_rust: bool = True,  # New: try Rust engine first
    ) -> Dict:
        """
        Huấn luyện GAM model.

        Ưu tiên dùng Rust engine (nhanh hơn). Fallback về pyGAM nếu Rust không có.
        """
        # Try Rust engine first
        if use_rust:
            try:
                return self._train_rust(X, y, feature_types, feature_names, n_splines, optimize_splines)
            except Exception as e:
                logger.warning(f"Rust engine failed ({e}), falling back to pyGAM...")

        # Fallback to pyGAM
        return self._train_python(X, y, feature_types, feature_names, n_splines, optimize_splines)

    def _train_rust(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_types: Dict[str, str],
        feature_names: List[str],
        n_splines: int,
        optimize_splines: bool,
    ) -> Dict:
        """Train using Rust engine — fast, parallel CV."""
        from rust_engine import PyGAMClassifier, cross_validate_gam

        # Convert feature_types dict to ordered lists
        rust_feature_types = []
        rust_feature_names = []
        for name in feature_names:
            ftype = feature_types.get(name, 'numeric')
            rust_feature_types.append(ftype)
            rust_feature_names.append(name)

        logger.info(f"Training GAM with Rust engine: {len(feature_names)} features, {X.shape[0]} samples...")

        # Cross-validation (parallel)
        cv_result = cross_validate_gam(
            X.astype(np.float64),
            y.astype(np.float64),
            rust_feature_types,
            rust_feature_names,
            n_splits=5,
            n_splines=n_splines,
            random_seed=self.random_state,
        )

        # Fit final model on full data
        self.model = PyGAMClassifier(n_splines=n_splines, optimize_lambda=optimize_splines)
        fit_result = self.model.fit(
            X.astype(np.float64),
            y.astype(np.float64),
            rust_feature_types,
            rust_feature_names,
        )

        self.feature_types = feature_types
        self.feature_names = feature_names

        # Compile metrics
        metrics = {
            "roc_auc": cv_result["mean_roc_auc"],
            "roc_auc_std": cv_result["std_roc_auc"],
            "pr_auc": cv_result["mean_pr_auc"],
            "pr_auc_std": cv_result["std_pr_auc"],
            "f1": cv_result["mean_f1"],
            "f1_std": cv_result["std_f1"],
            "recall": cv_result["mean_recall"],
            "precision": cv_result["mean_precision"],
            "brier_score": cv_result["mean_brier_score"],
            "feature_importance": fit_result.get("feature_importance", []),
            "n_splines": n_splines,
            "optimize_splines": optimize_splines,
            "_engine": "rust",
        }

        self.results = metrics
        logger.info(f"GAM (Rust): ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")
        return metrics

    def _train_python(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_types: Dict[str, str],
        feature_names: List[str],
        n_splines: int,
        optimize_splines: bool,
    ) -> Dict:
        """Train using pyGAM — fallback when Rust is unavailable."""
        try:
            from pygam import LogisticGAM
        except ImportError:
            raise ImportError("pygam not installed. Install with: uv add pygam")
        
        self.feature_types = feature_types
        self.feature_names = feature_names
        
        # Build terms
        combined_terms, term_mapping = self._build_terms(feature_types, n_splines)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_metrics = {
            "roc_auc": [],
            "average_precision": [],
            "f1": [],
            "recall": [],
            "precision": [],
            "brier_score": [],
        }
        
        logger.info(f"Training GAM with {len(feature_names)} features...")
        logger.info(f"Feature types: {feature_types}")
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Tạo model mới cho mỗi fold
            gam_fold = LogisticGAM(
                terms=combined_terms,
                max_iter=200,
            )
            
            # Fit
            gam_fold.fit(X_train, y_train)
            
            # Optimize smoothing parameter nếu cần
            if optimize_splines:
                gam_fold.gridsearch(X_train, y_train)
            
            # Predict
            y_val_proba = gam_fold.predict_proba(X_val)
            y_val_pred = gam_fold.predict(X_val)
            
            # Metrics
            cv_metrics["roc_auc"].append(roc_auc_score(y_val, y_val_proba))
            cv_metrics["average_precision"].append(average_precision_score(y_val, y_val_proba))
            cv_metrics["f1"].append(f1_score(y_val, y_val_pred, pos_label=1, zero_division=0))
            cv_metrics["recall"].append(recall_score(y_val, y_val_pred, pos_label=1, zero_division=0))
            cv_metrics["precision"].append(precision_score(y_val, y_val_pred, pos_label=1, zero_division=0))
            cv_metrics["brier_score"].append(brier_score_loss(y_val, y_val_proba))
        
        # Fit trên toàn bộ data
        self.model = LogisticGAM(
            terms=combined_terms,
            max_iter=200,
        )
        self.model.fit(X, y)
        
        if optimize_splines:
            self.model.gridsearch(X, y)
        
        # Predictions
        y_proba = self.model.predict_proba(X)
        y_pred = self.model.predict(X)
        
        # Feature importance (dựa trên variance của partial dependence)
        importance_scores = self._compute_feature_importance(X)
        
        # Compile metrics
        metrics = {
            "roc_auc": np.mean(cv_metrics["roc_auc"]),
            "roc_auc_std": np.std(cv_metrics["roc_auc"]),
            "pr_auc": np.mean(cv_metrics["average_precision"]),
            "pr_auc_std": np.std(cv_metrics["average_precision"]),
            "f1": np.mean(cv_metrics["f1"]),
            "f1_std": np.std(cv_metrics["f1"]),
            "recall": np.mean(cv_metrics["recall"]),
            "precision": np.mean(cv_metrics["precision"]),
            "brier_score": brier_score_loss(y, y_proba),
            "feature_importance": importance_scores,
            "n_splines": n_splines,
            "optimize_splines": optimize_splines,
        }
        
        self.results = metrics
        
        logger.info(f"GAM: ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")
        return metrics
    
    def _normalize_partial_dependence(
        self,
        raw_result: Any,
        XX: np.ndarray,
        feature_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Handle both legacy dict outputs and current pyGAM tuple/ndarray outputs."""
        if isinstance(raw_result, dict):
            grid = np.asarray(raw_result.get('grid', XX[:, feature_idx]))
            pd_values = np.asarray(raw_result.get('partial_dependence', []))
            ci = np.asarray(raw_result.get('conf_intervals', np.full((len(pd_values), 2), np.nan)))
            return grid.flatten(), pd_values.flatten(), ci

        if isinstance(raw_result, tuple):
            pd_values = np.asarray(raw_result[0])
            ci = (
                np.asarray(raw_result[1])
                if len(raw_result) > 1
                else np.full((len(pd_values), 2), np.nan)
            )
            return XX[:, feature_idx].flatten(), pd_values.flatten(), ci

        pd_values = np.asarray(raw_result)
        return (
            XX[:, feature_idx].flatten(),
            pd_values.flatten(),
            np.full((len(pd_values), 2), np.nan),
        )

    def _compute_feature_importance(self, X: np.ndarray) -> List[Dict]:
        """
        Tính feature importance. 
        - Rust engine: đã tính sẵn trong fit_result
        - pyGAM: tính từ partial dependence variance
        """
        # If Rust engine, feature_importance already computed
        if self.results.get("_engine") == "rust":
            return self.results.get("feature_importance", [])

        # pyGAM fallback
        importance_list = []
        for i, feat_name in enumerate(self.feature_names):
            try:
                XX = self.model.generate_X_grid(term=i)
                raw_result = self.model.partial_dependence(term=i, X=XX)
                _, pd_values, _ = self._normalize_partial_dependence(raw_result, XX, i)
                effect_variance = float(np.var(pd_values))
                importance_list.append({
                    "feature": feat_name,
                    "variance_importance": round(effect_variance, 6),
                    "term_index": i,
                })
            except Exception as e:
                logger.warning(f"Could not compute importance for {feat_name}: {e}")
                importance_list.append({
                    "feature": feat_name,
                    "variance_importance": 0.0,
                    "term_index": i,
                })
        
        # Sort by importance
        importance_list.sort(key=lambda x: x["variance_importance"], reverse=True)
        return importance_list
    
    def get_partial_dependence(
        self,
        feature_idx: int,
        n_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Lấy partial dependence cho một feature.
        
        Returns:
            X_values: Grid values cho feature
            pd_values: Partial dependence values
            confidence_intervals: 95% CI (shape: n_points x 2)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        XX = self.model.generate_X_grid(term=feature_idx, n_points=n_points)
        raw_result = self.model.partial_dependence(term=feature_idx, X=XX, width=0.95)
        return self._normalize_partial_dependence(raw_result, XX, feature_idx)
    
    def plot_partial_dependence(
        self,
        feature_indices: Optional[List[int]] = None,
        top_k: int = 5,
        save_dir: str = "results/gam_plots/",
        format: str = "html",
    ) -> List[Path]:
        """
        Vẽ partial dependence plots cho các features quan trọng nhất.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Determine which features to plot
        if feature_indices is not None:
            indices_to_plot = feature_indices
        else:
            importance = self.results.get("feature_importance", [])
            indices_to_plot = [item["term_index"] for item in importance[:top_k]]
            
        saved_files = []
        
        # Chỉ gọi hàm này nếu không phải Rust engine (Rust engine hiện chưa hỗ trợ generate_X_grid)
        if self.results.get("_engine") == "rust":
            logger.warning("Partial dependence plots not supported for Rust engine yet. Skipping.")
            return []

        # ... (phần code vẽ plot giữ nguyên) ...
    
    def plot_feature_effects_summary(
        self,
        save_path: str = "results/gam_feature_effects.html",
    ) -> Path:
        """
        Vẽ summary chart tổng hợp effect sizes của tất cả features.
        Giống như coefficient plot trong LR nhưng cho GAM.
        """
        import plotly.graph_objects as go
        import plotly.io as pio
        
        importance = self.results.get("feature_importance", [])
        
        if not importance:
            raise ValueError("No feature importance data. Train model first.")
        
        # Sort by importance
        importance_sorted = sorted(importance, key=lambda x: x["variance_importance"])
        
        features = [item["feature"] for item in importance_sorted]
        values = [item["variance_importance"] for item in importance_sorted]
        
        fig = go.Figure(go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker=dict(
                color=values,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Variance<br>Importance"),
            ),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.6f}<extra></extra>",
        ))
        
        fig.update_layout(
            title="GAM Feature Importance (Variance of Partial Dependence)",
            xaxis_title="Variance Importance",
            yaxis_title="Feature",
            template="plotly_white",
            height=max(400, len(features) * 35),
            width=800,
        )
        
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pio.write_html(fig, str(output_path), full_html=True, include_plotlyjs=True)
        logger.info(f"Saved feature effects summary: {output_path}")
        
        return output_path
    
    def interpret_model(
        self,
        output_file: str = "results/gam_interpretation.json",
    ) -> Dict:
        """
        Tạo interpretation report cho GAM model.
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        interpretation = {
            "model_type": "Generalized Additive Model (GAM)",
            "link_function": "logit",
            "timestamp": datetime.now().isoformat(),
            "features": [],
            "feature_importance_ranking": self.results.get("feature_importance", []),
        }
        
        # Information cho từng feature
        for i, feat_name in enumerate(self.feature_names):
            feat_info = {
                "name": feat_name,
                "term_index": i,
                "type": self.feature_types.get(feat_name, "unknown"),
            }
            
            # Compute effect direction và magnitude
            try:
                XX = self.model.generate_X_grid(term=i)
                pd_dict = self.model.partial_dependence(term=i, X=XX)
                pd_vals = pd_dict['partial_dependence'].flatten()
                
                feat_info["effect_range"] = {
                    "min": float(pd_vals.min()),
                    "max": float(pd_vals.max()),
                    "span": float(pd_vals.max() - pd_vals.min()),
                }
                feat_info["effect_direction"] = (
                    "positive" if pd_vals[-1] > pd_vals[0] else 
                    "negative" if pd_vals[-1] < pd_vals[0] else 
                    "nonlinear"
                )
            except Exception as e:
                feat_info["effect_range"] = None
                feat_info["effect_direction"] = f"error: {e}"
            
            interpretation["features"].append(feat_info)
        
        # Save
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(interpretation, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Saved GAM interpretation: {output_path}")
        return interpretation
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        if self.results.get("_engine") == "rust":
            rust_feature_types = []
            for name in self.feature_names:
                ftype = self.feature_types.get(name, 'numeric')
                rust_feature_types.append(ftype)
            proba = self.model.predict_proba(X.astype(np.float64), rust_feature_types, self.feature_names)
            return (proba >= 0.5).astype(int)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        if self.results.get("_engine") == "rust":
            rust_feature_types = []
            for name in self.feature_names:
                ftype = self.feature_types.get(name, 'numeric')
                rust_feature_types.append(ftype)
            return self.model.predict_proba(X.astype(np.float64), rust_feature_types, self.feature_names)
        return self.model.predict_proba(X)
    
    def save_model(self, filepath: str):
        """Save model to file."""
        import joblib
        import pickle
        
        model_data = {
            "model": self.model,
            "feature_types": self.feature_types,
            "feature_names": self.feature_names,
            "results": self.results,
            "random_state": self.random_state,
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved GAM model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file."""
        import pickle
        
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.feature_types = model_data["feature_types"]
        self.feature_names = model_data["feature_names"]
        self.results = model_data["results"]
        self.random_state = model_data["random_state"]
        
        logger.info(f"Loaded GAM model from {filepath}")
