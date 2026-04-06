"""
Explainable AI Module - SHAP-based
Explain model predictions with SHAP values

Usage:
    from src.ml_models import SHAPExplainer

    explainer = SHAPExplainer()
    explanation = explainer.explain_model(model, X_train, X_test)
"""

import shap
import numpy as np
import polars as pl
from typing import Dict, Optional, List, Union
import logging

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    Explain model predictions with SHAP (SHapley Additive exPlanations).
    Helps understand why the model makes a particular prediction.
    """
    
    def __init__(self):
        logger.info("SHAPExplainer initialized")
    
    # ==========================================
    # 🎯 TREE-BASED MODELS (XGBoost, LightGBM)
    # ==========================================
    
    def explain_tree_model(
        self,
        model,
        X_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
        max_display: int = 20
    ) -> Dict:
        """
        Explain tree-based model (XGBoost, LightGBM)

        Args:
            model: Trained tree model
            X_train: Training data (for background distribution)
            X_test: Test data to explain
            feature_names: Feature names
            max_display: Number of important features to show

        Returns:
            Dict with shap_values, summary_plot, dependence_plots
        """
        logger.info("Explaining tree model with SHAP...")
        
        # Create explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test)
        
        # Summary plot (global importance)
        fig_summary = shap.summary_plot(
            shap_values,
            X_test,
            feature_names=feature_names,
            show=False,
            max_display=max_display,
            plot_type="dot"
        )
        
        # Bar plot
        fig_bar = shap.summary_plot(
            shap_values,
            X_test,
            feature_names=feature_names,
            show=False,
            max_display=max_display,
            plot_type="bar"
        )
        
        # Force plot cho first prediction
        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[1]  # Binary classification
        
        fig_force = shap.force_plot(
            expected_value,
            shap_values[0, :],
            X_test[0, :],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        
        logger.info(f"SHAP explanation completed for {X_test.shape[1]} features")
        
        return {
            'shap_values': shap_values,
            'expected_value': expected_value,
            'explainer': explainer,
            'summary_plot': fig_summary,
            'bar_plot': fig_bar,
            'force_plot': fig_force
        }
    
    # ==========================================
    # 📊 DEPENDENCE PLOTS
    # ==========================================
    
    def plot_dependence(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature: str,
        feature_names: List[str],
        interaction_feature: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Dependence plot: SHAP value of a feature vs its value

        Args:
            feature: Feature to plot
            interaction_feature: Feature to color by (optional)
        """
        feature_idx = feature_names.index(feature) if feature in feature_names else None
        
        if feature_idx is None:
            raise ValueError(f"Feature '{feature}' not found in feature_names")
        
        fig = shap.dependence_plot(
            feature_idx,
            shap_values,
            X,
            feature_names=feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        
        if save_path:
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dependence plot saved: {save_path}")
        
        return fig
    
    def plot_all_dependence(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        top_n: int = 10,
        save_dir: Optional[str] = None
    ) -> List:
        """
        Plot dependence plots for top N features
        """
        # Get top features by mean |SHAP|
        mean_abs_shap = np.abs(shap_values).mean(0)
        top_indices = mean_abs_shap.argsort()[-top_n:][::-1]
        
        figs = []
        for idx in top_indices:
            feature = feature_names[idx]
            
            save_path = None
            if save_dir:
                save_path = f"{save_dir}/dependence_{feature}.png"
            
            fig = self.plot_dependence(
                shap_values, X, feature, feature_names,
                save_path=save_path
            )
            figs.append(fig)
        
        return figs
    
    # ==========================================
    # 🔍 INDIVIDUAL PREDICTION EXPLANATION
    # ==========================================
    
    def explain_prediction(
        self,
        model,
        X_sample: np.ndarray,
        X_background: np.ndarray,
        feature_names: List[str],
        sample_idx: int = 0
    ) -> Dict:
        """
        Explain individual prediction

        Args:
            X_sample: Sample to explain
            X_background: Background data
            feature_names: Feature names
            sample_idx: Sample index

        Returns:
            Dict with explanation for that prediction
        """
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Base value (expected output)
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1]
        
        # SHAP values cho sample
        if isinstance(shap_values, list):
            shap_values_sample = shap_values[1][sample_idx]
        else:
            shap_values_sample = shap_values[sample_idx]
        
        # Top features influencing prediction
        top_features_idx = np.abs(shap_values_sample).argsort()[::-1][:5]
        top_features = [(feature_names[i], shap_values_sample[i]) for i in top_features_idx]
        
        # Force plot
        fig_force = shap.force_plot(
            base_value,
            shap_values_sample,
            X_sample[sample_idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        
        logger.info(f"Explained prediction for sample {sample_idx}")
        logger.info(f"Top 5 features: {top_features}")
        
        return {
            'sample_idx': sample_idx,
            'base_value': base_value,
            'shap_values': shap_values_sample,
            'top_features': top_features,
            'force_plot': fig_force
        }
    
    # ==========================================
    # 📈 GLOBAL INTERPRETATION
    # ==========================================
    
    def global_interpretation(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str]
    ) -> pl.DataFrame:
        """
        Global interpretation of model

        Returns:
            DataFrame with feature importance, impact direction
        """
        # Mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Sort features by importance
        sorted_idx = mean_abs_shap.argsort()[::-1]
        
        interpretation = pl.DataFrame({
            'feature': [feature_names[i] for i in sorted_idx],
            'importance': mean_abs_shap[sorted_idx],
            'mean_shap': shap_values[:, sorted_idx].mean(axis=0),
            'std_shap': shap_values[:, sorted_idx].std(axis=0)
        })
        
        logger.info("Global interpretation:")
        print(interpretation.head(10))
        
        return interpretation
    
    # ==========================================
    # 💾 SAVE/LOAD
    # ==========================================
    
    def save_explanation(
        self,
        explanation: Dict,
        filepath: str
    ):
        """
        Save explanation
        """
        from pathlib import Path
        import joblib
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Only save SHAP values and metadata, not figures
        save_data = {
            'shap_values': explanation['shap_values'],
            'expected_value': explanation['expected_value'],
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Explanation saved: {filepath}")
    
    # ==========================================
    # 🎯 QUICK EXPLAIN
    # ==========================================
    
    def quick_explain(
        self,
        model,
        X_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: List[str],
        sample_idx: int = 0
    ) -> Dict:
        """
        Quick explanation pipeline

        Returns:
            Dict with all explanations
        """
        # Explain model
        model_explanation = self.explain_tree_model(
            model, X_train, X_test, feature_names
        )
        
        # Explain individual prediction
        sample_explanation = self.explain_prediction(
            model, X_test, X_train, feature_names, sample_idx
        )
        
        # Global interpretation
        global_interp = self.global_interpretation(
            model_explanation['shap_values'],
            X_test,
            feature_names
        )
        
        return {
            'model': model_explanation,
            'sample': sample_explanation,
            'global': global_interp
        }
