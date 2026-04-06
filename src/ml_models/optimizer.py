"""
Hyperparameter Optimization Module - Optuna-based
Auto-tune model parameters for best performance

Usage:
    from src.ml_models import HyperparameterOptimizer

    optimizer = HyperparameterOptimizer()
    best_params = optimizer.optimize_xgboost(X_train, y_train, n_trials=100)
"""

import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Auto-tune hyperparameters with Optuna.
    Finds the best params automatically instead of manual tuning.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.best_trials = {}
        logger.info("HyperparameterOptimizer initialized")
    
    # ==========================================
    # 🌲 XGBOOST OPTIMIZATION
    # ==========================================
    
    def _xgboost_objective(self, trial: optuna.Trial, X, y) -> float:
        """
        Objective function for XGBoost optimization
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'tree_method': 'gpu_hist' if trial.suggest_categorical('use_gpu', [True, False]) else 'hist',
            'random_state': self.random_state,
            'eval_metric': 'logloss'
        }
        
        model = xgb.XGBClassifier(**params)
        
        # Cross-validation score
        scores = cross_val_score(model, X, y, cv=5, scoring='f1', n_jobs=1)
        
        return scores.mean()
    
    def optimize_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 100,
        timeout: int = 3600,
        study_name: str = "xgboost_optimization",
        show_progress: bool = True
    ) -> Dict:
        """
        Optimize XGBoost hyperparameters

        Args:
            X: Training features
            y: Target labels
            n_trials: Number of trials (more is better, but takes longer)
            timeout: Timeout (seconds)
            study_name: Study name for logging
            show_progress: Show progress bar

        Returns:
            Dict with best_params, best_score, and study object
        """
        logger.info(f"Starting XGBoost optimization: {n_trials} trials")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Optimize
        study.optimize(
            lambda trial: self._xgboost_objective(trial, X, y),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress
        )
        
        # Best params
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Best F1 Score: {best_score:.4f}")
        logger.info(f"Best Parameters:")
        for param, value in best_params.items():
            logger.info(f"  {param}: {value}")
        
        # Save trial
        self.best_trials['xgboost'] = {
            'params': best_params,
            'score': best_score,
            'study': study
        }
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'study': study,
            'n_trials_completed': len(study.trials)
        }
    
    # ==========================================
    # 💡 LIGHTGBM OPTIMIZATION
    # ==========================================
    
    def _lightgbm_objective(self, trial: optuna.Trial, X, y) -> float:
        """
        Objective function for LightGBM optimization
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 200),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'device': 'gpu' if trial.suggest_categorical('use_gpu', [True, False]) else 'cpu',
            'random_state': self.random_state,
            'verbose': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        
        # Cross-validation score
        scores = cross_val_score(model, X, y, cv=5, scoring='f1', n_jobs=1)
        
        return scores.mean()
    
    def optimize_lightgbm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 100,
        timeout: int = 3600,
        study_name: str = "lightgbm_optimization",
        show_progress: bool = True
    ) -> Dict:
        """
        Optimize LightGBM hyperparameters
        """
        logger.info(f"Starting LightGBM optimization: {n_trials} trials")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Optimize
        study.optimize(
            lambda trial: self._lightgbm_objective(trial, X, y),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress
        )
        
        # Best params
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Best F1 Score: {best_score:.4f}")
        logger.info(f"Best Parameters:")
        for param, value in best_params.items():
            logger.info(f"  {param}: {value}")
        
        # Save trial
        self.best_trials['lightgbm'] = {
            'params': best_params,
            'score': best_score,
            'study': study
        }
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'study': study,
            'n_trials_completed': len(study.trials)
        }
    
    # ==========================================
    # 📊 OPTIMIZATION VISUALIZATION
    # ==========================================
    
    def plot_optimization_history(
        self,
        study: optuna.Study,
        save_path: Optional[str] = None
    ):
        """
        Plot optimization history
        """
        from optuna.visualization import plot_optimization_history, plot_param_importances
        
        # Optimization history
        fig_history = plot_optimization_history(study)
        
        # Parameter importances
        try:
            fig_importance = plot_param_importances(study)
        except:
            fig_importance = None
        
        if save_path:
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save as HTML
            fig_history.write_html(f"{save_path}_history.html")
            if fig_importance:
                fig_importance.write_html(f"{save_path}_importance.html")
            
            logger.info(f"Optimization plots saved: {save_path}")
        
        return fig_history, fig_importance
    
    def plot_parallel_coordinate(
        self,
        study: optuna.Study,
        save_path: Optional[str] = None
    ):
        """
        Plot parallel coordinate plot to see relationships between params
        """
        from optuna.visualization import plot_parallel_coordinate
        
        fig = plot_parallel_coordinate(study)
        
        if save_path:
            fig.write_html(f"{save_path}_parallel.html")
            logger.info(f"Parallel coordinate plot saved: {save_path}")
        
        return fig
    
    # ==========================================
    # 💾 SAVE/LOAD STUDIES
    # ==========================================
    
    def save_study(self, study: optuna.Study, filepath: str):
        """
        Save Optuna study for reuse
        """
        from pathlib import Path
        import joblib
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(study, filepath)
        
        logger.info(f"Study saved: {filepath}")
    
    def load_study(self, filepath: str) -> optuna.Study:
        """
        Load Optuna study from file
        """
        import joblib
        
        study = joblib.load(filepath)
        logger.info(f"Study loaded: {filepath}")
        
        return study
    
    # ==========================================
    # 🎯 QUICK OPTIMIZATION
    # ==========================================
    
    def quick_optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "xgboost",
        n_trials: int = 50
    ) -> Dict:
        """
        Quick optimization with fewer trials (for testing)
        """
        if model_type == "xgboost":
            return self.optimize_xgboost(X, y, n_trials=n_trials, timeout=600)
        elif model_type == "lightgbm":
            return self.optimize_lightgbm(X, y, n_trials=n_trials, timeout=600)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    # ==========================================
    # 📊 COMPARISON
    # ==========================================
    
    def compare_default_vs_optimized(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "xgboost",
        n_trials: int = 100
    ) -> Dict:
        """
        Compare default params vs optimized params
        """
        from sklearn.model_selection import cross_val_score
        
        # Run optimization
        if model_type == "xgboost":
            opt_result = self.optimize_xgboost(X, y, n_trials=n_trials)
            
            # Default model
            default_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            )
            
            # Optimized model
            opt_model = xgb.XGBClassifier(
                **opt_result['best_params'],
                random_state=self.random_state
            )
            
        elif model_type == "lightgbm":
            opt_result = self.optimize_lightgbm(X, y, n_trials=n_trials)
            
            # Default model
            default_model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                random_state=self.random_state
            )
            
            # Optimized model
            opt_model = lgb.LGBMClassifier(
                **opt_result['best_params'],
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Cross-validate both
        default_scores = cross_val_score(default_model, X, y, cv=5, scoring='f1')
        opt_scores = cross_val_score(opt_model, X, y, cv=5, scoring='f1')
        
        comparison = {
            'default': {
                'mean': default_scores.mean(),
                'std': default_scores.std(),
                'scores': default_scores.tolist()
            },
            'optimized': {
                'mean': opt_scores.mean(),
                'std': opt_scores.std(),
                'scores': opt_scores.tolist()
            },
            'improvement': opt_scores.mean() - default_scores.mean(),
            'improvement_percent': ((opt_scores.mean() - default_scores.mean()) / default_scores.mean()) * 100
        }
        
        logger.info(f"Default vs Optimized:")
        logger.info(f"  Default: {comparison['default']['mean']:.4f}")
        logger.info(f"  Optimized: {comparison['optimized']['mean']:.4f}")
        logger.info(f"  Improvement: {comparison['improvement']:.4f} ({comparison['improvement_percent']:.2f}%)")
        
        return comparison
