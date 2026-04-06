"""
Machine Learning Models Module - XGBoost/LightGBM
Train depression prediction models

Usage:
    from src.ml_models import DepressionPredictor

    predictor = DepressionPredictor()
    model = predictor.train_xgboost(X_train, y_train)
    metrics = predictor.evaluate(model, X_test, y_test)
"""

import polars as pl
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from typing import Optional, Dict, Tuple, Union
from pathlib import Path
import joblib
import logging
import numpy as np

logger = logging.getLogger(__name__)


class DepressionPredictor:
    """
    ML model training toolkit for depression prediction.
    Supports XGBoost and LightGBM with GPU acceleration.
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.models_dir = Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DepressionPredictor initialized (GPU={use_gpu})")
    
    # ==========================================
    # 📊 DATA PREPARATION
    # ==========================================
    
    def prepare_data(
        self,
        df: pl.DataFrame,
        target_col: str = "depression_score",
        feature_cols: Optional[list] = None,
        threshold: int = 16,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple:
        """
        Prepare data for ML (train/test split)

        Args:
            df: DataFrame
            target_col: Target column (score or binary label)
            feature_cols: Feature columns (default: all numeric)
            threshold: Depression classification threshold (default: 16)
            test_size: Test set ratio
            random_state: Seed for reproducibility

        Returns:
            X_train, X_test, y_train, y_test
        """
        # Auto-select feature columns
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]

        # Encode categorical columns
        df_encoded = df.clone()
        for col in df.columns:
            if df[col].dtype == pl.String:
                # Label encoding for categorical
                unique_vals = df[col].unique().drop_nulls()
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                df_encoded = df_encoded.with_columns(
                    pl.col(col).replace(mapping).cast(pl.Int64)
                )
        
        # Features and target
        X = df_encoded.select(feature_cols)
        
        # If target is continuous, convert to binary
        if df[target_col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            y = df_encoded.with_columns(
                (pl.col(target_col) >= threshold).cast(pl.Int64).alias("target")
            ).select("target")
        else:
            y = df_encoded.select(target_col)
        
        # Convert to numpy
        X_np = X.to_numpy()
        y_np = y.to_numpy().ravel()
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np,
            test_size=test_size,
            random_state=random_state,
            stratify=y_np if len(np.unique(y_np)) == 2 else None
        )
        
        logger.info(f"Data prepared: {X_train.shape[0]} train, {X_test.shape[0]} test")
        logger.info(f"Features: {len(feature_cols)}, Target distribution: {np.bincount(y_train)}")
        
        return X_train, X_test, y_train, y_test
    
    # ==========================================
    # 🌲 XGBOOST
    # ==========================================
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: Optional[Dict] = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        use_gpu: Optional[bool] = None
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost classifier

        Args:
            use_gpu: Override self.use_gpu
        """
        if use_gpu is None:
            use_gpu = self.use_gpu
        
        if params is None:
            params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'tree_method': 'gpu_hist' if use_gpu else 'hist',
                'eval_metric': 'logloss',
                'random_state': 42,
                'use_label_encoder': False
            }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        logger.info(f"XGBoost trained: {n_estimators} estimators, depth={max_depth}")
        logger.info(f"GPU: {use_gpu}")
        
        return model
    
    def train_xgboost_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        use_gpu: Optional[bool] = None
    ) -> xgb.XGBRegressor:
        """
        Train XGBoost regressor (for continuous target)
        """
        if use_gpu is None:
            use_gpu = self.use_gpu
        
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            tree_method='gpu_hist' if use_gpu else 'hist',
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        logger.info(f"XGBoost Regressor trained: {n_estimators} estimators")
        
        return model
    
    # ==========================================
    # 💡 LIGHTGBM
    # ==========================================
    
    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: Optional[Dict] = None,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        use_gpu: Optional[bool] = None
    ) -> lgb.LGBMClassifier:
        """
        Train LightGBM classifier
        """
        if use_gpu is None:
            use_gpu = self.use_gpu
        
        if params is None:
            params = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'num_leaves': num_leaves,
                'device': 'gpu' if use_gpu else 'cpu',
                'random_state': 42,
                'verbose': -1
            }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        logger.info(f"LightGBM trained: {n_estimators} estimators, leaves={num_leaves}")
        logger.info(f"GPU: {use_gpu}")
        
        return model
    
    def train_lightgbm_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        use_gpu: Optional[bool] = None
    ) -> lgb.LGBMRegressor:
        """
        Train LightGBM regressor
        """
        if use_gpu is None:
            use_gpu = self.use_gpu
        
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            device='gpu' if use_gpu else 'cpu',
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        logger.info(f"LightGBM Regressor trained: {n_estimators} estimators")
        
        return model
    
    # ==========================================
    # 📈 EVALUATION
    # ==========================================
    
    def evaluate(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_type: str = "classifier"
    ) -> Dict:
        """
        Evaluate model with multiple metrics

        Args:
            model: trained model
            X_test: Test features
            y_test: Test labels
            model_type: "classifier" or "regressor"

        Returns:
            Dict with metrics
        """
        y_pred = model.predict(X_test)
        
        if model_type == "classifier":
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            }
            
            # ROC AUC (if predict_proba available)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
            
            logger.info(f"Model Evaluation:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1: {metrics['f1']:.4f}")
            if 'roc_auc' in metrics:
                logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        
        elif model_type == "regressor":
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics = {
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred)
            }
            
            logger.info(f"Model Evaluation:")
            logger.info(f"  RMSE: {metrics['rmse']:.4f}")
            logger.info(f"  MAE: {metrics['mae']:.4f}")
            logger.info(f"  R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def cross_validate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = "accuracy"
    ) -> Dict:
        """
        Cross-validation to assess model stability

        Args:
            model: Unfitted model
            X: Features
            y: Labels
            cv: Number of folds
            scoring: Scoring metric

        Returns:
            Dict with mean, std of scores
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        result = {
            "mean": scores.mean(),
            "std": scores.std(),
            "scores": scores.tolist(),
            "cv": cv,
            "scoring": scoring
        }
        
        logger.info(f"Cross-validation ({cv} folds, {scoring}):")
        logger.info(f"  Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return result
    
    # ==========================================
    # 🔍 FEATURE IMPORTANCE
    # ==========================================
    
    def get_feature_importance(
        self,
        model,
        feature_names: list,
        top_n: int = 10
    ) -> pl.DataFrame:
        """
        Get feature importance

        Args:
            model: Trained model
            feature_names: Feature names
            top_n: Number of most important features

        Returns:
            DataFrame with feature names and importance scores
        """
        importance = model.feature_importances_
        
        df_importance = pl.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort("importance", descending=True).head(top_n)
        
        logger.info(f"Top {top_n} important features:")
        for row in df_importance.iter_rows():
            logger.info(f"  {row[0]}: {row[1]:.4f}")
        
        return df_importance
    
    # ==========================================
    # 💾 SAVE/LOAD MODELS
    # ==========================================
    
    def save_model(self, model, filename: str) -> Path:
        """
        Save model to file
        """
        filepath = self.models_dir / filename
        joblib.dump(model, filepath)
        logger.info(f"Model saved: {filepath}")
        return filepath
    
    def load_model(self, filename: str):
        """
        Load model from file
        """
        filepath = self.models_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = joblib.load(filepath)
        logger.info(f"Model loaded: {filepath}")
        
        return model
    
    # ==========================================
    # 🎯 PREDICTION
    # ==========================================
    
    def predict(
        self,
        model,
        X: np.ndarray,
        return_proba: bool = False
    ) -> Union[np.ndarray, Tuple]:
        """
        Predict with model

        Args:
            model: Trained model
            X: Features
            return_proba: Return probabilities instead of predictions

        Returns:
            Predictions or (predictions, probabilities)
        """
        predictions = model.predict(X)
        
        if return_proba and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)
            return predictions, probabilities
        
        return predictions
