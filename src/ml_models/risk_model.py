"""
Risk Modeling Module - Phân tầng nguy cơ trầm cảm

Theo kế hoạch khoa học:
  - Model 0: Dummy baseline (luôn dự đoán lớp đa số)
  - Model 1: Penalized Logistic Regression (trung tâm, giải thích được)
  - Model 2: GAM - Generalized Additive Model (linh hoạt, vẫn diễn giải được)
  - Model 3: CatBoost (dự báo bổ sung, mạnh hơn)

Ethical note: Đây là công cụ HỖ TRỢ sàng lọc, KHÔNG thay thế đánh giá lâm sàng.

Usage:
    from src.ml_models.risk_model import DepressionRiskModeler

    modeler = DepressionRiskModeler()
    results = modeler.run_full_pipeline(df, include_suicidal=True)
"""

import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (recall_score, precision_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    f1_score, classification_report,
    confusion_matrix, RocCurveDisplay,
    PrecisionRecallDisplay, ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import logging
import json
from datetime import datetime
from src.utils import Timer

logger = logging.getLogger(__name__)

# Các cột bị loại khỏi mô hình
EXCLUDED_COLUMNS = [
    "id",
    "Profession",
    "Work Pressure",
    "Job Satisfaction",
]

# Tên biến ordinal theo thứ tự logic
ORDINAL_COLUMNS = [
    "Academic Pressure",
    "Study Satisfaction",
    "Financial Stress",
]

# Tên biến numeric
NUMERIC_COLUMNS = [
    "Age",
    "CGPA",
    "Work/Study Hours",
]

# Tên biến nominal (categorical không thứ tự)
NOMINAL_COLUMNS = [
    "Gender",
    "City",
    "Degree",
    "Sleep Duration",
    "Dietary Habits",
    "Family History of Mental Illness",
]

# Biến nhạy cảm cần kiểm tra fairness
SENSITIVE_COLUMNS = [
    "Have you ever had suicidal thoughts ?",
]


class DepressionRiskModeler:
    """
    Mô hình phân tầng nguy cơ trầm cảm.

    Thiết kế theo nguyên tắc:
    - Logistic Regression làm mô hình trung tâm (giải thích được)
    - CatBoost làm mô hình dự báo bổ sung
    - Đánh giá bằng ROC-AUC, PR-AUC, Brier score, calibration
    - Không chỉ dùng accuracy (vì class imbalance)
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.preprocessors = {}
        logger.info("DepressionRiskModeler initialized")

    # ==========================================
    # 🔧 DATA PREPARATION
    # ==========================================

    def prepare_features(
        self,
        df: pl.DataFrame,
        include_suicidal: bool = True,
        target_col: str = "Depression",
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Chuẩn bị feature matrix và target vector.

        Xử lý:
        - Loại cột không cần thiết
        - Mã hóa biến nominal (one-hot)
        - Mã hóa biến ordinal (giữ thứ tự)
        - Chuẩn hóa biến numeric

        Returns:
            X, y, feature_names
        """
        df_work = df.clone()

        # Loại cột excluded
        cols_to_drop = [c for c in EXCLUDED_COLUMNS if c in df_work.columns]
        df_work = df_work.drop(cols_to_drop)

        # Loại biến suicidal nếu không include
        if not include_suicidal:
            for col in SENSITIVE_COLUMNS:
                if col in df_work.columns:
                    df_work = df_work.drop(col)

        # Target
        if target_col not in df_work.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        y = df_work[target_col].to_numpy()

        # Loại target khỏi features
        df_features = df_work.drop(target_col)

        # Xử lý missing values (rất ít nên dùng median/mode)
        cols_to_drop = []
        for col in df_features.columns:
            null_count = df_features[col].null_count()
            if null_count == 0:
                continue

            if null_count == df_features.height:
                # Cột 100% missing — không thể impute, phải loại
                cols_to_drop.append(col)
                logger.warning(f"⚠️  Cột '{col}' 100% missing — sẽ loại khỏi phân tích")
                continue

            if df_features[col].dtype in [pl.Int64, pl.Float64]:
                df_features = df_features.with_columns(
                    pl.col(col).fill_null(pl.col(col).median())
                )
            else:
                mode_val = df_features[col].mode().first()
                if mode_val is not None:
                    df_features = df_features.with_columns(
                        pl.col(col).fill_null(mode_val)
                    )
                else:
                    cols_to_drop.append(col)
                    logger.warning(f"⚠️  Cột '{col}' mode trả về null — sẽ loại")

        if cols_to_drop:
            df_features = df_features.drop(cols_to_drop)

        # === Build feature matrix ===
        feature_frames = []
        feature_names = []

        # 1. Numeric: chuẩn hóa
        numeric_cols = [c for c in NUMERIC_COLUMNS if c in df_features.columns]
        if numeric_cols:
            numeric_data = df_features.select(numeric_cols).to_pandas()
            scaler = StandardScaler()
            numeric_scaled = scaler.fit_transform(numeric_data)
            feature_frames.append(pd.DataFrame(numeric_scaled, columns=numeric_cols))
            feature_names.extend(numeric_cols)
            self.preprocessors["scaler"] = scaler

        # 2. Ordinal: mã hóa thứ tự (0, 1, 2, ...)
        ordinal_cols = [c for c in ORDINAL_COLUMNS if c in df_features.columns]
        if ordinal_cols:
            for col in ordinal_cols:
                values = df_features[col].to_pandas()
                # Tự động phát hiện levels
                unique_sorted = sorted(values.unique())
                ordinal_map = {v: i for i, v in enumerate(unique_sorted)}
                encoded = values.map(ordinal_map).values
                feature_frames.append(pd.DataFrame(encoded, columns=[col]))
                feature_names.append(col)
            self.preprocessors["ordinal_map"] = {col: sorted(df_features[col].unique()) for col in ordinal_cols}

        # 3. Nominal: one-hot encoding
        nominal_cols = [c for c in NOMINAL_COLUMNS if c in df_features.columns]
        if nominal_cols:
            nominal_data = df_features.select(nominal_cols).to_pandas()
            nominal_encoded = pd.get_dummies(nominal_data, drop_first=False, dtype=int)
            feature_frames.append(nominal_encoded)
            feature_names.extend(list(nominal_encoded.columns))

        # 4. Binary (suicidal thoughts, family history) nếu còn
        binary_cols = [
            c for c in SENSITIVE_COLUMNS + ["Family History of Mental Illness"]
            if c in df_features.columns and c not in nominal_cols
        ]
        # Family History đã nằm trong nominal rồi, chỉ check suicidal
        binary_actual = [c for c in binary_cols if c not in NOMINAL_COLUMNS]
        for col in binary_actual:
            values = df_features[col].to_pandas()
            unique_vals = sorted(values.unique())
            if len(unique_vals) == 2:
                binary_map = {unique_vals[0]: 0, unique_vals[1]: 1}
                encoded = values.map(binary_map).values
                feature_frames.append(pd.DataFrame(encoded, columns=[col]))
                feature_names.append(col)

        # Combine all
        X_df = pd.concat(feature_frames, axis=1)
        X = X_df.values

        return X, y, feature_names

    # ==========================================
    # 📊 MODEL 0: DUMMY BASELINE
    # ==========================================

    def train_dummy(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Model 0: Dummy classifier — luôn dự đoán lớp đa số.
        Mục đích: tránh ảo tưởng mô hình phức tạp làm tốt hơn baseline.
        """
        dummy = DummyClassifier(strategy="most_frequent", random_state=self.random_state)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_results = cross_validate(
            dummy, X, y, cv=cv,
            scoring=["roc_auc", "average_precision", "f1", "recall", "precision"],
            return_train_score=False,
        )

        dummy.fit(X, y)
        y_pred = dummy.predict(X)
        y_proba = np.full(len(y), y.mean())  # Dummy probability = class prior

        metrics = {
            "roc_auc": cv_results["test_roc_auc"].mean(),
            "pr_auc": cv_results["test_average_precision"].mean(),
            "f1": cv_results["test_f1"].mean(),
            "recall": cv_results["test_recall"].mean(),
            "precision": cv_results["test_precision"].mean(),
            "brier_score": brier_score_loss(y, y_proba),
            "cv_scores": {k: v.tolist() for k, v in cv_results.items()},
        }

        self.models["dummy"] = dummy
        self.results["dummy"] = metrics

        logger.info(f"Dummy model: ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")
        return metrics

    # ==========================================
    # 📈 MODEL 1: LOGISTIC REGRESSION (TRUNG TÂM)
    # ==========================================

    def train_logistic(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        penalty: str = "l2",
        C: float = 1.0,
    ) -> Dict:
        """
        Model 1: Penalized Logistic Regression — mô hình trung tâm.

        Ưu điểm:
        - Dễ diễn giải bằng odds ratio
        - Kiểm soát quá khớp qua regularization
        - Là lựa chọn chuẩn trong thống kê ứng dụng
        """
        # Class weights để xử lý imbalance
        lr = LogisticRegression(
            
            C=C,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=self.random_state,
        )

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_results = cross_validate(
            lr, X, y, cv=cv,
            scoring=["roc_auc", "average_precision", "f1", "recall", "precision", "neg_brier_score"],
            return_train_score=True,
        )

        # Fit trên toàn bộ data
        lr.fit(X, y)
        y_proba = lr.predict_proba(X)[:, 1]
        y_pred = lr.predict(X)

        # Feature coefficients
        coef_df = self._extract_coefficients(lr, feature_names)

        # Calibration
        cal_score = brier_score_loss(y, y_proba)

        metrics = {
            "roc_auc": cv_results["test_roc_auc"].mean(),
            "roc_auc_std": cv_results["test_roc_auc"].std(),
            "pr_auc": cv_results["test_average_precision"].mean(),
            "pr_auc_std": cv_results["test_average_precision"].std(),
            "f1": cv_results["test_f1"].mean(),
            "f1_std": cv_results["test_f1"].std(),
            "recall": cv_results["test_recall"].mean(),
            "precision": cv_results["test_precision"].mean(),
            "brier_score": cal_score,
            "train_roc_auc": cv_results["train_roc_auc"].mean(),
            "coeficients": coef_df.to_pandas().to_dict(orient="records"),
            "cv_scores": {k: v.tolist() for k, v in cv_results.items()},
        }

        self.models["logistic"] = lr
        self.results["logistic"] = metrics

        logger.info(f"Logistic Regression: ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")
        return metrics

    def _extract_coefficients(
        self, model: LogisticRegression, feature_names: List[str]
    ) -> pl.DataFrame:
        """
        Trích xuất hệ số, odds ratio, và sắp xếp theo độ quan trọng.
        """
        coefs = model.coef_[0]
        df_coef = pl.DataFrame({
            "feature": feature_names,
            "coefficient": coefs,
            "odds_ratio": np.exp(coefs),
            "abs_coefficient": np.abs(coefs),
        }).sort("abs_coefficient", descending=True)

        return df_coef

    # ==========================================
    # 📊 MODEL 2: GAM (GENERALIZED ADDITIVE MODEL)
    # ==========================================

    def train_gam(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        feature_types: Optional[Dict[str, str]] = None,
        n_splines: int = 10,
    ) -> Dict:
        """
        Model 2: Generalized Additive Model — cân bằng giữa interpretability và flexibility.

        Ưu điểm:
        - Capture được nonlinear relationships
        - Vẫn interpretability được (partial dependence plots)
        - Transparent hơn tree-based models
        - Good middle ground giữa LR và CatBoost
        """
        from src.ml_models.gam_model import GAMClassifier

        # Auto-detect feature types if not provided
        if feature_types is None:
            feature_types = {}
            for name in feature_names:
                # Check if nominal (has '=' indicating one-hot)
                if '=' in name:
                    # This is a one-hot encoded column from a categorical
                    base_feat = name.split('=')[0]
                    if base_feat not in feature_types:
                        feature_types[name] = 'nominal'
                    else:
                        feature_types[name] = feature_types[base_feat]
                elif name in NOMINAL_COLUMNS:
                    feature_types[name] = 'nominal'
                elif name in ORDINAL_COLUMNS:
                    feature_types[name] = 'ordinal'
                elif name in NUMERIC_COLUMNS:
                    feature_types[name] = 'numeric'
                else:
                    # Default to numeric for unknown
                    feature_types[name] = 'numeric'

        gam = GAMClassifier(random_state=self.random_state)

        metrics = gam.train(
            X, y,
            feature_types=feature_types,
            feature_names=feature_names,
            n_splines=n_splines,
            optimize_splines=True,
        )

        self.models["gam"] = gam
        self.results["gam"] = metrics
        self.preprocessors["gam_feature_types"] = feature_types

        logger.info(f"GAM: ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")
        return metrics

    # ==========================================
    # 🌲 MODEL 3: CATBOOST
    # ==========================================

    def train_catboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Dict:
        """
        Model 3: CatBoost — mô hình dự báo bổ sung.

        Ưu điểm:
        - Xử lý tốt dữ liệu bảng hỗn hợp
        - Tự động xử lý biến phân loại
        - Thường mạnh hơn logistic regression về dự báo
        - Hỗ trợ GPU acceleration
        """
        try:
            from catboost import CatBoostClassifier, Pool
            import torch

            # Tự động detect GPU cho CatBoost
            use_gpu = torch.cuda.is_available()
            device_info = "GPU" if use_gpu else "CPU"
            logger.info(f"CatBoost using device: {device_info}")

            # Tính class weights
            n_samples = len(y)
            n_class_0 = (y == 0).sum()
            n_class_1 = (y == 1).sum()
            class_weights = [n_samples / (2 * n_class_0), n_samples / (2 * n_class_1)]

            # CatBoost GPU params
            catboost_params = {
                "iterations": 500,
                "depth": 6,
                "learning_rate": 0.05,
                "class_weights": class_weights,
                "loss_function": "Logloss",
                "verbose": False,
                "random_seed": self.random_state,
                "early_stopping_rounds": 30,
            }

            # Bật GPU nếu có
            if use_gpu:
                catboost_params["task_type"] = "GPU"
                catboost_params["devices"] = "0"
                logger.info(f"  ✅ CatBoost GPU enabled: {torch.cuda.get_device_name(0)}")

            catboost = CatBoostClassifier(**catboost_params)

            # Cross-validation thủ công (CatBoost không hỗ trợ sklearn CV tốt)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_metrics = {
                "roc_auc": [],
                "average_precision": [],
                "f1": [],
                "recall": [],
                "precision": [],
                "brier_score": [],
            }

            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                cb_fold = CatBoostClassifier(**catboost_params)
                cb_fold.fit(X_train, y_train, eval_set=(X_val, y_val))
                y_val_proba = cb_fold.predict_proba(X_val)[:, 1]
                y_val_pred = cb_fold.predict(X_val)

                cv_metrics["roc_auc"].append(roc_auc_score(y_val, y_val_proba))
                cv_metrics["average_precision"].append(average_precision_score(y_val, y_val_proba))
                # Dùng positive-class metrics (pos_label=1) thay vì weighted avg
                # để không che giấu hiệu năng trên lớp thiểu số (học sinh trầm cảm)
                cv_metrics["f1"].append(f1_score(y_val, y_val_pred, pos_label=1, zero_division=0))
                cv_metrics["recall"].append(recall_score(y_val, y_val_pred, pos_label=1, zero_division=0))
                cv_metrics["precision"].append(precision_score(y_val, y_val_pred, pos_label=1, zero_division=0))
                cv_metrics["brier_score"].append(brier_score_loss(y_val, y_val_proba))

            # Fit trên toàn bộ data
            catboost.fit(X, y)
            y_proba = catboost.predict_proba(X)[:, 1]
            y_pred = catboost.predict(X)

            # Feature importance
            importance = catboost.get_feature_importance()
            importance_df = pl.DataFrame({
                "feature": feature_names,
                "importance": importance,
            }).sort("importance", descending=True)

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
                "feature_importance": importance_df.to_pandas().to_dict(orient="records"),
                "cv_scores": {k: v for k, v in cv_metrics.items()},
                "_used_gpu": use_gpu,  # Flag để verify GPU usage
            }

            self.models["catboost"] = catboost
            self.results["catboost"] = metrics

            logger.info(f"CatBoost: ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")
            return metrics

        except ImportError:
            logger.warning("CatBoost not installed. Skipping. Install with: uv add catboost")
            return {
                "error": "CatBoost not installed",
                "roc_auc": 0,
                "f1": 0,
                "brier_score": 0,
            }

    # ==========================================
    # 📊 CALIBRATION ANALYSIS
    # ==========================================

    def calibration_analysis(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "Model",
    ) -> Dict:
        """
        Phân tích hiệu chuẩn: xác suất dự báo có đáng tin không?
        """
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")
        brier = brier_score_loss(y_true, y_proba)

        return {
            "model": model_name,
            "brier_score": brier,
            "calibration_curve": {
                "prob_true": prob_true.tolist(),
                "prob_pred": prob_pred.tolist(),
            },
        }

    # ==========================================
    # ⚖️  FAIRNESS ANALYSIS
    # ==========================================

    def fairness_by_subgroup(
        self,
        df: pl.DataFrame,
        model_name: str = "logistic",
        include_suicidal: bool = True,
    ) -> pl.DataFrame:
        """
        Đánh giá hiệu năng mô hình theo từng subgroup.

        Subgroups:
        - Gender: Male vs Female
        - Age group: 18-22, 23-26, 27-30, 31+
        - Family History: Yes vs No

        Returns:
            DataFrame với metrics cho từng subgroup.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Train it first.")

        model = self.models[model_name]
        X, y, feature_names = self.prepare_features(df, include_suicidal=include_suicidal)

        # Nếu là GAM, chỉ lấy các features đã chọn khi train
        if model_name == "gam" and "gam_feature_indices" in self.preprocessors and self.preprocessors["gam_feature_indices"] is not None:
            X_input = X[:, self.preprocessors["gam_feature_indices"]]
        else:
            X_input = X

        y_pred_obj = model.predict_proba(X_input)
        if y_pred_obj.ndim == 2:
            y_proba = y_pred_obj[:, 1]
        else:
            y_proba = y_pred_obj
        
        y_pred = model.predict(X_input)
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()

        # Định nghĩa subgroups
        subgroups = {}

        # 1. Gender
        if "Gender" in df.columns:
            for val in df["Gender"].unique():
                mask = df["Gender"].to_numpy() == val
                subgroups[f"Gender={val}"] = mask

        # 2. Age groups
        if "Age" in df.columns:
            ages = df["Age"].to_numpy()
            bins = [(18, 22), (23, 26), (27, 30), (31, 100)]
            labels = ["18-22", "23-26", "27-30", "31+"]
            for label, (lo, hi) in zip(labels, bins):
                mask = (ages >= lo) & (ages <= hi)
                if mask.sum() > 0:
                    subgroups[f"Age={label}"] = mask

        # 3. Family History
        if "Family History of Mental Illness" in df.columns:
            for val in df["Family History of Mental Illness"].unique():
                mask = df["Family History of Mental Illness"].to_numpy() == val
                subgroups[f"FH={val}"] = mask

        # Tính metrics cho từng subgroup
        rows = []
        for name, mask in subgroups.items():
            if mask.sum() < 30:  # Bỏ subgroup quá nhỏ
                continue
            y_sub = y[mask]
            yp_sub = y_proba[mask]
            yd_sub = y_pred[mask]

            row = {"subgroup": name, "n_samples": int(mask.sum())}
            row["roc_auc"] = roc_auc_score(y_sub, yp_sub) if len(np.unique(y_sub)) > 1 else 0
            row["recall"] = recall_score(y_sub, yd_sub, zero_division=0)
            row["precision"] = precision_score(y_sub, yd_sub, zero_division=0)
            f1_vals = classification_report(y_sub, yd_sub, output_dict=True, zero_division=0)
            row["f1_weighted"] = f1_vals.get("weighted avg", {}).get("f1-score", 0)
            row["brier_score"] = brier_score_loss(y_sub, yp_sub)

            # False positive rate & False negative rate
            cm = confusion_matrix(y_sub, yd_sub)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                row["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0
                row["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0
            else:
                row["fpr"] = 0
                row["fnr"] = 0

            # Prevalence
            row["prevalence"] = float(y_sub.mean())

            rows.append(row)

        df_fairness = pl.DataFrame(rows)
        return df_fairness

    def print_fairness_report(
        self,
        df: pl.DataFrame,
        include_suicidal: bool = True,
    ):
        """
        In báo cáo fairness cho cả models.
        """
        print()
        print("=" * 80)
        version_tag = "B (ĐẦY ĐỦ)" if include_suicidal else "A (BẢO THỦ)"
        print(f" ⚖️  FAIRNESS ANALYSIS — PHIÊN BẢN {version_tag}")
        print("=" * 80)

        for model_name in ["logistic", "gam", "catboost"]:
            if model_name not in self.models:
                continue

            label_map = {
                "logistic": "Logistic Regression",
                "gam": "GAM",
                "catboost": "CatBoost",
            }
            label = label_map.get(model_name, model_name)
            print(f"\n  📊 {label}:")

            df_fair = self.fairness_by_subgroup(df, model_name, include_suicidal)
            print(f"\n  {df_fair}")

            # Kiểm tra chênh lệch
            if df_fair.height > 1:
                auc_range = df_fair["roc_auc"].max() - df_fair["roc_auc"].min()
                fpr_range = df_fair["fpr"].max() - df_fair["fpr"].min()
                fnr_range = df_fair["fnr"].max() - df_fair["fnr"].min()

                print(f"\n  🔍 Chênh lệch giữa các nhóm:")
                print(f"     ROC-AUC range:  {df_fair['roc_auc'].min():.4f} — {df_fair['roc_auc'].max():.4f} (Δ = {auc_range:.4f})")
                print(f"     FPR range:      {df_fair['fpr'].min():.4f} — {df_fair['fpr'].max():.4f} (Δ = {fpr_range:.4f})")
                print(f"     FNR range:      {df_fair['fnr'].min():.4f} — {df_fair['fnr'].max():.4f} (Δ = {fnr_range:.4f})")

                if auc_range > 0.05:
                    print(f"     ⚠️  Cảnh báo: Chênh lệch AUC > 0.05 — có thể có bias giữa các nhóm")
                if fpr_range > 0.10:
                    print(f"     ⚠️  Cảnh báo: Chênh lệch FPR > 0.10 — nhóm này bị báo động giả nhiều hơn")
                if fnr_range > 0.10:
                    print(f"     ⚠️  Cảnh báo: Chênh lệch FNR > 0.10 — nhóm này bị bỏ sót nhiều hơn")

    # ==========================================
    # 📊 THRESHOLD ANALYSIS & DECISION CURVE
    # ==========================================

    def threshold_analysis(
        self,
        df: pl.DataFrame,
        model_name: str = "logistic",
        include_suicidal: bool = True,
    ) -> pl.DataFrame:
        """
        Phân tích ngưỡng quyết định: thử nhiều ngưỡng khác nhau
        và báo cáo trade-off giữa recall, precision, FPR, FNR.

        Quan trọng: Ngưỡng mặc định 0.5 không phải lúc nào cũng tối ưu.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")

        model = self.models[model_name]
        X, y, _ = self.prepare_features(df, include_suicidal=include_suicidal)

        # Nếu là GAM, chỉ lấy các features đã chọn khi train
        if model_name == "gam" and "gam_feature_indices" in self.preprocessors and self.preprocessors["gam_feature_indices"] is not None:
            X_input = X[:, self.preprocessors["gam_feature_indices"]]
        else:
            X_input = X

        y_pred_obj = model.predict_proba(X_input)
        if y_pred_obj.ndim == 2:
            y_proba = y_pred_obj[:, 1]
        else:
            y_proba = y_pred_obj

        thresholds = np.arange(0.2, 0.8, 0.02)
        rows = []

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            cm = confusion_matrix(y, y_pred)

            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                n_flagged = y_pred.sum()

                rows.append({
                    "threshold": round(thresh, 2),
                    "recall": round(recall, 4),
                    "precision": round(precision, 4),
                    "f1": round(f1, 4),
                    "fpr": round(fpr, 4),
                    "fnr": round(fnr, 4),
                    "n_flagged": int(n_flagged),
                    "flagged_pct": round(n_flagged / len(y) * 100, 1),
                })

        return pl.DataFrame(rows)

    def print_threshold_report(
        self,
        df: pl.DataFrame,
        model_name: str = "logistic",
        include_suicidal: bool = True,
    ):
        """
        In báo cáo ngưỡng quyết định và khuyến nghị.
        """
        df_thresh = self.threshold_analysis(df, model_name, include_suicidal)

        label_map = {
            "logistic": "Logistic Regression",
            "gam": "GAM",
            "catboost": "CatBoost",
        }
        label = label_map.get(model_name, model_name)
        print(f"\n  🎯 PHÂN TÍCH NGƯỠNG QUYẾT ĐỊNH — {label}:")
        print()
        print("  Threshold | Recall | Precision |   F1   |  FPR   |  FNR   | Flagged")
        print("  " + "-" * 68)

        for row in df_thresh.iter_rows(named=True):
            marker = " ←" if abs(row["threshold"] - 0.5) < 0.01 else ""
            print(f"    {row['threshold']:.2f}      | {row['recall']:.4f} | {row['precision']:.4f}    | {row['f1']:.4f} | {row['fpr']:.4f} | {row['fnr']:.4f} | {row['flagged_pct']:.1f}%{marker}")

        # Khuyến nghị ngưỡng
        # Ngưỡng ưu tiên recall (sàng lọc): FNR < 0.15
        low_fnr = df_thresh.filter(pl.col("fnr") < 0.15)
        if low_fnr.height > 0:
            best = low_fnr.sort("f1", descending=True).row(0, named=True)
            print(f"\n  💡 Khuyến nghị ngưỡng ưu tiên Recall (sàng lọc): threshold = {best['threshold']:.2f}")
            print(f"     Recall = {best['recall']:.4f}, Precision = {best['precision']:.4f}, F1 = {best['f1']:.4f}")
            print(f"     FNR = {best['fnr']:.4f} (bỏ sót {best['fnr']*100:.1f}% cao trầm cảm)")

        # Ngưỡng cân bằng: F1 cao nhất
        best_f1 = df_thresh.sort("f1", descending=True).row(0, named=True)
        print(f"\n  💡 Khuyến nghị ngưỡng cân bằng (F1 max): threshold = {best_f1['threshold']:.2f}")
        print(f"     F1 = {best_f1['f1']:.4f}, Recall = {best_f1['recall']:.4f}, Precision = {best_f1['precision']:.4f}")

    # ==========================================
    # 📝 COMPARISON & REPORT
    # ==========================================

    def compare_models(self) -> pl.DataFrame:
        """
        So sánh tất cả các mô hình đã huấn luyện.
        """
        rows = []
        for name, metrics in self.results.items():
            row = {"model": name}
            for key in ["roc_auc", "pr_auc", "f1", "recall", "precision", "brier_score"]:
                row[key] = metrics.get(key, None)
                if isinstance(row[key], list):
                    row[key] = row[key][0] if row[key] else None
            rows.append(row)

        return pl.DataFrame(rows)

    def print_report(self, include_suicidal: bool = True):
        """
        In báo cáo đầy đủ.
        """
        print()
        print("=" * 80)
        print(" 📊 BÁO CÁO SO SÁNH MÔ HÌNH")
        print("=" * 80)

        version_label = "Phiên bản ĐẦY ĐỦ (có Suicidal thoughts)" if include_suicidal else "Phiên bản BẢO THỦ (không Suicidal thoughts)"
        print(f"\n  {version_label}")
        print()

        comparison = self.compare_models()
        print(comparison)

        # Find models that exist
        has_dummy = "dummy" in self.results
        has_lr = "logistic" in self.results
        has_gam = "gam" in self.results
        has_catboost = "catboost" in self.results and "error" not in self.results.get("catboost", {})

        # Dummy baseline
        if has_dummy:
            dm = self.results["dummy"]
            print(f"\n  🎯 Dummy Baseline:")
            print(f"     ROC-AUC: {dm['roc_auc']:.4f}")
            print(f"     F1:      {dm['f1']:.4f}")
            print(f"     Brier:   {dm['brier_score']:.4f}")

        # Logistic Regression
        if has_lr:
            lr = self.results["logistic"]
            print(f"\n  📈 Logistic Regression (mô hình trung tâm):")
            print(f"     ROC-AUC: {lr['roc_auc']:.4f} ± {lr['roc_auc_std']:.4f}")
            print(f"     PR-AUC:  {lr['pr_auc']:.4f} ± {lr['pr_auc_std']:.4f}")
            print(f"     F1:      {lr['f1']:.4f} ± {lr['f1_std']:.4f}")
            print(f"     Brier:   {lr['brier_score']:.4f}")

        # GAM
        if has_gam:
            gam = self.results["gam"]
            print(f"\n  🎨 GAM - Generalized Additive Model (linh hoạt):")
            print(f"     ROC-AUC: {gam['roc_auc']:.4f} ± {gam['roc_auc_std']:.4f}")
            print(f"     PR-AUC:  {gam['pr_auc']:.4f} ± {gam['pr_auc_std']:.4f}")
            print(f"     F1:      {gam['f1']:.4f} ± {gam['f1_std']:.4f}")
            print(f"     Brier:   {gam['brier_score']:.4f}")

        # CatBoost
        if has_catboost:
            cb = self.results["catboost"]
            print(f"\n  🌲 CatBoost (mô hình dự báo):")
            print(f"     ROC-AUC: {cb['roc_auc']:.4f} ± {cb['roc_auc_std']:.4f}")
            print(f"     PR-AUC:  {cb['pr_auc']:.4f} ± {cb['pr_auc_std']:.4f}")
            print(f"     F1:      {cb['f1']:.4f} ± {cb['f1_std']:.4f}")
            print(f"     Brier:   {cb['brier_score']:.4f}")

        # Feature importance (logistic)
        if has_lr and "logistic" in self.results:
            print(f"\n  🔍 Top 10 biến quan trọng nhất (Logistic Regression):")
            coefs = self.results["logistic"].get("coeficients", [])
            for i, item in enumerate(coefs[:10], 1):
                or_val = item["odds_ratio"]
                direction = "↑" if item["coefficient"] > 0 else "↓"
                print(f"     {i:2d}. {item['feature']:<40s} OR={or_val:.3f} {direction}")

        # Feature importance (GAM)
        if has_gam and "gam" in self.results:
            print(f"\n  🎨 Top 10 biến quan trọng nhất (GAM):")
            importance = self.results["gam"].get("feature_importance", [])
            for i, item in enumerate(importance[:10], 1):
                var_imp = item["variance_importance"]
                print(f"     {i:2d}. {item['feature']:<40s} VarImp={var_imp:.6f}")

        print()
        print("  ⚠️  LƯU Ý:")
        print("     - Đây là QUAN HỆ LIÊN QUAN, không phải nhân quả")
        print("     - Mô hình chỉ HỖ TRỢ, không thay thế đánh giá lâm sàng")
        print("     - Cần kiểm tra calibration trước khi dùng ngưỡng quyết định")
        print()

    # ==========================================
    # 🚀 FULL PIPELINE
    # ==========================================

    def run_full_pipeline(
        self,
        df: pl.DataFrame,
        include_suicidal: bool = True,
        output_dir: str = "results/",
        run_gam: bool = True,
    ) -> Dict:
        """
        Chạy toàn bộ pipeline: chuẩn bị dữ liệu → huấn luyện → đánh giá → báo cáo.
        
        Sequence: Baseline → Logistic → GAM → CatBoost
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print()
        print("=" * 80)
        print(" 🤖 GIAI ĐOẠN 7-8: XÂY DỰNG & ĐÁNH GIÁ MÔ HÌNH")
        print("=" * 80)

        # Device info
        import torch
        print()
        print(" 💻 DEVICE INFO:")
        if torch.cuda.is_available():
            print(f"     ✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"     ✅ CUDA: {torch.version.cuda}")
            print(f"     ✅ CatBoost sẽ dùng GPU acceleration")
        else:
            print(f"     ⚠️  CPU only (không có GPU)")
        print()

        version = "full" if include_suicidal else "conservative"
        print(f"\n  📌 Phiên bản: {'ĐẦY ĐỦ' if include_suicidal else 'BẢO THỦ'}")
        if include_suicidal:
            print("     ⚠️  Có biến 'Suicidal thoughts' — nguy cơ rò rỉ nhãn")
        else:
            print("     ✅ Không có biến 'Suicidal thoughts' — an toàn hơn")

        # 1. Prepare features
        print("\n  🔧 Chuẩn bị dữ liệu...")
        X, y, feature_names = self.prepare_features(df, include_suicidal=include_suicidal)
        print(f"     Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
        print(f"     Class distribution: {(y == 1).sum()} positive, {(y == 0).sum()} negative")

        # 2. Model 0: Dummy
        with Timer("Dummy Baseline"):
            print("\n  🎯 [1/4] Model 0: Dummy Baseline...")
            self.train_dummy(X, y)

        # 3. Model 1: Logistic Regression
        with Timer("Logistic Regression"):
            print("\n  📈 [2/4] Model 1: Logistic Regression (trung tâm)...")
            self.train_logistic(X, y, feature_names)

        # 4. Model 2: GAM
        if run_gam:
            with Timer("GAM Training"):
                print("\n  🎨 [3/4] Model 2: GAM - Generalized Additive Model (using Top 15 features)...")
                
                # Lấy Top 15 features từ Logistic Regression để train GAM
                if "logistic" in self.results:
                    coef_df = self._extract_coefficients(self.models["logistic"], feature_names)
                    top_features = coef_df["feature"].head(15).to_list()
                    
                    # Lọc X và feature_names cho GAM
                    feat_idx_map = {name: i for i, name in enumerate(feature_names)}
                    top_indices = [feat_idx_map[name] for name in top_features if name in feat_idx_map]
                    self.preprocessors["gam_feature_indices"] = top_indices # Lưu lại indices
                    X_gam = X[:, top_indices]
                    names_gam = [feature_names[i] for i in top_indices]
                    
                    # Lấy feature_types an toàn (tự tạo nếu chưa có)
                    if "gam_feature_types" in self.preprocessors:
                        types_gam = {name: self.preprocessors["gam_feature_types"].get(name, 'numeric') for name in names_gam}
                    else:
                        # Tự động phát hiện loại biến cho top features
                        types_gam = {}
                        for name in names_gam:
                            if '=' in name:
                                types_gam[name] = 'nominal'
                            elif name in NOMINAL_COLUMNS:
                                types_gam[name] = 'nominal'
                            elif name in ORDINAL_COLUMNS:
                                types_gam[name] = 'ordinal'
                            else:
                                types_gam[name] = 'numeric'
                else:
                    X_gam, names_gam, types_gam = X, feature_names, None
                    self.preprocessors["gam_feature_indices"] = None

                self.train_gam(X_gam, y, names_gam, feature_types=types_gam)
        else:
            print("\n  ⏭️  Bỏ qua GAM (run_gam=False)")

        # 5. Model 3: CatBoost
        with Timer("CatBoost Training"):
            print("\n  🌲 [4/4] Model 3: CatBoost (dự báo bổ sung)...")
            self.train_catboost(X, y, feature_names)

        # 6. Calibration
        print("\n  📊 Calibration Analysis...")
        for name in ["logistic", "gam", "catboost"]:
            if name in self.models:
                model = self.models[name]
                
                # Nếu là GAM, chỉ lấy các features đã chọn
                if name == "gam" and "gam_feature_indices" in self.preprocessors and self.preprocessors["gam_feature_indices"] is not None:
                    X_input = X[:, self.preprocessors["gam_feature_indices"]]
                else:
                    X_input = X
                    
                y_pred_obj = model.predict_proba(X_input)
                # Xử lý mảng 1D vs 2D
                if y_pred_obj.ndim == 2:
                    y_proba = y_pred_obj[:, 1]
                else:
                    y_proba = y_pred_obj
                    
                cal = self.calibration_analysis(y, y_proba, name.upper())
                print(f"     {name.upper()}: Brier score = {cal['brier_score']:.4f}")

        # 7. Compare & Report
        self.print_report(include_suicidal)

        # 8. Save results
        results_path = output_path / f"model_results_{version}.json"
        serializable_results = {}
        for name, metrics in self.results.items():
            serializable_results[name] = {}
            for k, v in metrics.items():
                if k in ["cv_scores", "coeficients", "feature_importance"]:
                    continue  # Skip large nested data
                if isinstance(v, (np.floating, float)):
                    serializable_results[name][k] = float(v)
                elif isinstance(v, (np.integer, int)):
                    serializable_results[name][k] = int(v)
                elif isinstance(v, np.ndarray):
                    serializable_results[name][k] = v.tolist()
                else:
                    serializable_results[name][k] = v

        serializable_results["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "version": version,
            "include_suicidal": include_suicidal,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "class_0": int((y == 0).sum()),
            "class_1": int((y == 1).sum()),
            "models_trained": list(self.models.keys()),
        }

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        print(f"  ✅ Kết quả đã lưu: {results_path}")

        return self.results
