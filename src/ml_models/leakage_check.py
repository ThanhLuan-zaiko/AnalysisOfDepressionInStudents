"""
Label Leakage Investigation — Kiểm tra rò rỉ nhãn

Mục tiêu:
  Xác định xem biến "Have you ever had suicidal thoughts ?" có phải là
  nguyên nhân gây rò rỉ nhãn (label leakage) hay không.

Phương pháp:
  1. Cross-tabulation: Suicidal thoughts × Depression
  2. So sánh hiệu năng mô hình: có vs không có biến này
  3. Stress test: bỏ từng biến nhạy cảm, xem hiệu năng giảm bao nhiêu
  4. Kiểm tra nguồn gốc dữ liệu (synthetic hay real)

Usage:
    uv run python -c "from src.ml_models.leakage_check import LabelLeakageInvestigator; ..."
"""

import numpy as np
import polars as pl
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class LabelLeakageInvestigator:
    """
    Điều tra rò rỉ nhãn từ biến "Suicidal thoughts" vào nhãn "Depression".
    """

    SENSITIVE_VAR = "Have you ever had suicidal thoughts ?"

    EXCLUDED_COLUMNS = [
        "id", "Profession", "Work Pressure", "Job Satisfaction",
    ]

    NUMERIC_COLUMNS = ["Age", "CGPA", "Work/Study Hours"]
    ORDINAL_COLUMNS = ["Academic Pressure", "Study Satisfaction", "Financial Stress"]
    NOMINAL_COLUMNS = [
        "Gender", "City", "Degree", "Sleep Duration", "Dietary Habits",
        "Family History of Mental Illness",
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    # ==========================================
    # 1. CROSS-TABULATION & ASSOCIATION
    # ==========================================

    def cross_tab_analysis(
        self, df: pl.DataFrame
    ) -> Dict[str, Any]:
        """
        Phân tích bảng chéo: Suicidal thoughts × Depression.

        Nếu tỷ lệ Depression=1 gần như 100% khi Suicidal=Yes (hoặc ngược lại),
        đây là dấu hiệu rò rỉ nhãn.
        """
        col = self.SENSITIVE_VAR
        if col not in df.columns or "Depression" not in df.columns:
            return {"error": "Missing required columns"}

        # Cross-tab counts
        ct = pd.crosstab(df[col].to_pandas(), df["Depression"].to_pandas())
        ct_pct = pd.crosstab(
            df[col].to_pandas(), df["Depression"].to_pandas(), normalize="index"
        ) * 100

        result = {
            "cross_tab_counts": ct.to_dict(),
            "cross_tab_pct": {k: {dk: round(dv, 2) for dk, dv in v.items()} for k, v in ct_pct.to_dict().items()},
            "total_n": df.height,
        }

        # Thông tin quan trọng: tỷ lệ Depression=1 trong từng nhóm Suicidal
        for val in sorted(df[col].unique()):
            subset = df.filter(pl.col(col) == val)
            dep_rate = subset["Depression"].mean() * 100
            result[f"Depression_rate_when_{val}"] = round(dep_rate, 2)

        # Cramer's V cho association strength
        from scipy import stats
        chi2, p_value, dof, expected = stats.chi2_contingency(ct)
        n = ct.sum().sum()
        k = min(ct.shape)
        cramers_v = np.sqrt(chi2 / (n * (k - 1)))

        result["chi2"] = round(chi2, 2)
        result["p_value"] = p_value
        result["cramers_v"] = round(cramers_v, 4)
        result["association_strength"] = (
            "Rất mạnh" if cramers_v > 0.5
            else "Mạnh" if cramers_v > 0.3
            else "Trung bình" if cramers_v > 0.1
            else "Yếu"
        )

        return result

    # ==========================================
    # 2. FEATURE IMPORTANCE RANKING
    # ==========================================

    def feature_importance_comparison(
        self, df: pl.DataFrame
    ) -> Dict[str, Any]:
        """
        Huấn Logistic Regression 2 lần:
        - Lần 1: CÓ Suicidal thoughts
        - Lần 2: KHÔNG có Suicidal thoughts

        So sánh hệ số (coefficient) và feature importance.
        """
        X_full, y_full, names_full = self._prepare_features(df, include_sensitive=True)
        X_no, y_no, names_no = self._prepare_features(df, include_sensitive=False)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        # Model đầy đủ
        lr_full = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=self.random_state)
        cv_full = cross_validate(lr_full, X_full, y_full, cv=cv, scoring="roc_auc", return_train_score=True)
        lr_full.fit(X_full, y_full)

        # Model không có Suicidal thoughts
        lr_no = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=self.random_state)
        cv_no = cross_validate(lr_no, X_no, y_no, cv=cv, scoring="roc_auc", return_train_score=True)
        lr_no.fit(X_no, y_no)

        # Feature coefficients
        coefs_full = {name: float(coef) for name, coef in zip(names_full, lr_full.coef_[0])}
        coefs_full_sorted = dict(sorted(coefs_full.items(), key=lambda x: abs(x[1]), reverse=True))

        coefs_no = {name: float(coef) for name, coef in zip(names_no, lr_no.coef_[0])}
        coefs_no_sorted = dict(sorted(coefs_no.items(), key=lambda x: abs(x[1]), reverse=True))

        return {
            "with_sensitive": {
                "roc_auc_cv": round(cv_full["test_score"].mean(), 4),
                "roc_auc_std": round(cv_full["test_score"].std(), 4),
                "top_10_features": dict(list(coefs_full_sorted.items())[:10]),
            },
            "without_sensitive": {
                "roc_auc_cv": round(cv_no["test_score"].mean(), 4),
                "roc_auc_std": round(cv_no["test_score"].std(), 4),
                "top_10_features": dict(list(coefs_no_sorted.items())[:10]),
            },
            "auc_drop": round(
                cv_full["test_score"].mean() - cv_no["test_score"].mean(), 4
            ),
            "auc_drop_pct": round(
                (cv_full["test_score"].mean() - cv_no["test_score"].mean())
                / cv_full["test_score"].mean() * 100, 2
            ),
        }

    # ==========================================
    # 3. STRESS TEST — BỎ TỪNG BIẾN NHẠY CẢM
    # ==========================================

    def stress_test(
        self, df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Thử bỏ TỪNG biến một (không chỉ Suicidal thoughts),
        xem biến nào làm giảm AUC nhiều nhất.

        Nếu bỏ Suicidal thoughts làm AUC giảm >> các biến khác → leakage.
        """
        # Danh sách tất cả các biến candidate
        all_candidate_cols = (
            self.NUMERIC_COLUMNS
            + self.ORDINAL_COLUMNS
            + self.NOMINAL_COLUMNS
            + [self.SENSITIVE_VAR]
        )
        # Lọc chỉ giữ cột có thật trong df
        all_candidate_cols = [c for c in all_candidate_cols if c in df.columns]

        rows = []

        # Baseline: dùng tất cả
        X_base, y_base, _ = self._prepare_features(df, include_sensitive=True)
        lr_base = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=self.random_state)
        cv_base = cross_validate(lr_base, X_base, y_base, cv=5, scoring="roc_auc")
        baseline_auc = cv_base["test_score"].mean()

        rows.append({
            "test": "FULL (baseline)",
            "variables_dropped": "—",
            "roc_auc": round(baseline_auc, 4),
            "auc_delta": 0.0,
            "auc_delta_pct": 0.0,
        })

        # Thử bỏ từng biến
        for col_to_drop in all_candidate_cols:
            X_st, y_st, _ = self._prepare_features(df, include_sensitive=True, drop_extra=[col_to_drop])
            lr_st = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=self.random_state)
            cv_st = cross_validate(lr_st, X_st, y_st, cv=5, scoring="roc_auc")
            auc_st = cv_st["test_score"].mean()

            delta = baseline_auc - auc_st

            rows.append({
                "test": f"Bỏ '{col_to_drop}'",
                "variables_dropped": col_to_drop,
                "roc_auc": round(auc_st, 4),
                "auc_delta": round(delta, 4),
                "auc_delta_pct": round(delta / baseline_auc * 100, 2),
            })

        df_result = pl.DataFrame(rows).sort("auc_delta", descending=True)
        return df_result

    # ==========================================
    # 4. SYNTHETIC DATA DETECTION
    # ==========================================

    def synthetic_data_check(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Kiểm tra dấu hiệu dữ liệu tổng hợp (synthetic):

        Dấu hiệu cảnh báo:
        - Tỷ lệ "Suicidal thoughts = Yes" > 50% (thực tế thường 10-20%)
        - Phân phối độ tuổi quá đều hoặc quá "đẹp"
        - CGPA phân phối quá tròn
        - Không có outlier thực tế
        """
        checks = {}

        # 1. Suicidal thoughts rate
        if self.SENSITIVE_VAR in df.columns:
            yes_count = df.filter(pl.col(self.SENSITIVE_VAR) == "Yes").height
            yes_pct = yes_count / df.height * 100
            checks["suicidal_thoughts_rate_pct"] = round(yes_pct, 2)
            checks["suicidal_thoughts_warning"] = (
                "RẤT CAO — dấu hiệu synthetic data" if yes_pct > 50
                else "CAO" if yes_pct > 30
                else "Bình thường"
            )

        # 2. Depression rate
        if "Depression" in df.columns:
            dep_rate = df["Depression"].mean() * 100
            checks["depression_rate_pct"] = round(dep_rate, 2)
            checks["depression_rate_warning"] = (
                "CAO BẤT THƯỜNG" if dep_rate > 50
                else "Cao" if dep_rate > 30
                else "Bình thường"
            )

        # 3. Age distribution — kiểm tra tính "tự nhiên"
        if "Age" in df.columns:
            ages = df["Age"].to_numpy()
            checks["age_mean"] = round(float(ages.mean()), 1)
            checks["age_std"] = round(float(ages.std()), 1)
            checks["age_min"] = int(ages.min())
            checks["age_max"] = int(ages.max())
            # Thực tế: tuổi sinh viên thường tập trung 18-24
            young_pct = ((ages >= 18) & (ages <= 24)).sum() / len(ages) * 100
            checks["age_18_24_pct"] = round(young_pct, 1)
            checks["age_distribution_warning"] = (
                "Không tập trung vào độ tuổi sinh viên — dấu hiệu synthetic" if young_pct < 50
                else "Bình thường"
            )

        # 4. CGPA — kiểm tra phân phối
        if "CGPA" in df.columns:
            cgpas = df["CGPA"].to_numpy()
            checks["cgpa_mean"] = round(float(cgpas.mean()), 2)
            checks["cgpa_std"] = round(float(cgpas.std()), 2)
            checks["cgpa_min"] = round(float(cgpas.min()), 2)
            checks["cgpa_max"] = round(float(cgpas.max()), 2)

        # 5. Số lượng thành phố — nếu quá nhiều → có thể synthetic
        if "City" in df.columns:
            n_cities = df["City"].n_unique()
            checks["n_cities"] = n_cities
            top_city = df["City"].value_counts().sort("count", descending=True).row(0, named=True)
            checks["top_city"] = top_city["City"]
            checks["top_city_pct"] = round(top_city["count"] / df.height * 100, 2)

        # 6. Balance của biến nhị phân
        if "Family History of Mental Illness" in df.columns:
            fh = df["Family History of Mental Illness"]
            yes_pct = fh.filter(fh == "Yes").len() / df.height * 100
            checks["family_history_yes_pct"] = round(yes_pct, 2)
            # Thực tế: ~20-25% có tiền sử gia đình
            checks["family_history_warning"] = (
                "Quá cân bằng (~50/50) — dấu hiệu synthetic" if 40 < yes_pct < 60
                else "Bình thường"
            )

        # Tổng kết
        n_warnings = sum(1 for k, v in checks.items() if k.endswith("_warning") and "bình thường" not in v.lower() and "bình thường" not in str(v).lower())
        checks["total_warnings"] = n_warnings
        checks["likely_synthetic"] = n_warnings >= 3

        return checks

    # ==========================================
    # 5. FULL REPORT
    # ==========================================

    def run_full_investigation(
        self, df: pl.DataFrame
    ) -> Dict[str, Any]:
        """
        Chạy toàn bộ điều tra rò rỉ nhãn.
        """
        print()
        print("=" * 80)
        print(" 🔎 ĐIỀU TRA RÒ RỈ NHÃN (LABEL LEAKAGE INVESTIGATION)")
        print("=" * 80)

        results = {}

        # 1. Cross-tab
        print("\n  1️⃣  Cross-tabulation: Suicidal thoughts × Depression")
        print("  " + "-" * 60)
        cross_tab = self.cross_tab_analysis(df)
        results["cross_tab"] = cross_tab

        if "error" not in cross_tab:
            for key, val in cross_tab.items():
                if key.startswith("Depression_rate"):
                    print(f"     {key}: {val}%")
            print(f"     Cramer's V: {cross_tab.get('cramers_v', 'N/A')}")
            print(f"     Association: {cross_tab.get('association_strength', 'N/A')}")
            print(f"     Chi2: {cross_tab.get('chi2', 'N/A')}, p-value: {cross_tab.get('p_value', 'N/A'):.2e}")

        # 2. Feature importance comparison
        print("\n  2️⃣  So sánh mô hình: CÓ vs KHÔNG có Suicidal thoughts")
        print("  " + "-" * 60)
        feat_comp = self.feature_importance_comparison(df)
        results["feature_importance_comparison"] = feat_comp

        print(f"     Với Suicidal thoughts:    AUC = {feat_comp['with_sensitive']['roc_auc_cv']:.4f}")
        print(f"     Không Suicidal thoughts:  AUC = {feat_comp['without_sensitive']['roc_auc_cv']:.4f}")
        print(f"     Δ AUC = {feat_comp['auc_drop']:.4f} ({feat_comp['auc_drop_pct']:.2f}%)")

        print(f"\n     Top 5 features (CÓ Suicidal thoughts):")
        for i, (name, coef) in enumerate(list(feat_comp["with_sensitive"]["top_10_features"].items())[:5], 1):
            direction = "↑" if coef > 0 else "↓"
            print(f"       {i}. {name:<45s} coef={coef:+.4f} {direction}")

        print(f"\n     Top 5 features (KHÔNG Suicidal thoughts):")
        for i, (name, coef) in enumerate(list(feat_comp["without_sensitive"]["top_10_features"].items())[:5], 1):
            direction = "↑" if coef > 0 else "↓"
            print(f"       {i}. {name:<45s} coef={coef:+.4f} {direction}")

        # 3. Stress test
        print("\n  3️⃣  Stress test — Bỏ từng biến, xem AUC giảm bao nhiêu")
        print("  " + "-" * 60)
        stress_df = self.stress_test(df)
        results["stress_test"] = stress_df.to_pandas().to_dict(orient="records")

        print(f"\n     {'Test':<50s} | AUC    | Δ AUC  | Δ%")
        print("     " + "-" * 70)
        for row in stress_df.iter_rows(named=True):
            marker = " ⚠️ " if row["auc_delta_pct"] > 5 else ""
            print(f"     {row['test']:<50s} | {row['roc_auc']:.4f} | {row['auc_delta']:+.4f} | {row['auc_delta_pct']:+.2f}%{marker}")

        # 4. Synthetic data check
        print("\n  4️⃣  Kiểm tra dấu hiệu Synthetic Data")
        print("  " + "-" * 60)
        synth_check = self.synthetic_data_check(df)
        results["synthetic_check"] = synth_check

        for key, val in synth_check.items():
            if not key.endswith("_warning"):
                print(f"     {key}: {val}")

        if synth_check.get("likely_synthetic", False):
            print(f"\n     🚨 KẾT LUẬN: Dữ liệu có khả năng cao là SYNTHETIC "
                  f"({synth_check['total_warnings']} cảnh báo)")
        else:
            print(f"\n     ✅ Không đủ dấu hiệu để kết luận synthetic data.")

        # 5. Final verdict
        print("\n  " + "=" * 80)
        print("  📋 KẾT LUẬN ĐIỀU TRA RÒ RỈ NHÃN")
        print("  " + "=" * 80)

        auc_drop_pct = feat_comp.get("auc_drop_pct", 0)
        cramers_v = cross_tab.get("cramers_v", 0)

        print(f"""
     Cramer's V (Suicidal thoughts × Depression): {cramers_v}
     AUC drop khi bỏ Suicidal thoughts: {feat_comp.get('auc_drop', 0):.4f} ({auc_drop_pct:.2f}%)
     Dữ liệu có khả năng synthetic: {synth_check.get('likely_synthetic', False)}

     💡 ĐÁNH GIÁ:
  """)

        if auc_drop_pct > 3:
            print("     ⚠️  RỦI RO CAO: Suicidal thoughts đóng góp quá lớn vào mô hình.")
            print("        Đây là dấu hiệu rõ rệt của label leakage hoặc chồng lấp khái niệm.")
            print("        Khuyến nghị: Sử dụng Phiên bản A (không Suicidal thoughts).")
        elif auc_drop_pct > 1:
            print("     ⚡ RỦI RO TRUNG BÌNH: Suicidal thoughts có đóng góp đáng kể.")
            print("        Cần giải thích rõ trong báo cáo.")
        else:
            print("     ✅ RỦI RO THẤP: Suicidal thoughts không chi phối mô hình.")

        print()
        return results

    # ==========================================
    # HELPER: Feature preparation
    # ==========================================

    def _prepare_features(
        self,
        df: pl.DataFrame,
        include_sensitive: bool = False,
        drop_extra: list = None,
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """Chuẩn bị feature matrix (đơn giản hóa cho investigation)."""
        df_work = df.clone()

        # Drop excluded
        cols_to_drop = [c for c in self.EXCLUDED_COLUMNS if c in df_work.columns]
        if cols_to_drop:
            df_work = df_work.drop(cols_to_drop)

        # Drop sensitive if requested
        if not include_sensitive and self.SENSITIVE_VAR in df_work.columns:
            df_work = df_work.drop(self.SENSITIVE_VAR)

        # Drop extra
        if drop_extra:
            cols = [c for c in drop_extra if c in df_work.columns]
            if cols:
                df_work = df_work.drop(cols)

        # Target
        y = df_work["Depression"].to_numpy()
        df_work = df_work.drop("Depression")

        # Handle missing
        for col in df_work.columns:
            if df_work[col].null_count() > 0:
                if df_work[col].dtype in [pl.Int64, pl.Float64]:
                    df_work = df_work.with_columns(pl.col(col).fill_null(pl.col(col).median()))
                else:
                    df_work = df_work.with_columns(pl.col(col).fill_null(pl.col(col).mode().first()))

        feature_frames = []
        feature_names = []

        # Numeric
        num_cols = [c for c in self.NUMERIC_COLUMNS if c in df_work.columns]
        if num_cols:
            num_data = df_work.select(num_cols).to_pandas()
            scaler = StandardScaler()
            num_scaled = scaler.fit_transform(num_data)
            feature_frames.append(pd.DataFrame(num_scaled, columns=num_cols))
            feature_names.extend(num_cols)

        # Ordinal
        ord_cols = [c for c in self.ORDINAL_COLUMNS if c in df_work.columns]
        for col in ord_cols:
            values = df_work[col].to_pandas()
            unique_sorted = sorted(values.unique())
            ordinal_map = {v: i for i, v in enumerate(unique_sorted)}
            encoded = values.map(ordinal_map).values
            feature_frames.append(pd.DataFrame(encoded, columns=[col]))
            feature_names.append(col)

        # Nominal (one-hot)
        nom_cols = [c for c in self.NOMINAL_COLUMNS if c in df_work.columns]
        if nom_cols:
            nom_data = df_work.select(nom_cols).to_pandas()
            nom_encoded = pd.get_dummies(nom_data, drop_first=False, dtype=int)
            feature_frames.append(nom_encoded)
            feature_names.extend(list(nom_encoded.columns))

        # Binary (sensitive — only if included)
        if include_sensitive and self.SENSITIVE_VAR in df_work.columns:
            values = df_work[self.SENSITIVE_VAR].to_pandas()
            unique_vals = sorted(values.unique())
            if len(unique_vals) == 2:
                binary_map = {unique_vals[0]: 0, unique_vals[1]: 1}
                encoded = values.map(binary_map).values
                feature_frames.append(pd.DataFrame(encoded, columns=[self.SENSITIVE_VAR]))
                feature_names.append(self.SENSITIVE_VAR)

        X = pd.concat(feature_frames, axis=1).values
        return X, y, feature_names
