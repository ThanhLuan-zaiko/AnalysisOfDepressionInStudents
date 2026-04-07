"""
Chuẩn hóa biểu diễn dữ liệu

Mục tiêu:
  1. Chuẩn hóa tên cột → snake_case tiếng Việt không dấu (dễ code, dễ đọc)
  2. Chuẩn hóa giá trị categorical → nhất quán (viết hoa, loại bỏ khoảng trắng thừa)
  3. Phân loại biến → numeric / ordinal / nominal / id / target
  4. Ước lượng số features sau one-hot encoding

Usage:
    from src.data_processing.standardizer import DataStandardizer

    std = DataStandardizer()
    df_std, report = std.standardize(df)
"""

import polars as pl
import unicodedata
import re
from typing import Dict, List, Any, Tuple


# ============================================================
# Ánh xạ tên cột gốc → tên chuẩn hóa
# ============================================================

COLUMN_RENAME_MAP = {
    "id": "id",
    "Gender": "gioi_tinh",
    "Age": "tuoi",
    "City": "thanh_pho",
    "Profession": "nghe_nghiep",
    "Academic Pressure": "ap_luc_hoc_tap",
    "Work Pressure": "ap_luc_cong_viec",
    "CGPA": "cgpa",
    "Study Satisfaction": "hai_long_hoc_tap",
    "Job Satisfaction": "hai_long_cong_viec",
    "Sleep Duration": "thoi_gian_ngu",
    "Dietary Habits": "che_do_an",
    "Degree": "nganh_hoc",
    "Have you ever had suicidal thoughts ?": "y_nghi_tu_tu",
    "Work/Study Hours": "gio_lam_hoc",
    "Financial Stress": "ap_luc_tai_chinh",
    "Family History of Mental Illness": "tien_su_benh_tm",
    "Depression": "tram_cam",
}


# ============================================================
# Giá trị chuẩn hóa cho các biến categorical quan trọng
# ============================================================

VALUE_NORMALIZE_MAP = {
    "gioi_tinh": {
        "Male": "Nam",
        "Female": "Nu",
    },
    "thoi_gian_ngu": {
        "Less than 5 hours": "Duoi_5h",
        "5-6 hours": "5_6h",
        "7-8 hours": "7_8h",
        "More than 8 hours": "Tren_8h",
        "Others": "Khac",
    },
    "che_do_an": {
        "Unhealthy": "Khong_lanh_manh",
        "Moderate": "Trung_binh",
        "Healthy": "Lanh_manh",
        "Others": "Khac",
    },
    "y_nghi_tu_tu": {
        "Yes": "Co",
        "No": "Khong",
    },
    "tien_su_benh_tm": {
        "Yes": "Co",
        "No": "Khong",
    },
}


class DataStandardizer:
    """Chuẩn hóa tên cột, giá trị, và phân loại biến."""

    def __init__(self):
        self.rename_map = COLUMN_RENAME_MAP
        self.value_map = VALUE_NORMALIZE_MAP
        self.classification = {}

    # ==========================================
    # BƯỚC 1: Chuẩn hóa tên cột
    # ==========================================

    def rename_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Đổi tên cột theo ánh xạ định sẵn."""
        rename_dict = {}
        for old_name, new_name in self.rename_map.items():
            if old_name in df.columns:
                rename_dict[old_name] = new_name

        # Cảnh báo cột không mapping
        unmapped = [c for c in df.columns if c not in self.rename_map]
        if unmapped:
            print(f"  ⚠️  Cột không có mapping: {unmapped}")

        return df.rename(rename_dict)

    # ==========================================
    # BƯỚC 2: Chuẩn hóa giá trị categorical
    # ==========================================

    def normalize_values(self, df: pl.DataFrame) -> pl.DataFrame:
        """Chuẩn hóa giá trị trong các biến categorical."""
        for col, val_map in self.value_map.items():
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).replace_strict(val_map, default=pl.col(col))
                )
        return df

    # ==========================================
    # BƯỚC 3: Phân loại biến
    # ==========================================

    def classify_variables(self, df: pl.DataFrame) -> Dict[str, List[str]]:
        """
        Phân loại biến thành các nhóm:
          - id: định danh, không dùng phân tích
          - target: biến mục tiêu
          - numeric: liên tục thực sự (tuổi, CGPA, giờ, ...)
          - ordinal: thang đo thứ tự (áp lực, hài lòng, ...) — lưu Int64 nhưng là ordinal
          - nominal: danh nghĩa (giới tính, thành phố, degree, ...)
        """
        classification = {
            "id": [],
            "target": [],
            "numeric": [],
            "ordinal": [],
            "nominal": [],
        }

        # Hardcoded dựa trên hiểu biết về dataset
        classification["id"] = ["id"]
        classification["target"] = ["tram_cam"]

        # Ordinal: thang đo thứ tự — có thể là String HOẶC Int64
        ordinal_set = {
            "thoi_gian_ngu",       # String: Duoi_5h, 5_6h, 7_8h, Tren_8h
            "che_do_an",           # String: Khong_lanh_manh, Trung_binh, Lanh_manh
            "ap_luc_hoc_tap",      # Int64: 0-5
            "ap_luc_cong_viec",    # Int64: 0-5
            "hai_long_hoc_tap",    # Int64: 0-5
            "hai_long_cong_viec",  # Int64: 0-5
            "ap_luc_tai_chinh",    # Int64: 1-5
        }

        # Nominal: danh nghĩa — String không có thứ tự
        nominal_set = {
            "gioi_tinh",
            "thanh_pho",
            "nghe_nghiep",
            "nganh_hoc",
            "y_nghi_tu_tu",
            "tien_su_benh_tm",
        }

        # Numeric: liên tục thực sự
        numeric_set = {"tuoi", "cgpa", "gio_lam_hoc"}

        for col in df.columns:
            if col in classification["id"] or col in classification["target"]:
                continue
            if col in ordinal_set:
                classification["ordinal"].append(col)
            elif col in nominal_set:
                classification["nominal"].append(col)
            elif col in numeric_set:
                classification["numeric"].append(col)
            else:
                # Fallback: dựa vào dtype
                if df[col].dtype in [pl.Int64, pl.Float64]:
                    classification["numeric"].append(col)
                else:
                    classification["nominal"].append(col)

        self.classification = classification
        return classification

    # ==========================================
    # BƯỚC 4: Ước lượng số features sau encoding
    # ==========================================

    def estimate_feature_count(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Ước lượng số cột trong feature matrix sau khi encoding."""
        estimate = {
            "numeric_features": len(self.classification.get("numeric", [])),
            "ordinal_features": 0,  # Ordinal giữ nguyên hoặc mã hóa 0..N-1
            "nominal_onehot_features": 0,
            "total_estimated": 0,
            "details": {},
        }

        # Ordinal: mã hóa thành 0, 1, 2, ... → 1 feature mỗi biến
        ordinal_cols = self.classification.get("ordinal", [])
        for col in ordinal_cols:
            if col in df.columns:
                n_unique = df[col].n_unique()
                estimate["ordinal_features"] += 1
                estimate["details"][col] = f"ordinal ({n_unique} levels → 1 feature)"

        # Nominal: one-hot encoding → (n_unique - 1) features
        nominal_cols = self.classification.get("nominal", [])
        for col in nominal_cols:
            if col in df.columns:
                n_unique = df[col].n_unique()
                n_features = n_unique - 1  # drop_first=True
                estimate["nominal_onehot_features"] += n_features
                estimate["details"][col] = f"one-hot ({n_unique} levels → {n_features} features)"

        estimate["total_estimated"] = (
            estimate["numeric_features"]
            + estimate["ordinal_features"]
            + estimate["nominal_onehot_features"]
        )

        return estimate

    # ==========================================
    # BƯỚC 5: In báo cáo phân loại
    # ==========================================

    def print_classification_report(self, df: pl.DataFrame) -> None:
        """In báo cáo chi tiết phân loại biến."""
        print()
        print("=" * 80)
        print(" 📊 PHÂN LOẠI BIẾN & ƯỚC LƯỢNG FEATURE MATRIX")
        print("=" * 80)

        # ID
        if self.classification["id"]:
            print(f"\n  🆔 ID ({len(self.classification['id'])}):")
            for col in self.classification["id"]:
                print(f"    • {col}")

        # Target
        if self.classification["target"]:
            print(f"\n  🎯 Target ({len(self.classification['target'])}):")
            for col in self.classification["target"]:
                vc = df[col].value_counts().sort("count", descending=True)
                n_total = df.height
                n_1 = df.filter(pl.col(col) == 1).height
                n_0 = df.filter(pl.col(col) == 0).height
                pct_1 = n_1 / n_total * 100
                print(f"    • {col}: 0={n_0:,} ({100-pct_1:.1f}%), 1={n_1:,} ({pct_1:.1f}%)")

        # Numeric
        numeric_cols = self.classification.get("numeric", [])
        if numeric_cols:
            print(f"\n  🔢 Numeric ({len(numeric_cols)}):")
            for col in numeric_cols:
                mean_val = df[col].mean()
                std_val = df[col].std()
                print(f"    • {col}: mean={mean_val:.2f}, std={std_val:.2f}")

        # Ordinal
        ordinal_cols = self.classification.get("ordinal", [])
        if ordinal_cols:
            print(f"\n  📶 Ordinal ({len(ordinal_cols)}):")
            for col in ordinal_cols:
                vals = df[col].drop_nulls().unique().to_list()
                # Int64 ordinal: sort numerically; String ordinal: try sorted
                if df[col].dtype in [pl.Int64, pl.Float64]:
                    vals = sorted(vals)
                else:
                    try:
                        vals = sorted(vals)
                    except TypeError:
                        pass  # Keep original order
                print(f"    • {col}: {vals}")

        # Nominal
        nominal_cols = self.classification.get("nominal", [])
        if nominal_cols:
            print(f"\n  🏷️  Nominal ({len(nominal_cols)}):")
            for col in nominal_cols:
                n_unique = df[col].n_unique()
                print(f"    • {col}: {n_unique} categories")

        # Feature count estimate
        estimate = self.estimate_feature_count(df)
        print(f"\n  📐 ƯỚC LƯỢNG FEATURE MATRIX:")
        print(f"     Numeric:     {estimate['numeric_features']} features (giữ nguyên)")
        print(f"     Ordinal:     {estimate['ordinal_features']} features (mã hóa 0..N-1)")
        print(f"     Nominal:     {estimate['nominal_onehot_features']} features (one-hot, drop_first)")
        print(f"     ─────────────────────────────────")
        print(f"     TỔNG:        ~{estimate['total_estimated']} features")

    # ==========================================
    # ENTRY POINT: Chạy toàn bộ
    # ==========================================

    def standardize(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Chạy toàn bộ pipeline chuẩn hóa.

        Returns:
            df_standardized: DataFrame đã chuẩn hóa
            report: Dict chứa phân loại + ước lượng
        """
        print()
        print("=" * 80)
        print(" 🔧 CHUẨN HÓA BIỂU DIỄN DỮ LIỆU")
        print("=" * 80)

        # Bước 1: Rename
        print("\n  1️⃣  Chuẩn hóa tên cột...")
        df_std = self.rename_columns(df)
        renamed = {k: v for k, v in self.rename_map.items() if k in df.columns and self.rename_map[k] != k}
        if renamed:
            for old, new in renamed.items():
                print(f"     {old!r} → {new!r}")
        else:
            print("     ✅ Không cần đổi tên")

        # Bước 2: Normalize values
        print("\n  2️⃣  Chuẩn hóa giá trị categorical...")
        df_std = self.normalize_values(df_std)
        for col in self.value_map:
            if col in df_std.columns:
                n_unique = df_std[col].n_unique()
                print(f"     {col}: {n_unique} unique values")

        # Bước 3: Classify
        print("\n  3️⃣  Phân loại biến...")
        self.classify_variables(df_std)

        # Bước 4: Print report
        self.print_classification_report(df_std)

        report = {
            "classification": self.classification,
            "feature_estimate": self.estimate_feature_count(df_std),
        }

        print()
        return df_std, report
