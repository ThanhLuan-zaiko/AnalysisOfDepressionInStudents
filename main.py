"""
Main Pipeline - Phân Tích Trầm Cảm Ở Học Sinh Sinh Viên

Pipeline được thiết kế theo hướng:
  - Phân tầng nguy cơ, KHÔNG chẩn đoán
  - Ưu tiên tính giải thích, hiệu chuẩn, và kiểm soát rủi ro đạo đức
  - Mô hình chỉ là công cụ hỗ trợ, không thay thế đánh giá lâm sàng

Usage:
    uv run python main.py --eda          # Giai đoạn 1: EDA + xác định bối cảnh
    uv run python main.py --stats        # Giai đoạn 4: EDA + kiểm định thống kê
    uv run python main.py --models       # Giai đoạn 7-8: Xây dựng & đánh giá mô hình
    uv run python main.py --full         # Chạy toàn bộ pipeline
    uv run python main.py --sample       # Chạy với sample data (test)
"""

import sys
import argparse
import logging
from pathlib import Path
import polars as pl
import numpy as np
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Setup logging
from src.utils import setup_logging, Timer, print_device_info

logger = setup_logging(level="INFO", log_file="logs/analysis.log")

# Đường dẫn dataset thực tế
DATASET_PATH = "Student_Depression_Dataset.csv"

# Các cột cần loại khỏi phân tích (phương sai thấp / định danh)
EXCLUDED_COLUMNS = [
    "id",                    # Identifier
    "Profession",            # 99.9% "Student"
    "Work Pressure",         # 99.99% = 0
    "Job Satisfaction",      # 99.98% = 0
]


# ==========================================
# GIAI ĐOẠN 0: Xác định bối cảnh & giới hạn an toàn
# ==========================================

def print_ethical_boundaries():
    """In cảnh báo đạo đức và giới hạn sử dụng."""
    print("=" * 80)
    print(" 📋 GIAI ĐOẠN 0: BỐI CẢNH & GIỚI HẠN AN TOÀN")
    print("=" * 80)
    print("""
⚠️  CẢNH BÁO ĐẠO ĐỨC & GIỚI HẠN SỬ DỤNG
═══════════════════════════════════════════════════════════════

Dự án này KHÔNG nhằm mục đích:
  ❌ Chẩn đoán trầm cảm thay thế chuyên gia y tế
  ❌ Tự động gán nhãn bệnh lý cho học sinh/sinh viên
  ❌ Làm căn cứ duy nhất cho quyết định hành chính, kỷ luật
     hoặc phân loại học sinh

Dự án này CHỈ nhằm mục đích:
  ✅ Ước lượng mức độ nguy cơ để hỗ trợ sàng lọc
  ✅ Xác định các yếu tố liên quan mạnh đến trầm cảm
  ✅ Hỗ trợ phát hiện sớm các trường hợp cần theo dõi thêm
  ✅ Gợi ý mức độ ưu tiên trong quy trình hỗ trợ tâm lý học đường
  ✅ Kích hoạt bước đánh giá tiếp theo bởi NGƯỜI CÓ CHUYÊN MÔN

Mô hình (nếu được xây dựng) là CÔNG CỤ HỖ TRỢ,
không thay thế quy trình đánh giá lâm sàng.
═══════════════════════════════════════════════════════════════
""")


# ==========================================
# GIAI ĐOẠN 1: Trực quan hóa dữ liệu
# ==========================================

def run_eda(df: pl.DataFrame, output_dir: str = "results/visualizations/"):
    """
    Giai đoạn 1: Trực quan hóa dữ liệu, loại bỏ cột không cần thiết,
    sinh báo cáo EDA hoàn chỉnh.
    """
    from src.visualization import ExploratoryAnalyzer

    print()
    print("=" * 80)
    print(" 🔍 GIAI ĐOẠN 1: TRỰC QUAN HÓA & KHÁM PHÁ DỮ LIỆU")
    print("=" * 80)

    # Thông báo cột bị loại
    cols_to_drop = [c for c in EXCLUDED_COLUMNS if c in df.columns]
    if cols_to_drop:
        print(f"\n🚫 Cột bị loại khỏi phân tích: {', '.join(cols_to_drop)}")
        print("   (Lý do: phương sai quá thấp hoặc là định danh)")

    eda = ExploratoryAnalyzer()

    results = eda.run_full_eda(
        df,
        output_dir=output_dir,
        save_html=True,
        save_report=True,
    )

    return results


# ==========================================
# GIAI ĐOẠN 2-3: Rà soát dữ liệu & chuẩn hóa
# ==========================================

def run_data_review(df: pl.DataFrame) -> dict:
    """
    Giai đoạn 2-3: Rà soát dữ liệu, kiểm soát chất lượng,
    chuẩn hóa biểu diễn.
    """
    print()
    print("=" * 80)
    print(" 🔎 GIAI ĐOẠN 2-3: RÀ SOÁT DỮ LIỆU & CHUẨN HÓA")
    print("=" * 80)

    review = {
        "original_shape": df.shape,
        "columns_dropped": [],
        "columns_kept": [],
        "missing_values": {},
        "rare_categories": {},
        "warnings": [],
    }

    # Loại cột không cần thiết
    cols_to_drop = [c for c in EXCLUDED_COLUMNS if c in df.columns]
    df_review = df.drop(cols_to_drop)
    review["columns_dropped"] = cols_to_drop
    review["columns_kept"] = df_review.columns

    # Missing values
    for col in df_review.columns:
        null_count = df_review[col].null_count()
        if null_count > 0:
            review["missing_values"][col] = null_count

    # Rare categories
    for col in df_review.select(pl.col(pl.String)).columns:
        n_unique = df_review[col].n_unique()
        if n_unique <= 20:
            vc = df_review[col].value_counts().sort("count", descending=True)
            for row in vc.iter_rows(named=True):
                pct = row["count"] / df_review.height * 100
                if pct < 1:
                    review["rare_categories"].setdefault(col, []).append({
                        "category": row[col],
                        "count": row["count"],
                        "pct": round(pct, 2),
                    })

    # Cảnh báo
    if "Have you ever had suicidal thoughts ?" in df_review.columns:
        suicidal_col = "Have you ever had suicidal thoughts ?"
        yes_count = df_review.filter(pl.col(suicidal_col) == "Yes").height
        yes_pct = yes_count / df_review.height * 100
        review["warnings"].append(
            f"Tỷ lệ 'Suicidal thoughts = Yes' = {yes_pct:.1f}% — "
            "RẤT CAO, cảnh báo synthetic data hoặc label leakage."
        )

    if "Depression" in df_review.columns:
        dep_1 = df_review.filter(pl.col("Depression") == 1).height
        dep_1_pct = dep_1 / df_review.height * 100
        review["warnings"].append(
            f"Class imbalance: lớp 1 = {dep_1_pct:.1f}% — "
            "cần stratified split và class weights."
        )

    # In báo cáo chi tiết
    print()
    print("=" * 80)
    print(" 📋 BÁO CÁO RÀ SOÁT DỮ LIỆU")
    print("=" * 80)

    # 1. Tổng quan
    print(f"\n  Shape gốc:        {review['original_shape']}")
    print(f"  Shape sau lọc:    {df_review.shape}")

    # 2. Biến hằng số / phương sai thấp
    if review["columns_dropped"]:
        print(f"\n  🚫 Biến đã loại (phương sai quá thấp / định danh):")
        for col in review["columns_dropped"]:
            if col in df.columns:
                vc = df[col].value_counts().sort("count", descending=True)
                top_row = vc.row(0, named=True)
                top_val = top_row[col]
                top_count = top_row["count"]
                top_pct = top_count / df.height * 100
                n_unique = df[col].n_unique()
                print(f"    • {col}: {n_unique} unique, '{top_val}' chiếm {top_pct:.1f}%")

    # 3. Biến gần-hằng-số (gợi ý loại thêm)
    near_constant = []
    for col in df_review.columns:
        if col in ["Depression"]:
            continue
        n_unique = df_review[col].n_unique()
        if n_unique <= 2:
            vc = df_review[col].value_counts().sort("count", descending=True)
            top_row = vc.row(0, named=True)
            top_pct = top_row["count"] / df_review.height * 100
            if top_pct > 95:
                near_constant.append({
                    "column": col,
                    "unique": n_unique,
                    "top_value": str(top_row[col]),
                    "top_pct": round(top_pct, 1),
                })

    if near_constant:
        print(f"\n  ⚠️  Biến gần-hằng-số (gợi ý xem xét loại):")
        for info in near_constant:
            print(f"    • {info['column']}: '{info['top_value']}' = {info['top_pct']}%")

    # 4. Missing values
    if review["missing_values"]:
        print(f"\n  ⚠️  Missing values:")
        for col, count in review["missing_values"].items():
            pct = count / df_review.height * 100
            print(f"    • {col}: {count:,} ({pct:.1f}%)")
    else:
        print(f"\n  ✅ Không có missing values")

    # 5. Rare categories
    if review["rare_categories"]:
        print(f"\n  ⚠️  Rare categories (< 1%):")
        for col, cats in review["rare_categories"].items():
            for cat in cats:
                print(f"    • {col}: '{cat['category']}' ({cat['count']} = {cat['pct']}%)")

    # 6. Cảnh báo
    if review["warnings"]:
        print(f"\n  🔔 Cảnh báo:")
        for w in review["warnings"]:
            print(f"    • {w}")

    # 7. Danh sách cột giữ lại
    print(f"\n  ✅ Biến giữ lại để phân tích ({len(review['columns_kept'])}):")
    print(f"     {', '.join(review['columns_kept'])}")

    print()
    return review


# ==========================================
# HÀM CHÍNH
# ==========================================

def load_dataset() -> pl.DataFrame:
    """Đọc dataset chính."""
    csv_path = Path(DATASET_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {DATASET_PATH}. "
            "Đảm bảo file nằm cùng cấp với main.py."
        )
    print(f"📂 Đang đọc: {DATASET_PATH}")
    df = pl.read_csv(str(csv_path))
    print(f"   ✅ {df.height:,} dòng × {df.width} cột")
    return df


def main(
    run_ethical: bool = True,
    run_eda_flag: bool = False,
    run_stats: bool = False,
    run_models: bool = False,
    run_leakage: bool = False,
    run_review: bool = False,
    run_standardize: bool = False,
    run_famd: bool = False,
    run_split: bool = False,
    conservative: bool = False,
):
    """
    Main analysis pipeline — chạy theo giai đoạn tùy chọn.
    """
    print("=" * 80)
    print(" 🧠 PHÂN TÍCH TRẦM CẢM Ở HỌC SINH SINH VIÊN")
    print("=" * 80)
    print()

    # ---- Giai đoạn 0: Ethical boundaries ----
    if run_ethical:
        print_ethical_boundaries()

    # ---- Load data ----
    with Timer("Load data"):
        df = load_dataset()

    # ---- Giai đoạn 1: EDA ----
    if run_eda_flag:
        with Timer("Exploratory Data Analysis"):
            eda_results = run_eda(df)
            print(f"\n  ✅ {len(eda_results['figures'])} biểu đồ đã lưu vào results/visualizations/")

    # ---- Giai đoạn 2-3: Data review ----
    review = None
    if run_review or run_eda_flag or run_stats or run_models or run_leakage:
        with Timer("Data Review"):
            review = run_data_review(df)

        # Quality gate: cảnh báo nếu có vấn đề nghiêm trọng
        if review and review.get("warnings"):
            for w in review["warnings"]:
                print(f"\n  ⚠️  [Quality Gate] {w}")
        if review and review.get("missing_values"):
            missing = review["missing_values"]
            n_missing = len(missing)
            total_missing = sum(missing.values())
            print(f"\n  📊 [Quality Gate] {n_missing} cột có tổng {total_missing:,} giá trị thiếu")
        if review and review.get("rare_categories"):
            rare = review["rare_categories"]
            print(f"\n  🔍 [Quality Gate] {len(rare)} cột có category hiếm (<1%)")

    # Lưu review kết quả để các giai đoạn sau có thể tham chiếu
    # (ví dụ: models có thể kiểm tra data quality trước khi train)

    # ---- Giai đoạn: Label leakage investigation ----
    if run_leakage:
        with Timer("Label Leakage Investigation"):
            run_leakage_investigation(df)

    # ---- Giai đoạn chuẩn hóa (auto chạy kèm khi --review) ----
    if run_standardize or run_review:
        with Timer("Data Standardization"):
            from src.data_processing import DataStandardizer
            std = DataStandardizer()
            df_std, std_report = std.standardize(df)
            print(f"\n  ✅ Feature estimate: ~{std_report['feature_estimate']['total_estimated']} features")

    # ---- Giai đoạn FAMD ----
    if run_famd:
        with Timer("FAMD Analysis"):
            run_famd_analysis(df)

    # ---- Giai đoạn chia tập Train/Test ----
    if run_split:
        with Timer("Train/Test Split"):
            run_split_analysis(df)

    # ---- Giai đoạn 4: Statistical analysis ----
    if run_stats:
        with Timer("Statistical Analysis"):
            run_statistical_analysis(df)

    # ---- Giai đoạn 7-8: Models ----
    if run_models:
        with Timer("Machine Learning"):
            run_ml_pipeline(df, conservative=conservative)

    # ---- Summary ----
    print()
    print("=" * 80)
    print(" ✅ HOÀN THÀNH")
    print("=" * 80)
    print()
    print("📁 Kết quả:")
    print("   • Visualizations: results/visualizations/")
    print("   • Logs: logs/analysis.log")
    print()
    print("⚠️  LƯU Ý: Mọi kết quả là QUAN HỆ LIÊN QUAN, không phải nhân quả.")
    print("   Mô hình chỉ là công cụ HỖ TRỢ, không thay thế đánh giá lâm sàng.")
    print()


def run_statistical_analysis(df: pl.DataFrame):
    """Giai đoạn 4: Kiểm định thống kê + cỡ ảnh hưởng (effect size)."""
    import pingouin as pg
    import pandas as pd

    print("\n" + "=" * 80)
    print(" 📊 GIAI ĐOẠN 4: KIỂM ĐỊNH THỐNG KÊ + CỠ ẢNH HƯỞNG")
    print("=" * 80)

    # Loại cột không cần thiết
    df_a = df.drop([c for c in EXCLUDED_COLUMNS if c in df.columns])

    # ============================================================
    # 1. Thống kê mô tả theo nhóm Trầm cảm
    # ============================================================
    print("\n" + "=" * 60)
    print(" 1️⃣  THỐNG KÊ MÔ TẢ THEO NHÓM TRẦM CẢM")
    print("=" * 60)

    n_total = df_a.height
    n_dep = df_a.filter(pl.col("Depression") == 1).height
    n_nodep = df_a.filter(pl.col("Depression") == 0).height
    print(f"\n  Tổng: {n_total:,} | Có TC: {n_dep:,} ({n_dep/n_total*100:.1f}%) | Không TC: {n_nodep:,} ({n_nodep/n_total*100:.1f}%)")

    # Numeric variables
    numeric_cols = [c for c in df_a.select(pl.selectors.numeric()).columns if c != "Depression"]
    if numeric_cols:
        print(f"\n  {'Biến':<40s} {'Nhóm':>6s} {'Mean':>8s} {'SD':>8s} {'Median':>8s} {'Min':>6s} {'Max':>6s}")
        print(f"  {'─'*40} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")
        for col in numeric_cols:
            for dep_val, label in [(0, "Không"), (1, "Có TC")]:
                subset = df_a.filter(pl.col("Depression") == dep_val)[col].drop_nulls()
                if len(subset) == 0:
                    continue
                m = subset.mean()
                s = subset.std()
                med = subset.median()
                mn = subset.min()
                mx = subset.max()
                label_short = f"{col}" if dep_val == 0 else ""
                if dep_val == 0:
                    print(f"  {col:<40s} {label:>6s} {m:>8.2f} {s:>8.2f} {med:>8.2f} {mn:>6.0f} {mx:>6.0f}")
                else:
                    print(f"  {'':<40s} {label:>6s} {m:>8.2f} {s:>8.2f} {med:>8.2f} {mn:>6.0f} {mx:>6.0f}")
            print()

    # ============================================================
    # 2. Kiểm định: Numeric variables (Mann-Whitney U / t-test)
    # ============================================================
    print("\n" + "=" * 60)
    print(" 2️⃣  SO SÁNH BIẾN SỐ GIỮA 2 NHÓM TRẦM CẢM")
    print("=" * 60)

    df_pd = df_a.to_pandas()

    print(f"\n  {'Biến':<40s} {'Test':>14s} {'U':>10s} {'p':>10s} {"Cohen d":>10s} {'RBC':>10s} {'Ý nghĩa':>10s}")
    print(f"  {'─'*40} {'─'*14} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    for col in numeric_cols:
        try:
            group0 = df_pd[df_pd["Depression"] == 0][col].dropna()
            group1 = df_pd[df_pd["Depression"] == 1][col].dropna()

            # Mann-Whitney U
            u_test = pg.mwu(x=group0, y=group1)
            u_val = u_test["U_val"].values[0]
            p_val = u_test["p_val"].values[0]

            # Cohen's d
            from pingouin.effsize import compute_effsize
            cohend = compute_effsize(group1, group0, eftype="cohen")

            # RBC (rank-biserial correlation) as effect size alternative
            rbc = u_test["RBC"].values[0] if "RBC" in u_test.columns else float("nan")
            rbc_str = f"{rbc:>8.3f}" if not np.isnan(rbc) else "       -"

            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

            abs_d = abs(cohend)
            if abs_d < 0.2:
                es_label = "Rất nhỏ"
            elif abs_d < 0.5:
                es_label = "Nhỏ"
            elif abs_d < 0.8:
                es_label = "TB"
            else:
                es_label = "LỚN"

            print(f"  {col:<40s} {'Mann-Whitney':>14s} {u_val:>10.0f} {p_val:>10.4f} {cohend:>10.3f} {rbc_str:>10s} {sig:>4s} [{es_label}]")
        except Exception as e:
            print(f"  {col:<40s} {'ERROR':>14s} {str(e)[:30]:>30s}")

    print(f"\n  Chú thích: p < 0.05 (*), < 0.01 (**), < 0.001 (***); ns = không ý nghĩa")
    print(f"  Cohen d: <0.2 rất nhỏ, <0.5 nhỏ, <0.8 trung bình, >=0.8 LỚN")
    print(f"  RBC: Rank-biserial correlation (effect size phi hạng)")

    # ============================================================
    # 3. Kiểm định: Categorical variables (Chi-square + Cramer's V)
    # ============================================================
    print("\n" + "=" * 60)
    print(" 3️⃣  LIÊN HỆ BIẾN PHÂN LOẠI VỚI TRẦM CẢM (Chi-square)")
    print("=" * 60)

    cat_cols = [c for c in df_a.select(pl.col(pl.String)).columns if c != "Depression"]

    print(f"\n  {'Biến':<40s} {'χ²':>10s} {'df':>6s} {'p':>10s} {"Cramer V":>10s} {'OR (max)':>10s} {'Ý nghĩa':>10s}")
    print(f"  {'─'*40} {'─'*10} {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    for col in cat_cols:
        try:
            # Chi-square test (scipy)
            from scipy.stats import chi2_contingency
            ct = pd.crosstab(df_pd[col], df_pd["Depression"])
            chi2_stat, p_val, dof, expected = chi2_contingency(ct)

            # Cramer's V
            n = ct.sum().sum()
            phi2 = chi2_stat / n
            k = min(ct.shape) - 1
            cramers_v = np.sqrt(phi2 / k) if k > 0 else 0.0

            # Odds Ratio: lấy mức có OR lớn nhất (so với reference = hàng đầu)
            if ct.shape[1] == 2:
                or_values = []
                ref_a = ct.iloc[0, 1] + 0.5
                ref_b = ct.iloc[0, 0] + 0.5
                for i in range(ct.shape[0]):
                    a = ct.iloc[i, 1] + 0.5
                    b = ct.iloc[i, 0] + 0.5
                    if i == 0:
                        or_values.append(1.0)
                    else:
                        or_values.append((a * ref_b) / (b * ref_a))
                max_or = max(or_values) if or_values else float("nan")
                min_or = min(or_values) if or_values else float("nan")
                # Report the more extreme OR
                max_or = max(abs(max_or), abs(min_or)) if min_or < 1 else max_or
            else:
                max_or = float("nan")

            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            or_str = f"{max_or:>8.2f}" if not np.isnan(max_or) else "       -"

            abs_v = abs(cramers_v)
            if abs_v < 0.1:
                es_label = "Rất nhỏ"
            elif abs_v < 0.3:
                es_label = "Nhỏ"
            elif abs_v < 0.5:
                es_label = "TB"
            else:
                es_label = "LỚN"

            print(f"  {col:<40s} {chi2_stat:>10.2f} {dof:>6d} {p_val:>10.4f} {cramers_v:>10.3f} {or_str:>10s} {sig:>4s} [{es_label}]")
        except Exception as e:
            print(f"  {col:<40s} {'ERROR':>14s} {str(e)[:30]:>30s}")

    print(f"\n  Chú thích: Cramer's V: <0.1 rất nhỏ, <0.3 nhỏ, <0.5 trung bình, >=0.5 LỚN")

    # ============================================================
    # 4. Tương quan giữa các biến số
    # ============================================================
    print("\n" + "=" * 60)
    print(" 4️⃣  TƯƠNG QUAN VỚI BIẾN TRẦM CẢM (Spearman)")
    print("=" * 60)

    if len(numeric_cols) >= 2:
        print(f"\n  {'Biến':<40s} {'rho':>8s} {'p':>10s} {'n':>8s} {'95% CI':>18s} {'Ý nghĩa':>10s}")
        print(f"  {'─'*40} {'─'*8} {'─'*10} {'─'*8} {'─'*18} {'─'*10}")

        for col in numeric_cols:
            try:
                corr = pg.corr(
                    x=df_pd[col],
                    y=df_pd["Depression"],
                    method="spearman"
                )
                rho = corr["r"].values[0]
                p_val = corr["p_val"].values[0]
                n = corr["n"].values[0]
                ci = corr["CI95"].values[0]
                ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]"

                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

                abs_r = abs(rho)
                if abs_r < 0.1:
                    es_label = "Rất nhỏ"
                elif abs_r < 0.3:
                    es_label = "Nhỏ"
                elif abs_r < 0.5:
                    es_label = "TB"
                else:
                    es_label = "MẠNH"

                print(f"  {col:<40s} {rho:>8.3f} {p_val:>10.4f} {n:>8.0f} {ci_str:>18s} {sig:>4s} [{es_label}]")
            except Exception as e:
                print(f"  {col:<40s} {'ERROR':>14s} {str(e)[:30]:>30s}")


def run_split_analysis(df: pl.DataFrame):
    """Chia tập train/test stratified + báo cáo cân bằng."""
    from src.ml_models import StratifiedSplitter

    splitter = StratifiedSplitter()

    train_df, test_df, report = splitter.split(
        df,
        test_size=0.2,
        excluded_cols=EXCLUDED_COLUMNS,
        target_col="Depression",
        verbose=True,
    )

    # Lưu report dưới dạng JSON
    import json
    from pathlib import Path

    output_path = Path("results/")
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert numpy types
    serializable = json.loads(json.dumps(report, default=str))

    report_path = output_path / "split_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Báo cáo đã lưu: {report_path}")


def run_famd_analysis(df: pl.DataFrame):
    """FAMD — Giảm chiều dữ liệu hỗn hợp + biểu đồ."""
    from src.ml_models import FAMDAnalyzer

    analyzer = FAMDAnalyzer()

    results = analyzer.run_famd(
        df,
        n_components=10,
        excluded_cols=EXCLUDED_COLUMNS,
        target_col="Depression",
        verbose=True,
    )

    # Lưu biểu đồ
    print("\n  📊 Lưu biểu đồ FAMD...")
    saved = analyzer.save_all_plots(
        output_dir="results/visualizations/",
        save_html=True,
    )
    for name, path in saved.items():
        print(f"     ✅ {name}: {path}")


def run_leakage_investigation(df: pl.DataFrame):
    """Điều tra rò rỉ nhãn từ Suicidal thoughts."""
    from src.ml_models import LabelLeakageInvestigator

    investigator = LabelLeakageInvestigator()
    results = investigator.run_full_investigation(df)

    # Lưu kết quả
    import json
    from pathlib import Path
    from datetime import datetime

    output_path = Path("results/")
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON
    serializable = {}
    for key, val in results.items():
        if key == "stress_test":
            serializable[key] = val  # Already polars records
        elif key == "cross_tab":
            serializable[key] = val
        elif key == "feature_importance_comparison":
            serializable[key] = val
        elif key == "synthetic_check":
            serializable[key] = {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in val.items()
            }

    report_path = output_path / "leakage_investigation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  ✅ Báo cáo đã lưu: {report_path}")


def run_ml_pipeline(df: pl.DataFrame, conservative: bool = False):
    """Giai đoạn 7-9: Xây dựng mô hình + Fairness + Threshold.

    Args:
        df: DataFrame đã làm sạch.
        conservative: Nếu True → dùng Phiên bản A (không có Suicidal thoughts).
                      Nếu False → dùng Phiên bản B (mặc định, có Suicidal thoughts).
    """
    from src.ml_models import DepressionRiskModeler

    include_suicidal = not conservative
    version_label = "A (bảo thủ)" if conservative else "B (đầy đủ)"
    version_emoji = "🅰️" if conservative else "🅱️"

    # ================================================================
    print(f"\n" + "=" * 80)
    print(f" {version_emoji}  PHIÊN BẢN {version_label}: {'KHÔNG có' if conservative else 'CÓ'} 'Suicidal thoughts'")
    if conservative:
        print("     (An toàn hơn, không rủi ro rò rỉ nhãn)")
    else:
        print("     (Hiệu năng cao nhất, nhưng OR = 12.388 cho Suicidal thoughts)")
    print("=" * 80)

    modeler = DepressionRiskModeler()
    results = modeler.run_full_pipeline(
        df,
        include_suicidal=include_suicidal,
        output_dir="results/",
    )

    # ================================================================
    # GIAI ĐOẠN 9: FAIRNESS ANALYSIS
    # ================================================================
    modeler.print_fairness_report(df, include_suicidal=include_suicidal)

    # ================================================================
    # GIAI ĐOẠN 8b: THRESHOLD ANALYSIS
    # ================================================================
    print()
    print("=" * 80)
    print(" 🎯 GIAI ĐOẠN 8b: CHỌN NGƯỠNG QUYẾT ĐỊNH")
    print("=" * 80)

    for model_name in ["logistic", "catboost"]:
        if model_name in modeler.models:
            modeler.print_threshold_report(df, model_name, include_suicidal=include_suicidal)


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phân Tích Trầm Cảm — Pipeline có kiểm soát đạo đức"
    )
    parser.add_argument("--eda", action="store_true",
                        help="Giai đoạn 1: Trực quan hóa & EDA")
    parser.add_argument("--stats", action="store_true",
                        help="Giai đoạn 4: Kiểm định thống kê")
    parser.add_argument("--models", action="store_true",
                        help="Giai đoạn 7-8: Xây dựng mô hình")
    parser.add_argument("--leakage", action="store_true",
                        help="Điều tra rò rỉ nhãn (Suicidal thoughts)")
    parser.add_argument("--full", action="store_true",
                        help="Chạy toàn bộ pipeline")
    parser.add_argument("--no-ethical", action="store_true",
                        help="Bỏ qua cảnh báo đạo đức (không khuyến nghị)")
    parser.add_argument("--conservative", action="store_true",
                        help="Dùng Phiên bản A — không có 'Suicidal thoughts'")
    parser.add_argument("--review", action="store_true",
                        help="Giai đoạn 2-3: Rà soát dữ liệu, phát hiện biến hằng số, missing, rare categories")
    parser.add_argument("--standardize", action="store_true",
                        help="Chuẩn hóa tên cột, giá trị categorical, phân loại biến, ước lượng feature matrix")
    parser.add_argument("--famd", action="store_true",
                        help="FAMD — Giảm chiều dữ liệu hỗn hợp (numeric + categorical), biểu đồ, top biến đóng góp")
    parser.add_argument("--split", action="store_true",
                        help="Chia tập train/test stratified — kiểm tra cân bằng phân phối")

    args = parser.parse_args()

    # Default: nếu không có flag nào → chạy EDA
    any_flag = args.eda or args.stats or args.models or args.leakage or args.full or args.review or args.standardize or args.famd or args.split

    try:
        if args.full:
            main(
                run_ethical=not args.no_ethical,
                run_eda_flag=True,
                run_stats=True,
                run_models=True,
                run_review=True,
                run_standardize=True,
                run_famd=False,
                run_split=False,
                conservative=args.conservative,
            )
        elif any_flag:
            main(
                run_ethical=not args.no_ethical,
                run_eda_flag=args.eda,
                run_stats=args.stats,
                run_models=args.models,
                run_leakage=args.leakage,
                run_review=args.review,
                run_standardize=args.standardize,
                run_famd=args.famd,
                run_split=args.split,
                conservative=args.conservative,
            )
        else:
            # Default: chỉ chạy EDA + ethical
            main(
                run_ethical=True,
                run_eda_flag=True,
                run_stats=False,
                run_models=False,
            )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")
        print("📋 Details in logs/analysis.log")
        import traceback
        traceback.print_exc()
        sys.exit(1)
