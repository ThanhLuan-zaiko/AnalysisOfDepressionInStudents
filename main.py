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

    # In báo cáo
    print(f"\n  Shape gốc:        {review['original_shape']}")
    print(f"  Shape sau lọc:    {df_review.shape}")
    print(f"  Cột đã loại:      {review['columns_dropped']}")

    if review["missing_values"]:
        print(f"\n  ⚠️  Missing values:")
        for col, count in review["missing_values"].items():
            print(f"    {col}: {count}")
    else:
        print(f"\n  ✅ Không có missing values")

    if review["rare_categories"]:
        print(f"\n  ⚠️  Rare categories (< 1%):")
        for col, cats in review["rare_categories"].items():
            for cat in cats:
                print(f"    {col}: '{cat['category']}' ({cat['count']} = {cat['pct']}%)")

    if review["warnings"]:
        print(f"\n  ⚠️  Cảnh báo:")
        for w in review["warnings"]:
            print(f"    • {w}")

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
    if run_eda_flag or run_stats or run_models or run_leakage:
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
    """Giai đoạn 4: EDA + kiểm định thống kê."""
    from src.statistical_analysis import StatisticalAnalyzer

    print("\n" + "=" * 80)
    print(" 📊 GIAI ĐOẠN 4: KIỂM ĐỊNH THỐNG KÊ")
    print("=" * 80)

    # Loại cột không cần thiết
    df_analysis = df.drop([c for c in EXCLUDED_COLUMNS if c in df.columns])

    analyzer = StatisticalAnalyzer()

    # Descriptive statistics theo nhóm Depression
    if "Depression" in df_analysis.columns:
        print("\n📋 Thống kê mô tả theo nhóm Trầm cảm:")
        for dep_val in df_analysis["Depression"].unique():
            label = "Có trầm cảm" if dep_val == 1 else "Không trầm cảm"
            subset = df_analysis.filter(pl.col("Depression") == dep_val)
            print(f"\n  --- {label} (Depression={dep_val}) ---")
            numeric_cols = subset.select(pl.selectors.numeric()).columns
            if numeric_cols:
                desc = subset.select(numeric_cols).describe()
                print(desc)


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

    args = parser.parse_args()

    # Default: nếu không có flag nào → chạy EDA
    any_flag = args.eda or args.stats or args.models or args.leakage or args.full

    try:
        if args.full:
            main(
                run_ethical=not args.no_ethical,
                run_eda_flag=True,
                run_stats=True,
                run_models=True,
                conservative=args.conservative,
            )
        elif any_flag:
            main(
                run_ethical=not args.no_ethical,
                run_eda_flag=args.eda,
                run_stats=args.stats,
                run_models=args.models,
                run_leakage=args.leakage,
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
