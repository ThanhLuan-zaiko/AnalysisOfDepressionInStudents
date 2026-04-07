"""
Stratified Train-Test Split — Chia tập dữ liệu phân tầng

Đảm bảo tỷ lệ lớp (class ratio) và phân phối các biến nhân khẩu học
(giới tính, độ tuổi, tiền sử bệnh) được giữ nguyên giữa train và test.

Usage:
    from src.ml_models.stratified_split import StratifiedSplitter

    splitter = StratifiedSplitter()
    train_df, test_df, report = splitter.split(df, test_size=0.2)
"""

import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class StratifiedSplitter:
    """
    Chia tập train/test phân tầng đa biến.

    Không chỉ stratify theo target (Depression), mà còn kiểm tra
    cân bằng phân phối các biến nhân khẩu học giữa train/test.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def split(
        self,
        df: pl.DataFrame,
        test_size: float = 0.2,
        target_col: str = "Depression",
        stratify_cols: Optional[List[str]] = None,
        excluded_cols: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, Any]]:
        """
        Chia tập train/test stratified.

        Args:
            df: DataFrame gốc
            test_size: Tỷ lệ tập test (mặc định 0.2)
            target_col: Cột target để stratify
            stratify_cols: Các cột bổ sung để kiểm tra cân bằng
            excluded_cols: Cột loại khỏi phân tích
            verbose: In báo cáo

        Returns:
            (train_df, test_df, report)
        """
        if excluded_cols:
            df_work = df.drop([c for c in excluded_cols if c in df.columns])
        else:
            df_work = df

        if target_col not in df_work.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        # Auto-detect stratify columns
        if stratify_cols is None:
            stratify_cols = self._auto_detect_stratify_cols(df_work, target_col)

        # Convert sang pandas
        df_pd = df_work.to_pandas()

        # Tạo composite stratification key
        stratify_key = df_pd[target_col].astype(str)
        for col in stratify_cols:
            if col in df_pd.columns and col != target_col:
                stratify_key = stratify_key + "_" + df_pd[col].astype(str)

        # Gộp nhóm quá nhỏ (< 2 samples) để tránh lỗi stratify
        group_counts = stratify_key.value_counts()
        small_groups = group_counts[group_counts < 2]
        if len(small_groups) > 0:
            for group_name in small_groups.index:
                mask = stratify_key == group_name
                stratify_key[mask] = df_pd.loc[mask, target_col].astype(str)

        # Stratified split
        train_idx, test_idx = train_test_split(
            np.arange(len(df_pd)),
            test_size=test_size,
            stratify=stratify_key,
            random_state=self.random_state,
        )

        # Lọc bằng row index
        df_with_idx = df_work.with_row_index("__row_idx__")
        train_df = df_with_idx.filter(pl.col("__row_idx__").is_in(train_idx.tolist())).drop("__row_idx__")
        test_df = df_with_idx.filter(pl.col("__row_idx__").is_in(test_idx.tolist())).drop("__row_idx__")

        # Báo cáo
        report = self._generate_report(df_work, train_df, test_df, target_col, stratify_cols)

        if verbose:
            self._print_report(train_df, test_df, target_col, report)

        return train_df, test_df, report

    def _auto_detect_stratify_cols(
        self, df: pl.DataFrame, target_col: str
    ) -> List[str]:
        """Tự động phát hiện biến nhân khẩu học để stratify."""
        stratify_cols = []

        # Giới tính
        for col in df.columns:
            if col.lower() in ["gender", "gioi_tinh"]:
                if df[col].n_unique() <= 10:
                    stratify_cols.append(col)
                break

        # Tiền sử bệnh tâm thần
        for col in df.columns:
            if "family_history" in col.lower() or "tien_su" in col.lower():
                if df[col].n_unique() <= 10:
                    stratify_cols.append(col)
                break

        return stratify_cols

    def _generate_report(
        self,
        df_orig: pl.DataFrame,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
        target_col: str,
        stratify_cols: List[str],
    ) -> Dict[str, Any]:
        """Tạo báo cáo so sánh phân phối giữa train/test."""
        report = {
            "sizes": {
                "total": df_orig.height,
                "train": train_df.height,
                "test": test_df.height,
                "train_pct": round(train_df.height / df_orig.height * 100, 1),
                "test_pct": round(test_df.height / df_orig.height * 100, 1),
            },
            "target_distribution": {},
            "stratify_cols_balance": {},
            "ks_statistics": {},
        }

        # Target distribution
        for label in [0, 1]:
            n_total = df_orig.filter(pl.col(target_col) == label).height
            n_train = train_df.filter(pl.col(target_col) == label).height
            n_test = test_df.filter(pl.col(target_col) == label).height
            report["target_distribution"][str(label)] = {
                "total": {"count": n_total, "pct": round(n_total / df_orig.height * 100, 1)},
                "train": {"count": n_train, "pct": round(n_train / train_df.height * 100, 1)},
                "test": {"count": n_test, "pct": round(n_test / test_df.height * 100, 1)},
            }

        # Balance check cho stratify cols
        for col in stratify_cols:
            if col not in df_orig.columns:
                continue
            values = df_orig[col].unique().to_list()
            balance = {}
            for val in values:
                n_total = df_orig.filter(pl.col(col) == val).height
                n_train = train_df.filter(pl.col(col) == val).height
                n_test = test_df.filter(pl.col(col) == val).height
                balance[str(val)] = {
                    "total_pct": round(n_total / df_orig.height * 100, 1),
                    "train_pct": round(n_train / train_df.height * 100, 1),
                    "test_pct": round(n_test / test_df.height * 100, 1),
                    "max_diff": round(
                        max(
                            abs(n_train / train_df.height - n_total / df_orig.height),
                            abs(n_test / test_df.height - n_total / df_orig.height),
                        ) * 100, 1
                    ),
                }
            report["stratify_cols_balance"][col] = balance

        # KS test cho biến số
        numeric_cols = [c for c in df_orig.select(pl.selectors.numeric()).columns if c != target_col]
        train_pd = train_df.to_pandas()
        test_pd = test_df.to_pandas()

        for col in numeric_cols:
            if col in train_pd.columns and col in test_pd.columns:
                stat, p_val = ks_2samp(train_pd[col].dropna(), test_pd[col].dropna())
                report["ks_statistics"][col] = {
                    "D": round(stat, 4),
                    "p_value": round(p_val, 4),
                    "distributions_similar": bool(p_val > 0.05),
                }

        return report

    def _print_report(
        self,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
        target_col: str,
        report: Dict[str, Any],
    ):
        """In báo cáo chia tập."""
        print()
        print("=" * 80)
        print(" 📊 BÁO CÁO CHIA TẬP TRAIN/TEST (STRATIFIED)")
        print("=" * 80)

        # Sizes
        sizes = report["sizes"]
        print(f"\n  Kích thước:")
        print(f"    Tổng:    {sizes['total']:,}")
        print(f"    Train:   {sizes['train']:,} ({sizes['train_pct']}%)")
        print(f"    Test:    {sizes['test']:,} ({sizes['test_pct']}%)")

        # Target distribution
        print(f"\n  Phân phối Target ({target_col}):")
        print(f"  {'Label':>6s} {'Total':>14s} {'Train':>14s} {'Test':>14s}")
        print(f"  {'─'*6} {'─'*14} {'─'*14} {'─'*14}")
        for label in sorted(report["target_distribution"].keys()):
            d = report["target_distribution"][label]
            print(f"  {label:>6s} {d['total']['count']:>6,} ({d['total']['pct']:>5.1f}%) {d['train']['count']:>6,} ({d['train']['pct']:>5.1f}%) {d['test']['count']:>6,} ({d['test']['pct']:>5.1f}%)")

        # Balance check
        if report["stratify_cols_balance"]:
            print(f"\n  Cân bằng biến nhân khẩu học:")
            for col, balance in report["stratify_cols_balance"].items():
                print(f"\n    {col}:")
                print(f"    {'Value':>10s} {'Total %':>9s} {'Train %':>9s} {'Test %':>9s} {'Diff':>8s}")
                print(f"    {'─'*10} {'─'*9} {'─'*9} {'─'*9} {'─'*8}")
                max_diff_overall = 0
                for val, info in balance.items():
                    marker = " ⚠️" if info["max_diff"] > 1 else ""
                    print(f"    {val:>10s} {info['total_pct']:>8.1f}% {info['train_pct']:>8.1f}% {info['test_pct']:>8.1f}% {info['max_diff']:>6.1f}%{marker}")
                    max_diff_overall = max(max_diff_overall, info["max_diff"])
                status = "✅ OK" if max_diff_overall <= 1 else "⚠️  Chênh lệch > 1%"
                print(f"    → {status}")

        # KS test
        if report["ks_statistics"]:
            print(f"\n  Kiểm định KS (phân phối biến số train vs test):")
            print(f"  {'Biến':<40s} {'D':>8s} {'p-value':>10s} {'Kết luận':>14s}")
            print(f"  {'─'*40} {'─'*8} {'─'*10} {'─'*14}")
            for col, ks in report["ks_statistics"].items():
                conclusion = "✅ Giống nhau" if ks["distributions_similar"] else "⚠️  Khác biệt"
                print(f"  {col:<40s} {ks['D']:>8.4f} {ks['p_value']:>10.4f} {conclusion:>14s}")
