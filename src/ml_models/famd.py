"""
FAMD (Factor Analysis of Mixed Data) — Giảm chiều dữ liệu hỗn hợp

Phân tích đồng thời biến số (numeric) và biến phân loại (categorical).
Tương tự PCA nhưng xử lý được cả 2 loại biến.

Usage:
    from src.ml_models.famd import FAMDAnalyzer

    analyzer = FAMDAnalyzer()
    results = analyzer.run_famd(df, n_components=10)
"""

import polars as pl
import pandas as pd
import numpy as np
import prince
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class FAMDAnalyzer:
    """
    FAMD analysis for mixed-type depression data.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.famd = None
        self.results = {}

    def run_famd(
        self,
        df: pl.DataFrame,
        n_components: int = 10,
        excluded_cols: Optional[List[str]] = None,
        target_col: str = "Depression",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Chạy FAMD trên dataset hỗn hợp.

        Args:
            df: DataFrame gốc (chưa drop cột)
            n_components: Số thành phần giữ lại
            excluded_cols: Cột loại khỏi phân tích (id, biến hằng số...)
            target_col: Cột target — giữ lại để tô màu biểu đồ

        Returns:
            Dict chứa: eigenvalues, coordinates, contributions, correlations,
                       cumulative_variance, plots
        """
        if excluded_cols is None:
            excluded_cols = []

        # Chuẩn bị dữ liệu
        df_work = df.drop([c for c in excluded_cols if c in df.columns])

        if target_col not in df_work.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        # Tách numeric và categorical
        numeric_cols = [
            c for c in df_work.select(pl.selectors.numeric()).columns
            if c != target_col
        ]
        cat_cols = [
            c for c in df_work.select(pl.col(pl.String)).columns
            if c != target_col
        ]

        if verbose:
            print()
            print("=" * 80)
            print(" 🔬 FAMD — PHÂN TÍCH THÀNH PHẦN DỮ LIỆU HỖN HỢP")
            print("=" * 80)
            print(f"\n  Numeric:  {len(numeric_cols)} biến")
            print(f"  Categorical: {len(cat_cols)} biến")
            print(f"  Tổng features: {len(numeric_cols) + len(cat_cols)}")
            print(f"  Samples: {df_work.height:,}")
            print(f"  Target: {target_col}")

        # Chuyển sang pandas cho prince
        df_pd = df_work.to_pandas()

        # Impute missing values — FAMD không xử lý NaN
        for col in numeric_cols:
            if df_pd[col].isna().any():
                median_val = df_pd[col].median()
                df_pd[col] = df_pd[col].fillna(median_val)

        for col in cat_cols:
            if df_pd[col].isna().any():
                mode_val = df_pd[col].mode()
                if len(mode_val) > 0:
                    df_pd[col] = df_pd[col].fillna(mode_val[0])

        # Chuyển categorical thành category dtype
        for col in cat_cols:
            df_pd[col] = df_pd[col].astype("category")

        # Chạy FAMD
        if verbose:
            print(f"\n  ⏳ Đang chạy FAMD với {n_components} thành phần...")

        self.famd = prince.FAMD(n_components=n_components, random_state=self.random_state)
        self.famd = self.famd.fit(df_pd[numeric_cols + cat_cols])

        # ==========================================
        # Thu thập kết quả
        # ==========================================

        # 1. Eigenvalues (phương sai giải thích)
        eigenvalues = self.famd.eigenvalues_
        total_variance = sum(eigenvalues)
        explained_var_ratio = [ev / total_variance for ev in eigenvalues]
        cumulative_var = np.cumsum(explained_var_ratio)

        if verbose:
            print(f"\n  📊 PHƯƠNG SAI GIẢI THÍCH:")
            print(f"  {'PC':>4s} {'Eigenvalue':>12s} {'Variance %':>12s} {'Cumulative %':>14s}")
            print(f"  {'─'*4} {'─'*12} {'─'*12} {'─'*14}")
            for i, (ev, vr, cv) in enumerate(zip(eigenvalues, explained_var_ratio, cumulative_var), 1):
                print(f"  {i:>4d} {ev:>12.2f} {vr*100:>11.1f}% {cv*100:>13.1f}%")

        # 2. Coordinates (tọa độ các sample trên các PC)
        coordinates = self.famd.transform(df_pd[numeric_cols + cat_cols])
        coordinates.columns = [f"F{i+1}" for i in range(coordinates.shape[1])]
        coordinates[target_col] = df_work[target_col].to_pandas().values

        # 3. Variable correlations với các PC
        # Correlations giữa biến gốc và các thành phần
        correlations = {}

        # Numeric: correlation Pearson giữa biến gốc và PC scores
        for col in numeric_cols:
            corrs = []
            for i in range(n_components):
                pc_scores = coordinates[f"F{i+1}"]
                r = pc_scores.corr(df_pd[col])
                corrs.append(r)
            correlations[col] = {
                "type": "numeric",
                "correlations": corrs,
                "cos2": [r**2 for r in corrs],
                "contrib": [(r**2) / explained_var_ratio[i] if explained_var_ratio[i] > 0 else 0
                           for i, r in enumerate(corrs)],
            }

        # Categorical: eta² (correlation ratio) giữa biến category và PC scores
        for col in cat_cols:
            cos2_vals = []
            contrib_vals = []
            for i in range(n_components):
                pc_scores = coordinates[f"F{i+1}"]
                # Eta²: tỷ lệ phương sai của PC được giải thích bởi biến category
                # Tính qua ANOVA: eta² = SS_between / SS_total
                grand_mean = pc_scores.mean()
                ss_total = ((pc_scores - grand_mean) ** 2).sum()
                ss_between = 0
                for cat in df_pd[col].cat.categories:
                    mask = df_pd[col] == cat
                    if mask.sum() == 0:
                        continue
                    group_mean = pc_scores[mask].mean()
                    ss_between += mask.sum() * (group_mean - grand_mean) ** 2
                eta_sq = ss_between / ss_total if ss_total > 0 else 0
                cos2_vals.append(eta_sq)
                contrib_vals.append(eta_sq / explained_var_ratio[i] if explained_var_ratio[i] > 0 else 0)

            correlations[col] = {
                "type": "categorical",
                "correlations": [np.sqrt(v) for v in cos2_vals],  # sqrt(eta²) ≈ |r|
                "cos2": cos2_vals,
                "contrib": contrib_vals,
            }

        # 4. Top contributing variables per PC
        top_contributions = {}
        for i in range(min(5, n_components)):
            contribs = []
            for var_name, var_data in correlations.items():
                contribs.append({
                    "variable": var_name,
                    "type": var_data["type"],
                    "contribution": var_data["contrib"][i],
                    "cos2": var_data["cos2"][i],
                })
            contribs.sort(key=lambda x: x["contribution"], reverse=True)
            top_contributions[f"F{i+1}"] = contribs[:10]

        # ==========================================
        # Lưu kết quả
        # ==========================================
        self.results = {
            "n_components": n_components,
            "n_samples": df_work.height,
            "n_numeric": len(numeric_cols),
            "n_categorical": len(cat_cols),
            "eigenvalues": eigenvalues,
            "explained_variance_ratio": explained_var_ratio,
            "cumulative_variance": cumulative_var.tolist(),
            "coordinates": coordinates,
            "correlations": correlations,
            "top_contributions": top_contributions,
            "numeric_cols": numeric_cols,
            "categorical_cols": cat_cols,
        }

        # ==========================================
        # In báo cáo
        # ==========================================
        if verbose:
            self._print_report()

        return self.results

    def _print_report(self):
        """In báo cáo FAMD chi tiết."""
        print(f"\n  🏆 TOP BIẾN ĐÓNG GÓI NHIỀU NHẤT:")

        for pc_name, top_vars in self.results["top_contributions"].items():
            print(f"\n  {pc_name} (giải thích {self.results['explained_variance_ratio'][int(pc_name[1:])-1]*100:.1f}% phương sai):")
            for j, var in enumerate(top_vars[:5], 1):
                icon = "🔢" if var["type"] == "numeric" else "🏷️"
                print(f"    {j}. {icon} {var['variable']:<35s} contrib={var['contribution']:.3f}  cos²={var['cos2']:.3f}")

    # ==========================================
    # VISUALIZATION
    # ==========================================

    def plot_variance_explained(
        self,
        max_components: int = 10,
        title: str = "Phương sai giải thích bởi các thành phần FAMD"
    ):
        """Scree plot — eigenvalues và cumulative variance."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        n = min(max_components, len(self.results["eigenvalues"]))
        ev = self.results["eigenvalues"][:n]
        vr = [v * 100 for v in self.results["explained_variance_ratio"][:n]]
        cv = [v * 100 for v in self.results["cumulative_variance"][:n]]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Bar: variance %
        fig.add_trace(go.Bar(
            x=[f"F{i+1}" for i in range(n)],
            y=vr,
            name="Phương sai (%)",
            marker_color="#4CAF50",
            text=[f"{v:.1f}%" for v in vr],
            textposition="outside",
        ), secondary_y=False)

        # Line: cumulative
        fig.add_trace(go.Scatter(
            x=[f"F{i+1}" for i in range(n)],
            y=cv,
            name="Tích lũy (%)",
            mode="lines+markers",
            line=dict(color="#F44336", width=3),
            marker=dict(size=8),
            text=[f"{v:.1f}%" for v in cv],
            textposition="top center",
        ), secondary_y=True)

        fig.update_layout(
            title={"text": title, "x": 0.5, "xanchor": "center"},
            template="plotly_white",
            height=400,
            showlegend=False,
        )
        fig.update_yaxes(title_text="Phương sai (%)", secondary_y=False)
        fig.update_yaxes(title_text="Tích lũy (%)", secondary_y=True, range=[0, 105])

        return fig

    def plot_correlation_circle(
        self,
        dims: tuple = (0, 1),
        title: str = "Tương quan biến — Thành phần FAMD"
    ):
        """
        Correlation circle — chỉ cho numeric variables.
        dims: (pc_x, pc_y) — index 0-based của 2 PC cần vẽ
        """
        import plotly.graph_objects as go

        pc_x, pc_y = dims
        numeric_vars = [v for v, d in self.results["correlations"].items() if d["type"] == "numeric"]

        fig = go.Figure()

        # Axes
        fig.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(color="gray", width=1, dash="dash"))
        fig.add_shape(type="line", x0=0, y0=-1, x1=0, y1=1, line=dict(color="gray", width=1, dash="dash"))
        fig.add_shape(type="circle", xref="x", yref="y", x0=-1, y0=-1, x1=1, y1=1,
                      line=dict(color="gray", width=1, dash="dot"))

        for var in numeric_vars:
            data = self.results["correlations"][var]
            x = data["correlations"][pc_x]
            y = data["correlations"][pc_y]
            cos2 = data["cos2"][pc_x] + data["cos2"][pc_y]

            color = "#1f77b4" if cos2 > 0.5 else "#ff7f0e" if cos2 > 0.2 else "#d3d3d3"

            fig.add_trace(go.Scatter(
                x=[0, x], y=[0, y],
                mode="lines+text",
                line=dict(color=color, width=2),
                text=[None, var],
                textposition="top right",
                textfont=dict(size=9, color=color),
                showlegend=False,
            ))

        fig.update_layout(
            title={"text": title, "x": 0.5, "xanchor": "center"},
            template="plotly_white",
            width=600, height=600,
            xaxis=dict(scaleanchor="y", scaleratio=1, range=[-1.1, 1.1]),
            yaxis=dict(range=[-1.1, 1.1]),
            showlegend=False,
        )

        return fig

    def plot_sample_projection(
        self,
        dims: tuple = (0, 1),
        target_col: str = "Depression",
        title: str = "Phân bố mẫu trên mặt phẳng FAMD"
    ):
        """
        Scatter plot — chiếu các sample lên 2 PC, tô màu theo target.
        Dùng sample nếu quá lớn (plotly giới hạn ~100k points mượt).
        """
        import plotly.express as px

        coords = self.results["coordinates"].copy()
        pc_x, pc_y = dims
        x_col = f"F{pc_x+1}"
        y_col = f"F{pc_y+1}"

        # Sample nếu > 5000
        if len(coords) > 5000:
            coords = coords.sample(5000, random_state=self.random_state)

        coords[target_col] = coords[target_col].map({0: "Không TC", 1: "Có TC"})

        fig = px.scatter(
            coords,
            x=x_col, y=y_col,
            color=target_col,
            color_discrete_map={"Không TC": "#4CAF50", "Có TC": "#F44336"},
            opacity=0.4,
            size_max=3,
            title=title,
            labels={x_col: f"{x_col} ({self.results['explained_variance_ratio'][pc_x]*100:.1f}%)",
                    y_col: f"{y_col} ({self.results['explained_variance_ratio'][pc_y]*100:.1f}%)"},
        )

        fig.update_layout(
            template="plotly_white",
            height=500,
            title={"x": 0.5, "xanchor": "center"},
        )

        return fig

    # ==========================================
    # SAVE
    # ==========================================

    def save_all_plots(
        self,
        output_dir: str = "results/visualizations/",
        save_html: bool = True,
    ) -> Dict[str, str]:
        """Lưu tất cả biểu đồ FAMD."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved = {}

        # 1. Scree plot
        fig1 = self.plot_variance_explained()
        if save_html:
            p = output_path / "famd_variance_explained.html"
            fig1.write_html(p)
            saved["variance"] = str(p)

        # 2. Correlation circle
        fig2 = self.plot_correlation_circle()
        if save_html:
            p = output_path / "famd_correlation_circle.html"
            fig2.write_html(p)
            saved["correlation_circle"] = str(p)

        # 3. Sample projection
        fig3 = self.plot_sample_projection()
        if save_html:
            p = output_path / "famd_sample_projection.html"
            fig3.write_html(p)
            saved["sample_projection"] = str(p)

        # Additional dimensions
        fig4 = self.plot_correlation_circle(dims=(1, 2))
        if save_html:
            p = output_path / "famd_corr_circle_F2_F3.html"
            fig4.write_html(p)
            saved["correlation_circle_F2_F3"] = str(p)

        fig5 = self.plot_sample_projection(dims=(1, 2))
        if save_html:
            p = output_path / "famd_sample_F2_F3.html"
            fig5.write_html(p)
            saved["sample_projection_F2_F3"] = str(p)

        return saved
