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
import json
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
        for i in range(n_components):
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

    def plot_variable_contributions(
        self,
        component_idx: int = 0,
        top_n: int = 20,
    ):
        """Bar chart for the variables contributing most to one FAMD component."""
        import plotly.graph_objects as go

        component_name = f"F{component_idx + 1}"
        top_vars = self.results["top_contributions"].get(component_name, [])[:top_n]
        top_vars = list(reversed(top_vars))

        fig = go.Figure(go.Bar(
            x=[row["contribution"] for row in top_vars],
            y=[row["variable"] for row in top_vars],
            orientation="h",
            marker_color=[
                "#2f6f9f" if row["type"] == "numeric" else "#d9822b"
                for row in top_vars
            ],
            customdata=[row["type"] for row in top_vars],
            hovertemplate="<b>%{y}</b><br>Contribution: %{x:.4f}<br>Type: %{customdata}<extra></extra>",
        ))
        variance = self.results["explained_variance_ratio"][component_idx] * 100
        fig.update_layout(
            title={
                "text": f"Top biến đóng góp cho {component_name} ({variance:.1f}% phương sai)",
                "x": 0.5,
                "xanchor": "center",
            },
            template="plotly_white",
            height=max(420, len(top_vars) * 28),
            width=900,
            xaxis_title="Contribution",
            yaxis_title="Variable",
        )
        return fig

    def _component_pairs(self, max_components: Optional[int] = None) -> List[tuple[int, int]]:
        n_available = len(self.results.get("explained_variance_ratio", []))
        n = min(max_components or n_available, n_available)
        return [(idx, idx + 1) for idx in range(max(0, n - 1))]

    def _json_safe_summary(self) -> Dict[str, Any]:
        return {
            "n_components": int(self.results["n_components"]),
            "n_samples": int(self.results["n_samples"]),
            "n_numeric": int(self.results["n_numeric"]),
            "n_categorical": int(self.results["n_categorical"]),
            "numeric_cols": list(self.results["numeric_cols"]),
            "categorical_cols": list(self.results["categorical_cols"]),
            "eigenvalues": [float(value) for value in self.results["eigenvalues"]],
            "explained_variance_ratio": [
                float(value) for value in self.results["explained_variance_ratio"]
            ],
            "cumulative_variance": [
                float(value) for value in self.results["cumulative_variance"]
            ],
            "top_contributions": self.results["top_contributions"],
        }

    def save_result_tables(self, output_dir: str = "results/visualizations/") -> Dict[str, str]:
        """Save FAMD coordinates and a compact JSON summary for downstream reports."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved: Dict[str, str] = {}
        coords_path = output_path / "famd_coordinates.csv"
        self.results["coordinates"].to_csv(coords_path, index=False)
        saved["coordinates"] = str(coords_path)

        summary_path = output_path / "famd_summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(self._json_safe_summary(), handle, indent=2, ensure_ascii=False)
        saved["summary"] = str(summary_path)
        return saved

    # ==========================================
    # SAVE
    # ==========================================

    def save_all_plots(
        self,
        output_dir: str = "results/visualizations/",
        save_html: bool = True,
        max_components: Optional[int] = None,
    ) -> Dict[str, str]:
        """Save all FAMD plots, including adjacent component pairs."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved: Dict[str, str] = {}

        fig1 = self.plot_variance_explained()
        if save_html:
            p = output_path / "famd_variance_explained.html"
            fig1.write_html(p)
            saved["variance"] = str(p)

        n_components = min(
            max_components or int(self.results["n_components"]),
            len(self.results["explained_variance_ratio"]),
        )
        for component_idx in range(n_components):
            fig = self.plot_variable_contributions(component_idx=component_idx)
            if save_html:
                component_name = f"F{component_idx + 1}"
                p = output_path / f"famd_contributions_{component_name}.html"
                fig.write_html(p)
                saved[f"contributions_{component_name}"] = str(p)

        for pc_x, pc_y in self._component_pairs(max_components=n_components):
            pair_name = f"F{pc_x + 1}_F{pc_y + 1}"
            title_suffix = f"F{pc_x + 1} - F{pc_y + 1}"

            fig_corr = self.plot_correlation_circle(
                dims=(pc_x, pc_y),
                title=f"Tương quan biến - {title_suffix}",
            )
            fig_proj = self.plot_sample_projection(
                dims=(pc_x, pc_y),
                title=f"Phân bố mẫu FAMD - {title_suffix}",
            )

            if save_html:
                corr_path = output_path / f"famd_correlation_circle_{pair_name}.html"
                proj_path = output_path / f"famd_sample_projection_{pair_name}.html"
                fig_corr.write_html(corr_path)
                fig_proj.write_html(proj_path)
                saved[f"correlation_circle_{pair_name}"] = str(corr_path)
                saved[f"sample_projection_{pair_name}"] = str(proj_path)

                if (pc_x, pc_y) == (0, 1):
                    legacy_corr = output_path / "famd_correlation_circle.html"
                    legacy_proj = output_path / "famd_sample_projection.html"
                    fig_corr.write_html(legacy_corr)
                    fig_proj.write_html(legacy_proj)
                    saved["correlation_circle"] = str(legacy_corr)
                    saved["sample_projection"] = str(legacy_proj)
                elif (pc_x, pc_y) == (1, 2):
                    legacy_corr = output_path / "famd_corr_circle_F2_F3.html"
                    legacy_proj = output_path / "famd_sample_F2_F3.html"
                    fig_corr.write_html(legacy_corr)
                    fig_proj.write_html(legacy_proj)
                    saved["correlation_circle_F2_F3"] = str(legacy_corr)
                    saved["sample_projection_F2_F3"] = str(legacy_proj)

        saved.update(self.save_result_tables(output_dir=str(output_path)))

        return saved

    # ==========================================
    # CLUSTERING
    # ==========================================

    def _famd_matrix(self, n_dims: Optional[int] = None) -> tuple[pd.DataFrame, List[str]]:
        coords = self.results["coordinates"].copy()
        component_cols = [col for col in coords.columns if col.startswith("F")]
        component_cols = component_cols[: n_dims or len(component_cols)]
        if not component_cols:
            raise ValueError("No FAMD coordinates available. Run run_famd() first.")
        return coords, component_cols

    def _component_interpretation(self, component_name: str) -> str:
        top_vars = self.results.get("top_contributions", {}).get(component_name, [])[:3]
        if not top_vars:
            return component_name
        names = ", ".join(row["variable"] for row in top_vars)
        return f"{component_name}: liên quan mạnh đến {names}"

    def _cluster_profiles(
        self,
        labels: np.ndarray,
        method: str,
        component_cols: List[str],
        target_col: str,
    ) -> List[Dict[str, Any]]:
        coords = self.results["coordinates"].copy()
        coords["cluster"] = labels
        profiles: List[Dict[str, Any]] = []

        for label in sorted(pd.unique(labels)):
            mask = coords["cluster"] == label
            subset = coords.loc[mask]
            component_means = {
                col: float(subset[col].mean())
                for col in component_cols
            }
            dominant_components = sorted(
                component_means.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:2]

            if int(label) == -1:
                interpretation = "Nhóm nhiễu/outlier: DBSCAN không gán các mẫu này vào cụm mật độ rõ."
            else:
                parts = []
                for component_name, mean_value in dominant_components:
                    direction = "cao" if mean_value >= 0 else "thấp"
                    parts.append(
                        f"{component_name} {direction} ({mean_value:.2f}) - "
                        f"{self._component_interpretation(component_name)}"
                    )
                interpretation = "; ".join(parts)

            profile = {
                "method": method,
                "cluster": int(label),
                "n_samples": int(mask.sum()),
                "share_pct": float(mask.mean() * 100),
                "component_means": component_means,
                "interpretation_vi": interpretation,
            }
            if target_col in coords.columns:
                target_values = pd.to_numeric(subset[target_col], errors="coerce")
                profile["depression_rate"] = float(target_values.mean())
            profiles.append(profile)

        return profiles

    def run_clustering(
        self,
        n_dims: Optional[int] = None,
        target_col: str = "Depression",
        k_range: tuple[int, int] = (2, 10),
    ) -> Dict[str, Any]:
        """Cluster FAMD coordinates with K-Means and DBSCAN."""
        from sklearn.cluster import DBSCAN, KMeans
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        coords, component_cols = self._famd_matrix(n_dims=n_dims)
        X = coords[component_cols].to_numpy(dtype=float)
        n_samples = X.shape[0]
        if n_samples < 4:
            raise ValueError("Need at least 4 samples for FAMD clustering.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        k_min, k_max = k_range
        k_max = min(k_max, n_samples - 1)
        kmeans_candidates: List[Dict[str, Any]] = []
        best_kmeans: Dict[str, Any] | None = None
        for k in range(k_min, k_max + 1):
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
            labels = model.fit_predict(X_scaled)
            silhouette = silhouette_score(X_scaled, labels)
            candidate = {
                "k": int(k),
                "silhouette": float(silhouette),
                "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
                "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
                "inertia": float(model.inertia_),
                "labels": labels,
                "centers": model.cluster_centers_.tolist(),
            }
            kmeans_candidates.append(candidate)
            if best_kmeans is None or candidate["silhouette"] > best_kmeans["silhouette"]:
                best_kmeans = candidate

        min_samples = max(5, 2 * X_scaled.shape[1])
        min_samples = min(min_samples, max(2, n_samples - 1))
        neighbor_count = min(min_samples, n_samples)
        distances, _ = NearestNeighbors(n_neighbors=neighbor_count).fit(X_scaled).kneighbors(X_scaled)
        kth_distances = np.sort(distances[:, -1])
        quantiles = np.linspace(0.50, 0.95, 10)

        dbscan_candidates: List[Dict[str, Any]] = []
        best_dbscan: Dict[str, Any] | None = None
        for quantile in quantiles:
            eps = float(np.quantile(kth_distances, quantile))
            if eps <= 0:
                continue
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
            non_noise = labels != -1
            cluster_labels = sorted(set(labels[non_noise]))
            n_clusters = len(cluster_labels)
            noise_fraction = float((labels == -1).mean())
            silhouette = None
            score = None
            if 2 <= n_clusters < int(non_noise.sum()):
                silhouette = float(silhouette_score(X_scaled[non_noise], labels[non_noise]))
                score = silhouette - noise_fraction * 0.25

            candidate = {
                "eps": eps,
                "eps_quantile": float(quantile),
                "min_samples": int(min_samples),
                "n_clusters": int(n_clusters),
                "noise_fraction": noise_fraction,
                "silhouette_non_noise": silhouette,
                "score": score,
                "labels": labels,
            }
            dbscan_candidates.append(candidate)
            if score is not None and (best_dbscan is None or score > best_dbscan["score"]):
                best_dbscan = candidate

        if best_kmeans is None:
            raise ValueError("K-Means could not be fitted on FAMD coordinates.")

        kmeans_labels = np.asarray(best_kmeans["labels"], dtype=int)
        dbscan_labels = (
            np.asarray(best_dbscan["labels"], dtype=int)
            if best_dbscan is not None
            else np.full(n_samples, -1, dtype=int)
        )

        result = {
            "n_samples": int(n_samples),
            "n_dims": int(len(component_cols)),
            "component_cols": component_cols,
            "target_col": target_col,
            "kmeans": {
                "best_k": best_kmeans["k"],
                "silhouette": best_kmeans["silhouette"],
                "calinski_harabasz": best_kmeans["calinski_harabasz"],
                "davies_bouldin": best_kmeans["davies_bouldin"],
                "inertia": best_kmeans["inertia"],
                "labels": kmeans_labels.tolist(),
                "profiles": self._cluster_profiles(kmeans_labels, "kmeans", component_cols, target_col),
                "candidates": [
                    {key: value for key, value in candidate.items() if key != "labels"}
                    for candidate in kmeans_candidates
                ],
            },
            "dbscan": {
                "found_valid_clusters": best_dbscan is not None,
                "eps": best_dbscan["eps"] if best_dbscan is not None else None,
                "eps_quantile": best_dbscan["eps_quantile"] if best_dbscan is not None else None,
                "min_samples": int(min_samples),
                "n_clusters": int(best_dbscan["n_clusters"]) if best_dbscan is not None else 0,
                "noise_fraction": float(best_dbscan["noise_fraction"]) if best_dbscan is not None else 1.0,
                "silhouette_non_noise": (
                    best_dbscan["silhouette_non_noise"] if best_dbscan is not None else None
                ),
                "labels": dbscan_labels.tolist(),
                "profiles": self._cluster_profiles(dbscan_labels, "dbscan", component_cols, target_col),
                "candidates": [
                    {key: value for key, value in candidate.items() if key != "labels"}
                    for candidate in dbscan_candidates
                ],
            },
        }
        self.results["clustering"] = result
        return result

    def plot_cluster_projection(
        self,
        labels: List[int],
        method_name: str,
        dims: tuple[int, int] = (0, 1),
        target_col: str = "Depression",
    ):
        """Scatter FAMD projection colored by cluster labels."""
        import plotly.express as px

        coords = self.results["coordinates"].copy()
        pc_x, pc_y = dims
        x_col = f"F{pc_x + 1}"
        y_col = f"F{pc_y + 1}"
        coords["cluster"] = [str(label) for label in labels]
        if len(coords) > 5000:
            coords = coords.sample(5000, random_state=self.random_state)
        if target_col in coords.columns:
            coords[target_col] = coords[target_col].map({0: "Không TC", 1: "Có TC"}).fillna(coords[target_col])

        hover_data = [target_col] if target_col in coords.columns else None
        fig = px.scatter(
            coords,
            x=x_col,
            y=y_col,
            color="cluster",
            hover_data=hover_data,
            opacity=0.5,
            title=f"{method_name}: cụm trên mặt phẳng {x_col}-{y_col}",
            labels={
                x_col: f"{x_col} ({self.results['explained_variance_ratio'][pc_x] * 100:.1f}%)",
                y_col: f"{y_col} ({self.results['explained_variance_ratio'][pc_y] * 100:.1f}%)",
                "cluster": "Cụm",
            },
        )
        fig.update_layout(template="plotly_white", height=560, title={"x": 0.5, "xanchor": "center"})
        return fig

    def generate_clustering_report_html(
        self,
        clustering: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a Vietnamese HTML report explaining K-Means and DBSCAN clusters."""
        clustering = clustering or self.results.get("clustering")
        if clustering is None:
            raise ValueError("No clustering result. Call run_clustering() first.")

        def rows_html(profiles: List[Dict[str, Any]]) -> str:
            rows = []
            for profile in profiles:
                dep_rate = profile.get("depression_rate")
                dep_text = f"{dep_rate * 100:.1f}%" if dep_rate is not None else "N/A"
                rows.append(
                    "<tr>"
                    f"<td>{profile['cluster']}</td>"
                    f"<td>{profile['n_samples']:,}</td>"
                    f"<td>{profile['share_pct']:.1f}%</td>"
                    f"<td>{dep_text}</td>"
                    f"<td>{profile['interpretation_vi']}</td>"
                    "</tr>"
                )
            return "\n".join(rows)

        dbscan = clustering["dbscan"]
        dbscan_summary = (
            f"DBSCAN tìm được {dbscan['n_clusters']} cụm với "
            f"{dbscan['noise_fraction'] * 100:.1f}% điểm nhiễu."
            if dbscan["found_valid_clusters"]
            else "DBSCAN không tìm được cấu trúc cụm mật độ rõ; phần lớn mẫu được xem là nhiễu hoặc gom không ổn định."
        )

        return f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Báo cáo FAMD Clustering</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1120px; margin: 0 auto; padding: 24px; color: #1f2933; line-height: 1.55; }}
    h1, h2 {{ color: #102a43; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 10px; text-align: left; vertical-align: top; }}
    th {{ background: #e6f6ff; color: #102a43; }}
    .note {{ background: #f0f4f8; border-left: 4px solid #2f6f9f; padding: 12px 14px; }}
  </style>
</head>
<body>
  <h1>Báo cáo phân cụm trên không gian FAMD</h1>
  <p class="note">Clustering được fit trên {clustering['n_dims']} tọa độ FAMD đầu tiên ({', '.join(clustering['component_cols'])}); biến Depression chỉ dùng để diễn giải sau khi phân cụm.</p>

  <h2>K-Means</h2>
  <p>K-Means chọn <strong>k={clustering['kmeans']['best_k']}</strong> theo silhouette cao nhất. Silhouette={clustering['kmeans']['silhouette']:.3f}, Calinski-Harabasz={clustering['kmeans']['calinski_harabasz']:.1f}, Davies-Bouldin={clustering['kmeans']['davies_bouldin']:.3f}.</p>
  <table>
    <thead><tr><th>Cụm</th><th>Số mẫu</th><th>Tỷ trọng</th><th>Tỷ lệ Depression</th><th>Diễn giải</th></tr></thead>
    <tbody>{rows_html(clustering['kmeans']['profiles'])}</tbody>
  </table>

  <h2>DBSCAN</h2>
  <p>{dbscan_summary}</p>
  <table>
    <thead><tr><th>Cụm</th><th>Số mẫu</th><th>Tỷ trọng</th><th>Tỷ lệ Depression</th><th>Diễn giải</th></tr></thead>
    <tbody>{rows_html(clustering['dbscan']['profiles'])}</tbody>
  </table>
</body>
</html>"""

    def save_clustering_outputs(
        self,
        output_dir: str = "results/visualizations/",
        n_dims: Optional[int] = None,
        target_col: str = "Depression",
    ) -> Dict[str, str]:
        """Run K-Means/DBSCAN and save JSON, plots, and the Vietnamese HTML report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        clustering = self.run_clustering(n_dims=n_dims, target_col=target_col)
        saved: Dict[str, str] = {}

        json_path = output_path / "famd_clustering_results.json"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(clustering, handle, indent=2, ensure_ascii=False)
        saved["clustering_json"] = str(json_path)

        kmeans_plot = self.plot_cluster_projection(
            labels=clustering["kmeans"]["labels"],
            method_name="K-Means",
            target_col=target_col,
        )
        kmeans_path = output_path / "famd_clusters_kmeans.html"
        kmeans_plot.write_html(kmeans_path)
        saved["kmeans_plot"] = str(kmeans_path)

        dbscan_plot = self.plot_cluster_projection(
            labels=clustering["dbscan"]["labels"],
            method_name="DBSCAN",
            target_col=target_col,
        )
        dbscan_path = output_path / "famd_clusters_dbscan.html"
        dbscan_plot.write_html(dbscan_path)
        saved["dbscan_plot"] = str(dbscan_path)

        report_path = output_path / "famd_clustering_report.html"
        report_path.write_text(
            self.generate_clustering_report_html(clustering),
            encoding="utf-8",
        )
        saved["clustering_report"] = str(report_path)
        return saved
