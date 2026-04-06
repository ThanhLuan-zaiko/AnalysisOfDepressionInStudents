"""
Visualization Module - Plotly-based
Interactive visualization of depression data

Usage:
    from src.visualization import DepressionVisualizer, ExploratoryAnalyzer

    viz = DepressionVisualizer()
    fig = viz.plot_score_distribution(df)
    fig.show()

    eda = ExploratoryAnalyzer()
    report = eda.run_full_eda(df, output_dir="results/visualizations/")
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Default theme
PLOTLY_THEME = "plotly_white"
COLOR_SCALE = "viridis"
COLOR_SEQUENCE = px.colors.qualitative.Set2


# ==========================================
# 🚫 COLUMNS TO EXCLUDE FROM ANALYSIS
# ==========================================
# These columns are kept in the raw data but excluded from visualizations
# and modeling due to near-zero variance or being identifiers.

EXCLUDED_COLUMNS = [
    "id",                    # Identifier, not a feature
    "Profession",            # 99.9% "Student" - near-constant
    "Work Pressure",         # 99.99% = 0 - near-constant
    "Job Satisfaction",      # 99.98% = 0 - near-constant
]

# Friendly Vietnamese names for columns
COLUMN_LABELS = {
    "id": "ID",
    "Gender": "Giới tính",
    "Age": "Tuổi",
    "City": "Thành phố",
    "Profession": "Nghề nghiệp",
    "Academic Pressure": "Áp lực học tập",
    "Work Pressure": "Áp lực công việc",
    "CGPA": "CGPA",
    "Study Satisfaction": "Hài lòng học tập",
    "Job Satisfaction": "Hài lòng công việc",
    "Sleep Duration": "Thời lượng ngủ",
    "Dietary Habits": "Thói quen ăn uống",
    "Degree": "Bậc học",
    "Have you ever had suicidal thoughts ?": "Ý nghĩ tự tử",
    "Work/Study Hours": "Giờ học/tuần",
    "Financial Stress": "Áp lực tài chính",
    "Family History of Mental Illness": "Tiền sử bệnh tâm lý gia đình",
    "Depression": "Trầm cảm",
}


class ExploratoryAnalyzer:
    """
    Exploratory Data Analysis (EDA) for the Student Depression Dataset.

    This class handles:
    - Data profiling and quality checks
    - Class imbalance analysis
    - Distribution visualizations
    - Missing value reports
    - Low-variance column detection
    - Full EDA report generation

    Ethical note: This is for EXPLORATION only, not diagnosis.
    """

    def __init__(self, theme: str = PLOTLY_THEME):
        self.theme = theme
        self.excluded_cols = EXCLUDED_COLUMNS
        self.column_labels = COLUMN_LABELS

    # ==========================================
    # 📋 DATA PROFILE
    # ==========================================

    def generate_data_profile(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive data profile report.

        Returns dict with:
        - shape, columns, dtypes, missing_values, unique_counts, low_variance_cols
        """
        profile = {
            "timestamp": datetime.now().isoformat(),
            "shape": {"rows": df.height, "cols": df.width},
            "columns": df.columns,
            "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
            "missing_values": {},
            "missing_pct": {},
            "unique_counts": {},
            "unique_pct": {},
            "low_variance_columns": [],
            "class_distribution": {},
        }

        # Missing values
        for col in df.columns:
            null_count = df[col].null_count()
            profile["missing_values"][col] = null_count
            profile["missing_pct"][col] = round(null_count / df.height * 100, 2)

        # Unique counts
        for col in df.columns:
            n_unique = df[col].n_unique()
            profile["unique_counts"][col] = n_unique
            profile["unique_pct"][col] = round(n_unique / df.height * 100, 4)

        # Low variance detection (categorical/binary)
        for col in df.columns:
            if col in self.excluded_cols:
                continue
            n_unique = df[col].n_unique()
            if n_unique <= 10:  # Categorical or ordinal
                value_counts = df[col].value_counts().sort("count", descending=True)
                top_count = value_counts["count"][0]
                top_pct = top_count / df.height * 100
                if top_pct > 95:
                    profile["low_variance_columns"].append({
                        "column": col,
                        "top_value": str(value_counts[col][0]),
                        "top_pct": round(top_pct, 2),
                    })

        # Class distribution (Depression column)
        if "Depression" in df.columns:
            dep_counts = df["Depression"].value_counts().sort("Depression")
            for row in dep_counts.iter_rows(named=True):
                label = "Có trầm cảm (1)" if row["Depression"] == 1 else "Không trầm cảm (0)"
                profile["class_distribution"][label] = {
                    "count": row["count"],
                    "pct": round(row["count"] / df.height * 100, 2),
                }

        return profile

    # ==========================================
    # 📊 EDA VISUALIZATIONS
    # ==========================================

    def plot_class_imbalance(
        self,
        df: pl.DataFrame,
        target_col: str = "Depression",
        title: str = "Phân phối lớp trầm cảm"
    ) -> go.Figure:
        """
        Donut chart showing class distribution with imbalance warning.
        """
        value_counts = df[target_col].value_counts().sort(target_col)

        labels = ["Không trầm cảm (0)", "Có trầm cảm (1)"]
        values = []
        colors = []

        for i, row in enumerate(value_counts.iter_rows(named=True)):
            values.append(row["count"])
            if row[target_col] == 0:
                colors.append("#4CAF50")  # Green
            else:
                colors.append("#F44336")  # Red

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            marker_colors=colors,
            textinfo="label+percent+value",
            hovertemplate="<b>%{label}</b><br>Số lượng: %{value}<br>Tỷ lệ: %{percent}<extra></extra>",
        )])

        # Calculate imbalance ratio
        pct_1 = values[1] / sum(values) * 100 if len(values) > 1 else 0
        pct_0 = values[0] / sum(values) * 100 if len(values) > 0 else 0

        fig.update_layout(
            title={
                "text": f"{title}<br><sup>Tỷ lệ: {pct_1:.1f}% (1) vs {pct_0:.1f}% (0) — Mất cân bằng lớp đáng kể</sup>",
                "x": 0.5,
                "xanchor": "center",
            },
            template=self.theme,
            annotations=[dict(
                text="Lớp", x=0.5, y=0.5,
                font_size=16, showarrow=False
            )],
        )

        return fig

    def plot_feature_distributions(
        self,
        df: pl.DataFrame,
        by_depression: bool = True,
        title: str = "Phân phối các biến theo nhóm trầm cảm"
    ) -> go.Figure:
        """
        Box plots for numeric features grouped by Depression status.
        Excludes low-variance columns.
        """
        numeric_cols = [
            col for col in df.select(pl.selectors.numeric()).columns
            if col not in self.excluded_cols and col != "Depression"
        ]

        n_cols = len(numeric_cols)
        if n_cols == 0:
            fig = go.Figure()
            fig.add_annotation(text="Không có biến số nào để hiển thị", x=0.5, y=0.5, showarrow=False)
            return fig

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[self.column_labels.get(col, col) for col in numeric_cols],
            vertical_spacing=0.1,
            horizontal_spacing=0.08,
        )

        positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

        for idx, col in enumerate(numeric_cols[:6]):  # Max 6 numeric cols
            row, col_pos = positions[idx]
            label = self.column_labels.get(col, col)

            if by_depression and "Depression" in df.columns:
                for dep_val, dep_label, color in [(0, "Không TC", "#4CAF50"), (1, "Có TC", "#F44336")]:
                    fig.add_trace(
                        go.Box(
                            y=df.filter(pl.col("Depression") == dep_val)[col],
                            name=dep_label,
                            marker_color=color,
                            boxmean=True,
                            showlegend=(idx == 0),
                        ),
                        row=row, col=col_pos,
                    )
            else:
                fig.add_trace(
                    go.Box(
                        y=df[col],
                        name=label,
                        boxmean=True,
                    ),
                    row=row, col=col_pos,
                )

        fig.update_layout(
            title={"text": title, "x": 0.5, "xanchor": "center"},
            template=self.theme,
            showlegend=True,
            height=600,
        )

        return fig

    def plot_categorical_distribution(
        self,
        df: pl.DataFrame,
        title: str = "Phân phối các biến phân loại theo nhóm trầm cảm"
    ) -> go.Figure:
        """
        Stacked bar charts for categorical/ordinal features by Depression.
        Excludes low-variance columns.
        """
        categorical_cols = [
            col for col in df.columns
            if col not in self.excluded_cols
            and col != "Depression"
            and df[col].dtype == pl.String
            and df[col].n_unique() <= 15  # Reasonable number of categories
        ]

        n_cols = len(categorical_cols)
        if n_cols == 0:
            fig = go.Figure()
            fig.add_annotation(text="Không có biến phân loại nào để hiển thị", x=0.5, y=0.5, showarrow=False)
            return fig

        # Grid layout
        cols_per_row = 3
        n_rows = (n_cols + cols_per_row - 1) // cols_per_row

        fig = make_subplots(
            rows=n_rows, cols=cols_per_row,
            subplot_titles=[self.column_labels.get(col, col) for col in categorical_cols],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        positions = []
        for r in range(1, n_rows + 1):
            for c in range(1, cols_per_row + 1):
                positions.append((r, c))

        for idx, col in enumerate(categorical_cols):
            row, col_pos = positions[idx]
            label = self.column_labels.get(col, col)

            # Cross-tab: col x Depression
            ct = pd.crosstab(
                df[col].to_pandas(),
                df["Depression"].to_pandas().map({0: "Không TC", 1: "Có TC"}),
                normalize="index",
            ) * 100

            for dep_label in ct.columns:
                fig.add_trace(
                    go.Bar(
                        x=ct.index,
                        y=ct[dep_label],
                        name=f"{dep_label}",
                        marker_color="#4CAF50" if "Không" in dep_label else "#F44336",
                        showlegend=(idx == 0),
                        legendgroup=dep_label,
                    ),
                    row=row, col=col_pos,
                )

        fig.update_layout(
            title={"text": title, "x": 0.5, "xanchor": "center"},
            template=self.theme,
            barmode="stack",
            height=300 * n_rows,
            showlegend=True,
        )

        return fig

    def plot_missing_values(
        self,
        df: pl.DataFrame,
        title: str = "Giá trị thiếu trong dữ liệu"
    ) -> go.Figure:
        """
        Bar chart showing missing value counts per column.
        """
        missing_info = {}
        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                missing_info[col] = null_count

        if not missing_info:
            fig = go.Figure()
            fig.add_annotation(
                text="✅ Không có giá trị thiếu nào trong dataset",
                x=0.5, y=0.5, showarrow=False, font_size=14
            )
            fig.update_layout(template=self.theme, height=200)
            return fig

        fig = go.Figure(data=[go.Bar(
            x=[self.column_labels.get(col, col) for col in missing_info.keys()],
            y=list(missing_info.values()),
            marker_color="#FF9800",
            text=list(missing_info.values()),
            textposition="outside",
        )])

        fig.update_layout(
            title={"text": title, "x": 0.5, "xanchor": "center"},
            template=self.theme,
            xaxis_title="Biến",
            yaxis_title="Số lượng thiếu",
            height=300,
        )

        return fig

    def plot_suicidal_thoughts_analysis(
        self,
        df: pl.DataFrame,
        title: str = "Mối quan hệ: Ý nghĩ tự tử và Trầm cảm"
    ) -> go.Figure:
        """
        Critical analysis: Suicidal thoughts vs Depression.
        This is flagged as a potential label leakage risk.
        """
        col_name = "Have you ever had suicidal thoughts ?"
        if col_name not in df.columns or "Depression" not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="Không đủ dữ liệu để phân tích", x=0.5, y=0.5, showarrow=False)
            return fig

        # Cross-tab
        ct = pd.crosstab(df[col_name].to_pandas(), df["Depression"].to_pandas())
        ct_pct = pd.crosstab(
            df[col_name].to_pandas(), df["Depression"].to_pandas(), normalize="index"
        ) * 100

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Số lượng", "Tỷ lệ (%)"),
            horizontal_spacing=0.1,
        )

        # Count subplot
        for dep_val, dep_label, color in [(0, "Không TC", "#4CAF50"), (1, "Có TC", "#F44336")]:
            fig.add_trace(
                go.Bar(x=ct.index, y=ct[dep_val], name=dep_label, marker_color=color),
                row=1, col=1,
            )

        # Percentage subplot
        for dep_val, dep_label, color in [(0, "Không TC", "#4CAF50"), (1, "Có TC", "#F44336")]:
            fig.add_trace(
                go.Bar(x=ct_pct.index, y=ct_pct[dep_val], name=dep_label, marker_color=color, showlegend=False),
                row=1, col=2,
            )

        fig.update_layout(
            title={
                "text": f"{title}<br><sup>⚠️ Biến này có nguy cơ rò rỉ nhãn — cần được kiểm tra kỹ</sup>",
                "x": 0.5,
                "xanchor": "center",
            },
            template=self.theme,
            showlegend=True,
            height=400,
        )

        return fig

    def plot_correlation_numeric(
        self,
        df: pl.DataFrame,
        title: str = "Tương quan giữa các biến số"
    ) -> go.Figure:
        """
        Correlation heatmap for numeric columns only.
        """
        numeric_cols = [
            col for col in df.select(pl.selectors.numeric()).columns
            if col not in self.excluded_cols and col != "Depression"
        ]

        if len(numeric_cols) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Không đủ biến số để tính tương quan", x=0.5, y=0.5, showarrow=False)
            return fig

        corr_df = df.select(numeric_cols).to_pandas().corr()

        labels = [self.column_labels.get(col, col) for col in corr_df.columns]

        fig = go.Figure(data=[go.Heatmap(
            z=corr_df.values,
            x=labels,
            y=labels,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=np.round(corr_df.values, 2),
            texttemplate="%{text:.2f}",
            colorbar=dict(title="Tương quan"),
        )])

        fig.update_layout(
            title={"text": title, "x": 0.5, "xanchor": "center"},
            template=self.theme,
            height=500,
            width=600,
        )

        return fig

    # ==========================================
    # 📝 FULL EDA REPORT
    # ==========================================

    def run_full_eda(
        self,
        df: pl.DataFrame,
        output_dir: str = "results/visualizations/",
        save_html: bool = True,
        save_report: bool = True,
    ) -> Dict[str, Any]:
        """
        Run complete EDA and generate all visualizations.

        Returns:
            Dict with profile, figure paths, and summary.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {
            "profile": None,
            "figures": {},
            "summary": {},
        }

        print("=" * 80)
        print(" 🔍 EXPLORATORY DATA ANALYSIS (EDA)")
        print("=" * 80)

        # 1. Data Profile
        print("\n📋 Generating data profile...")
        profile = self.generate_data_profile(df)
        results["profile"] = profile

        print(f"   Shape: {profile['shape']['rows']:,} rows × {profile['shape']['cols']} cols")
        print(f"   Target: Depression (0/1)")

        # Print class distribution
        if profile["class_distribution"]:
            print("\n   Class distribution:")
            for label, info in profile["class_distribution"].items():
                print(f"     {label}: {info['count']:,} ({info['pct']}%)")

        # Print low-variance columns
        if profile["low_variance_columns"]:
            print("\n   ⚠️  Low-variance columns (excluded from analysis):")
            for info in profile["low_variance_columns"]:
                print(f"     {info['column']}: {info['top_value']} ({info['top_pct']}%)")

        # Save profile as JSON
        if save_report:
            profile_path = output_path / "eda_data_profile.json"
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
            print(f"\n   ✅ Profile saved: {profile_path}")

        # 2. Visualizations
        print("\n📊 Generating visualizations...")

        # Class imbalance
        print("   → Class imbalance chart...")
        fig_imbalance = self.plot_class_imbalance(df)
        if save_html:
            path = output_path / "eda_class_imbalance.html"
            fig_imbalance.write_html(path)
            results["figures"]["class_imbalance"] = str(path)

        # Missing values
        print("   → Missing values chart...")
        fig_missing = self.plot_missing_values(df)
        if save_html:
            path = output_path / "eda_missing_values.html"
            fig_missing.write_html(path)
            results["figures"]["missing_values"] = str(path)

        # Numeric distributions
        print("   → Numeric feature distributions...")
        fig_numeric = self.plot_feature_distributions(df)
        if save_html:
            path = output_path / "eda_numeric_distributions.html"
            fig_numeric.write_html(path)
            results["figures"]["numeric_distributions"] = str(path)

        # Categorical distributions
        print("   → Categorical feature distributions...")
        fig_cat = self.plot_categorical_distribution(df)
        if save_html:
            path = output_path / "eda_categorical_distributions.html"
            fig_cat.write_html(path)
            results["figures"]["categorical_distributions"] = str(path)

        # Suicidal thoughts analysis
        print("   → Suicidal thoughts analysis...")
        fig_suicide = self.plot_suicidal_thoughts_analysis(df)
        if save_html:
            path = output_path / "eda_suicidal_thoughts.html"
            fig_suicide.write_html(path)
            results["figures"]["suicidal_thoughts"] = str(path)

        # Correlation heatmap
        print("   → Correlation heatmap...")
        fig_corr = self.plot_correlation_numeric(df)
        if save_html:
            path = output_path / "eda_correlation_numeric.html"
            fig_corr.write_html(path)
            results["figures"]["correlation"] = str(path)

        # 3. Summary
        results["summary"] = {
            "total_rows": profile["shape"]["rows"],
            "total_cols": profile["shape"]["cols"],
            "class_1_pct": profile["class_distribution"].get("Có trầm cảm (1)", {}).get("pct", 0),
            "class_0_pct": profile["class_distribution"].get("Không trầm cảm (0)", {}).get("pct", 0),
            "low_variance_cols": [c["column"] for c in profile["low_variance_columns"]],
            "missing_cols": {k: v for k, v in profile["missing_values"].items() if v > 0},
            "figures_generated": len(results["figures"]),
            "excluded_columns": self.excluded_cols,
        }

        print("\n" + "=" * 80)
        print(" ✅ EDA COMPLETE")
        print("=" * 80)
        print(f"\n   📁 {len(results['figures'])} visualizations saved to: {output_path}")
        print(f"   📊 Data profile saved: {output_path / 'eda_data_profile.json'}")

        return results


class DepressionVisualizer:
    """
    Depression data visualization toolkit.
    All charts are interactive (zoom, hover, pan).
    """
    
    def __init__(self, theme: str = PLOTLY_THEME):
        self.theme = theme
        self.color_scale = COLOR_SCALE
        self.color_sequence = COLOR_SEQUENCE
    
    # ==========================================
    # 📊 DISTRIBUTION PLOTS
    # ==========================================
    
    def plot_score_distribution(
        self,
        df: pl.DataFrame,
        score_col: str = "depression_score",
        color_by: Optional[str] = None,
        title: str = "Depression Score Distribution"
    ) -> go.Figure:
        """
        Histogram of depression score distribution

        Args:
            df: DataFrame
            score_col: Column containing depression scores
            color_by: Column to color by (e.g., gender)
            title: Chart title
        """
        df_pandas = df.to_pandas()
        
        fig = px.histogram(
            df_pandas,
            x=score_col,
            color=color_by,
            title=title,
            labels={score_col: "Depression Score", "count": "Count"},
            color_discrete_sequence=self.color_sequence,
            nbins=30,
            barmode="overlay" if color_by else "relative",
            opacity=0.8 if color_by else 1.0
        )
        
        fig.update_layout(
            template=self.theme,
            showlegend=True if color_by else False,
            bargap=0.1
        )
        
        # Add vertical line at threshold
        fig.add_vline(x=16, line_dash="dash", line_color="red", annotation_text="Threshold (16)")
        
        return fig
    
    def plot_box_by_category(
        self,
        df: pl.DataFrame,
        score_col: str = "depression_score",
        category_col: str = "gender",
        title: str = "Depression Score By Group"
    ) -> go.Figure:
        """
        Box plot comparing depression scores between groups
        """
        df_pandas = df.to_pandas()
        
        fig = px.box(
            df_pandas,
            x=category_col,
            y=score_col,
            title=title,
            labels={score_col: "Depression Score", category_col: category_col.title()},
            color=category_col,
            color_discrete_sequence=self.color_sequence,
            points="outliers"
        )
        
        fig.update_layout(template=self.theme)
        
        return fig
    
    def plot_violin_by_category(
        self,
        df: pl.DataFrame,
        score_col: str = "depression_score",
        category_col: str = "gender",
        title: str = "Depression Score Distribution (Violin Plot)"
    ) -> go.Figure:
        """
        Violin plot - combines box plot + density plot
        """
        df_pandas = df.to_pandas()
        
        fig = px.violin(
            df_pandas,
            x=category_col,
            y=score_col,
            title=title,
            labels={score_col: "Depression Score", category_col: category_col.title()},
            color=category_col,
            color_discrete_sequence=self.color_sequence,
            box=True,
            points="all"
        )
        
        fig.update_layout(template=self.theme)
        
        return fig
    
    # ==========================================
    # 📈 CORRELATION & SCATTER PLOTS
    # ==========================================
    
    def plot_correlation_heatmap(
        self,
        df: pl.DataFrame,
        columns: Optional[List[str]] = None,
        title: str = "Correlation Matrix"
    ) -> go.Figure:
        """
        Heatmap of correlation matrix between variables
        """
        if columns is None:
            # Auto-select numeric columns
            columns = df.select(pl.selectors.numeric()).columns
        
        corr_df = df.select(columns).to_pandas().corr()
        
        fig = px.imshow(
            corr_df,
            labels=dict(x="Variable", y="Variable", color="Correlation"),
            title=title,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            text_auto=".2f"
        )
        
        fig.update_layout(template=self.theme)
        
        return fig
    
    def plot_scatter_with_trend(
        self,
        df: pl.DataFrame,
        x_col: str,
        y_col: str = "depression_score",
        color_by: Optional[str] = None,
        trend_line: bool = True,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Scatter plot with trend line

        Usage:
            fig = viz.plot_scatter_with_trend(df, "gpa", "depression_score", color_by="gender")
        """
        df_pandas = df.to_pandas()

        if title is None:
            title = f"Relationship: {x_col} and {y_col}"
        
        fig = px.scatter(
            df_pandas,
            x=x_col,
            y=y_col,
            color=color_by,
            title=title,
            labels={x_col: x_col.title(), y_col: y_col.title()},
            color_discrete_sequence=self.color_sequence,
            opacity=0.7,
            trendline="ols" if trend_line else None
        )
        
        fig.update_layout(template=self.theme)
        
        return fig
    
    # ==========================================
    # 📉 AGGREGATION PLOTS
    # ==========================================
    
    def plot_bar_by_group(
        self,
        df: pl.DataFrame,
        group_col: str,
        value_col: str = "depression_score",
        agg_func: str = "mean",
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Bar chart comparing mean values between groups
        """
        # Aggregate
        agg_expr = getattr(pl.col(value_col), agg_func)()
        agg_df = df.group_by(group_col).agg(agg_expr.alias(value_col))
        agg_df_pandas = agg_df.to_pandas()
        
        if title is None:
            title = f"{agg_func.title()} {value_col.title()} Theo {group_col.title()}"
        
        fig = px.bar(
            agg_df_pandas,
            x=group_col,
            y=value_col,
            title=title,
            labels={group_col: group_col.title(), value_col: value_col.title()},
            color=group_col,
            color_discrete_sequence=self.color_sequence,
            text=value_col
        )
        
        fig.update_layout(template=self.theme)
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        
        return fig
    
    def plot_line_over_time(
        self,
        df: pl.DataFrame,
        time_col: str,
        value_col: str = "depression_score",
        group_col: Optional[str] = None,
        title: str = "Trend Over Time"
    ) -> go.Figure:
        """
        Line chart showing trend over time
        """
        df_pandas = df.to_pandas()
        
        fig = px.line(
            df_pandas,
            x=time_col,
            y=value_col,
            color=group_col,
            title=title,
            labels={time_col: time_col.title(), value_col: value_col.title()},
            color_discrete_sequence=self.color_sequence,
            markers=True
        )
        
        fig.update_layout(template=self.theme)
        
        return fig
    
    # ==========================================
    # 🎯 DEPRESSION CATEGORY PLOTS
    # ==========================================
    
    def plot_category_distribution(
        self,
        df: pl.DataFrame,
        category_col: str = "depression_category",
        title: str = "Depression Category Distribution"
    ) -> go.Figure:
        """
        Pie/Donut chart showing distribution of depression categories
        """
        df_pandas = df.to_pandas()
        category_counts = df_pandas[category_col].value_counts().reset_index()
        category_counts.columns = [category_col, 'count']
        
        # Custom color map cho categories
        color_map = {
            "Không trầm cảm": "green",
            "Nhẹ": "yellow",
            "Trung bình": "orange",
            "Khá nặng": "red",
            "Nặng": "darkred"
        }
        
        fig = px.pie(
            category_counts,
            names=category_col,
            values='count',
            title=title,
            color=category_col,
            color_discrete_map=color_map,
            hole=0.4  # Donut chart
        )
        
        fig.update_layout(template=self.theme)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig
    
    def plot_category_by_group(
        self,
        df: pl.DataFrame,
        category_col: str = "depression_category",
        group_col: str = "gender",
        title: str = "Depression Level By Group"
    ) -> go.Figure:
        """
        Stacked bar chart: category distribution by group
        """
        df_pandas = df.to_pandas()
        cross_tab = pd.crosstab(df_pandas[group_col], df_pandas[category_col])
        
        fig = px.bar(
            cross_tab,
            x=cross_tab.index,
            barmode="stack",
            title=title,
            labels={"x": group_col.title(), "value": "Count"},
            color_discrete_sequence=self.color_sequence
        )
        
        fig.update_layout(template=self.theme)
        
        return fig
    
    # ==========================================
    # 📊 DASHBOARD
    # ==========================================
    
    def create_dashboard(
        self,
        df: pl.DataFrame,
        score_col: str = "depression_score",
        category_col: str = "gender"
    ) -> go.Figure:
        """
        Create comprehensive dashboard with multiple charts

        Usage:
            fig = viz.create_dashboard(df)
            fig.show()
        """
        # Create 2x2 subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Depression Score Distribution",
                "Box Plot By Gender",
                "Variable Correlation",
                "Depression Categories"
            ),
            specs=[[{"type": "histogram"}, {"type": "box"}],
                   [{"type": "heatmap"}, {"type": "pie"}]]
        )
        
        df_pandas = df.to_pandas()
        
        # 1. Histogram
        fig.add_trace(
            go.Histogram(x=df_pandas[score_col], nbinsx=30, name="Distribution"),
            row=1, col=1
        )
        
        # 2. Box plot
        fig.add_trace(
            go.Box(x=df_pandas[category_col], y=df_pandas[score_col], name="Box"),
            row=1, col=2
        )
        
        # 3. Correlation heatmap
        numeric_df = df.select(pl.selectors.numeric())
        if numeric_df.width > 1:
            corr = numeric_df.to_pandas().corr()
            fig.add_trace(
                go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu_r"),
                row=2, col=1
            )
        
        # 4. Pie chart
        if "depression_category" in df.columns:
            cat_counts = df_pandas["depression_category"].value_counts()
            fig.add_trace(
                go.Pie(labels=cat_counts.index, values=cat_counts.values, name="Categories"),
                row=2, col=2
            )
        
        fig.update_layout(
            template=self.theme,
            showlegend=False,
            title_text="📊 DEPRESSION ANALYSIS DASHBOARD",
            height=800
        )
        
        return fig
    
    # ==========================================
    # 💾 EXPORT
    # ==========================================
    
    def save_figure(self, fig: go.Figure, filename: str, format: str = "html"):
        """
        Save figure to file

        Args:
            fig: Plotly figure
            filename: File name (without extension)
            format: "html" (interactive) or "png" (static)
        """
        from pathlib import Path
        
        output_dir = Path("results/visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if format == "html":
            filepath = output_dir / f"{filename}.html"
            fig.write_html(filepath)
        elif format == "png":
            filepath = output_dir / f"{filename}.png"
            fig.write_image(filepath)
        else:
            raise ValueError(f"Format not supported: {format}")
        
        logger.info(f"Saved figure: {filepath}")
        return filepath
