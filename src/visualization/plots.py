"""
Visualization Module - Plotly-based
Interactive visualization of depression data

Usage:
    from src.visualization import DepressionVisualizer

    viz = DepressionVisualizer()
    fig = viz.plot_score_distribution(df)
    fig.show()
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
import pandas as pd
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

# Default theme
PLOTLY_THEME = "plotly_white"
COLOR_SCALE = "viridis"
COLOR_SEQUENCE = px.colors.qualitative.Set2


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
            columns = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        
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
        numeric_df = df.select(pl.col(pl.NUMERIC_DTYPES))
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
