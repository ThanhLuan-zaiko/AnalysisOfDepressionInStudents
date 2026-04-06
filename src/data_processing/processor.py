"""
Data Processing Module - Polars-based
Fast depression data processing with Polars (Rust backend)

Usage:
    from src.data_processing import DataProcessor

    processor = DataProcessor()
    df = processor.load_csv("data/raw/depression.csv")
    cleaned = processor.clean_data(df)
    filtered = processor.filter_at_risk(cleaned, threshold=16)
"""

import polars as pl
from pathlib import Path
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Depression data processing toolkit with Polars.
    Focused on performance and memory efficiency.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataProcessor initialized: {self.data_dir}")
    
    # ==========================================
    # 📥 LOAD DATA
    # ==========================================
    
    def load_csv(self, filename: str, lazy: bool = True) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Load CSV file with Polars (lazy by default for memory optimization)

        Args:
            filename: CSV file name
            lazy: If True, returns LazyFrame (doesn't load into RAM)

        Returns:
            LazyFrame or DataFrame
        """
        filepath = self.raw_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File does not exist: {filepath}")
        
        if lazy:
            logger.info(f"Loading CSV (lazy): {filepath}")
            return pl.scan_csv(filepath)
        else:
            logger.info(f"Loading CSV (eager): {filepath}")
            return pl.read_csv(filepath)
    
    def load_parquet(self, filename: str, lazy: bool = True) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Load Parquet file (faster than CSV, smaller file size)
        """
        filepath = self.processed_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File does not exist: {filepath}")
        
        if lazy:
            logger.info(f"Loading Parquet (lazy): {filepath}")
            return pl.scan_parquet(filepath)
        else:
            logger.info(f"Loading Parquet (eager): {filepath}")
            return pl.read_parquet(filepath)
    
    # ==========================================
    # 🧹 CLEANING
    # ==========================================
    
    def clean_data(self, df: Union[pl.DataFrame, pl.LazyFrame]) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Automatic data cleaning:
        - Remove rows with missing values in critical columns
        - Fill missing values in numeric columns with median
        - Standardize column names
        """
        logger.info("Cleaning data...")

        # Standardize column names (lowercase, snake_case)
        df = df.rename({
            col: col.strip().lower().replace(" ", "_").replace("-", "_")
            for col in df.columns
        })
        
        # Count missing values before cleaning
        null_counts = df.select(pl.all().null_count())
        logger.info(f"Missing values before cleaning:\n{null_counts}")
        
        # Fill numeric columns with median
        numeric_cols = df.select(pl.selectors.numeric()).columns
        
        if numeric_cols:
            for col in numeric_cols:
                df = df.with_columns(
                    pl.col(col).fill_null(pl.col(col).median())
                )
        
        # Drop rows with missing values in critical columns
        critical_cols = ["depression_score", "age", "gender"]
        existing_critical = [col for col in critical_cols if col in df.columns]
        
        if existing_critical:
            df = df.drop_nulls(subset=existing_critical)
        
        logger.info(f"Data cleaned. Shape: {df.collect_schema() if isinstance(df, pl.LazyFrame) else df.shape}")
        return df
    
    # ==========================================
    # 🔍 FILTERING
    # ==========================================
    
    def filter_at_risk(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        threshold: int = 16,
        column: str = "depression_score"
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Filter at-risk students

        Args:
            df: DataFrame
            threshold: Score cutoff (default 16 = moderate risk)
            column: Column containing depression scores

        Returns:
            Filtered DataFrame
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in DataFrame")
        
        logger.info(f"Filtering at-risk students (threshold >= {threshold})")
        return df.filter(pl.col(column) >= threshold)
    
    def filter_by_demographics(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        gender: Optional[str] = None,
        age_min: Optional[int] = None,
        age_max: Optional[int] = None,
        education_level: Optional[str] = None
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Filter by demographics (gender, age, education)

        Usage:
            # Filter females aged 18-25
            filtered = processor.filter_by_demographics(df, gender="Female", age_min=18, age_max=25)
        """
        if gender:
            df = df.filter(pl.col("gender").str.to_lowercase() == gender.lower())
        
        if age_min is not None:
            df = df.filter(pl.col("age") >= age_min)
        
        if age_max is not None:
            df = df.filter(pl.col("age") <= age_max)
        
        if education_level:
            df = df.filter(pl.col("education_level").str.to_lowercase() == education_level.lower())
        
        return df
    
    # ==========================================
    # 📊 AGGREGATION
    # ==========================================
    
    def aggregate_by_group(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        group_col: str,
        metrics: Optional[dict] = None
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Aggregate data by group

        Args:
            df: DataFrame
            group_col: Column to group by
            metrics: Dict of custom metrics (default: mean, std, count)

        Usage:
            # Default aggregation
            result = processor.aggregate_by_group(df, "gender")

            # Custom metrics
            result = processor.aggregate_by_group(df, "gender", metrics={
                "avg_score": pl.col("depression_score").mean(),
                "max_score": pl.col("depression_score").max()
            })
        """
        if metrics is None:
            metrics = {
                "mean": "depression_score",
                "std": "depression_score",
                "median": "depression_score",
                "count": "depression_score",
                "min": "depression_score",
                "max": "depression_score"
            }
        
        agg_exprs = []
        for func_name, col_name in metrics.items():
            if hasattr(pl.col(col_name), func_name):
                agg_exprs.append(getattr(pl.col(col_name), func_name)().alias(f"{col_name}_{func_name}"))
        
        return df.group_by(group_col).agg(agg_exprs)
    
    def calculate_depression_categories(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        score_col: str = "depression_score"
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Categorize depression by score:
        - 0-4: No depression
        - 5-9: Mild
        - 10-14: Moderate
        - 15-19: Moderately severe
        - 20+: Severe

        Adds column "depression_category"
        """
        if score_col not in df.columns:
            raise ValueError(f"Column '{score_col}' does not exist")
        
        df = df.with_columns(
            pl.when(pl.col(score_col) <= 4)
            .then(pl.lit("Không trầm cảm"))
            .when(pl.col(score_col) <= 9)
            .then(pl.lit("Nhẹ"))
            .when(pl.col(score_col) <= 14)
            .then(pl.lit("Trung bình"))
            .when(pl.col(score_col) <= 19)
            .then(pl.lit("Khá nặng"))
            .otherwise(pl.lit("Nặng"))
            .alias("depression_category")
        )
        
        return df
    
    # ==========================================
    # 💾 SAVE DATA
    # ==========================================
    
    def save_parquet(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        filename: str
    ) -> Path:
        """
        Save DataFrame as Parquet (5-10x smaller than CSV, faster reads)
        """
        filepath = self.processed_dir / filename
        
        if isinstance(df, pl.LazyFrame):
            df.collect().write_parquet(filepath)
        else:
            df.write_parquet(filepath)
        
        logger.info(f"Saved to: {filepath}")
        return filepath
    
    def save_csv(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        filename: str
    ) -> Path:
        """
        Save DataFrame as CSV
        """
        filepath = self.processed_dir / filename
        
        if isinstance(df, pl.LazyFrame):
            df.collect().write_csv(filepath)
        else:
            df.write_csv(filepath)
        
        logger.info(f"Saved to: {filepath}")
        return filepath
    
    # ==========================================
    # 📈 PROFILING
    # ==========================================
    
    def profile_data(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame]
    ) -> pl.DataFrame:
        """
        Quick statistics about dataset (similar to pandas describe but more detailed)
        """
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        
        profile = df.describe()
        
        print("=" * 80)
        print(" 📊 DATA PROFILE")
        print("=" * 80)
        print(f"Rows: {df.height:,}")
        print(f"Columns: {df.width}")
        print(f"\nColumns: {df.columns}")
        print(f"\nStatistics:")
        print(profile)
        print("=" * 80)
        
        return profile
    
    # ==========================================
    # ⚡ LAPIPELINE (Lazy Evaluation Chain)
    # ==========================================
    
    def create_pipeline(self, filename: str) -> pl.LazyFrame:
        """
        Create processing pipeline with lazy evaluation

        Usage:
            pipeline = processor.create_pipeline("depression.csv")
            result = (
                pipeline
                .pipe(processor.clean_data)
                .pipe(processor.calculate_depression_categories)
                .filter(pl.col("depression_category") == "Severe")
                .collect()
            )
        """
        return self.load_csv(filename, lazy=True)


# ==========================================
# 🛠️ Helper functions
# ==========================================

def load_sample_data(n_rows: int = 1000) -> pl.DataFrame:
    """
    Generate sample data for testing (no CSV file needed)

    Usage:
        df = load_sample_data(1000)
    """
    import numpy as np
    
    np.random.seed(42)
    
    return pl.DataFrame({
        "student_id": range(1, n_rows + 1),
        "age": np.random.randint(15, 30, n_rows),
        "gender": np.random.choice(["Nam", "Nữ"], n_rows),
        "education_level": np.random.choice(["High School", "University", "Graduate"], n_rows),
        "depression_score": np.random.normal(12, 5, n_rows).clip(0, 27).astype(int),
        "gpa": np.random.uniform(2.0, 4.0, n_rows).round(2),
        "sleep_hours": np.random.normal(6.5, 1.5, n_rows).clip(3, 12).round(1),
        "exercise_hours_per_week": np.random.exponential(2, n_rows).clip(0, 20).round(1),
        "social_support_score": np.random.normal(50, 15, n_rows).clip(0, 100).round(1),
        "stress_level": np.random.randint(1, 11, n_rows),
    })
