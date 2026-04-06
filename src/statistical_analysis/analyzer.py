"""
Statistical Analysis Module - Pingouin/Statsmodels-based
Comprehensive statistical testing for depression data

Usage:
    from src.statistical_analysis import StatisticalAnalyzer

    analyzer = StatisticalAnalyzer()
    t_test_result = analyzer.t_test(df, "depression_score", "gender")
"""

import polars as pl
import pingouin as pg
import statsmodels.api as sm
from statsmodels.formula.api import ols
from typing import Optional, List, Dict, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Statistical testing toolkit for depression analysis.
    """
    
    def __init__(self):
        logger.info("StatisticalAnalyzer initialized")
    
    # ==========================================
    # 📊 DESCRIPTIVE STATISTICS
    # ==========================================
    
    def descriptive_stats(
        self,
        df: pl.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """
        Descriptive statistics for variables

        Returns:
            DataFrame with: mean, std, median, min, max, skewness, kurtosis
        """
        if columns is None:
            columns = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        
        stats_dict = {
            "column": [],
            "mean": [],
            "std": [],
            "median": [],
            "min": [],
            "max": [],
            "q25": [],
            "q75": [],
            "skewness": [],
            "kurtosis": [],
            "missing": []
        }
        
        for col in columns:
            if col not in df.columns:
                continue
            
            col_data = df[col].drop_nulls()
            
            if len(col_data) == 0:
                continue
            
            # Only process numeric types
            if col_data.dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt64, pl.UInt32]:
                stats_dict["column"].append(col)
                stats_dict["mean"].append(col_data.mean())
                stats_dict["std"].append(col_data.std())
                stats_dict["median"].append(col_data.median())
                stats_dict["min"].append(col_data.min())
                stats_dict["max"].append(col_data.max())
                stats_dict["q25"].append(col_data.quantile(0.25))
                stats_dict["q75"].append(col_data.quantile(0.75))
                stats_dict["skewness"].append(col_data.skew())
                stats_dict["kurtosis"].append(col_data.kurtosis())
                stats_dict["missing"].append(df[col].null_count())
        
        return pl.DataFrame(stats_dict)
    
    # ==========================================
    # 🧪 T-TEST & ANOVA
    # ==========================================
    
    def t_test(
        self,
        df: pl.DataFrame,
        dv: str,
        between: str,
        paired: bool = False
    ) -> pl.DataFrame:
        """
        T-test: Compare depression scores between 2 groups

        Args:
            df: DataFrame
            dv: Dependent variable (e.g., "depression_score")
            between: Independent variable (e.g., "gender")
            paired: Whether it's a paired t-test

        Usage:
            # Compare male/female
            result = analyzer.t_test(df, "depression_score", "gender")
        """
        df_pandas = df.to_pandas()

        # Split into two groups
        groups = df_pandas[between].unique()
        if len(groups) != 2:
            raise ValueError(f"T-test requires exactly 2 groups, got {len(groups)}: {groups}")
        
        group1 = df_pandas[df_pandas[between] == groups[0]][dv]
        group2 = df_pandas[df_pandas[between] == groups[1]][dv]

        result = pg.ttest(group1, group2, paired=paired)
        
        logger.info(f"T-test: {dv} by {between}")
        logger.info(f"Result columns: {result.columns.tolist()}")
        
        # Handle different pingouin column names
        p_col = 'p-val' if 'p-val' in result.columns else 'p'
        t_col = 'T' if 'T' in result.columns else 't'
        
        if t_col in result.columns:
            logger.info(f"T-statistic: {result[t_col].values[0]:.4f}")
        if p_col in result.columns:
            logger.info(f"P-value: {result[p_col].values[0]:.6f}")
        
        return pl.DataFrame(result)
    
    def anova_one_way(
        self,
        df: pl.DataFrame,
        dv: str,
        between: str
    ) -> pl.DataFrame:
        """
        One-way ANOVA: Compare depression scores across multiple groups

        Usage:
            # Compare by education level
            result = analyzer.anova_one_way(df, "depression_score", "education_level")
        """
        df_pandas = df.to_pandas()
        
        result = pg.anova(
            data=df_pandas,
            dv=dv,
            between=between,
            detailed=True
        )
        
        logger.info(f"One-way ANOVA: {dv} by {between}")
        logger.info(f"F-statistic: {result['F'].values[0]:.4f}")
        logger.info(f"P-value: {result['p-unc'].values[0]:.6f}")
        
        return pl.DataFrame(result)
    
    def anova_two_way(
        self,
        df: pl.DataFrame,
        dv: str,
        between: List[str]
    ) -> pl.DataFrame:
        """
        Two-way ANOVA: Analyze impact of 2 independent variables

        Usage:
            result = analyzer.anova_two_way(df, "depression_score", ["gender", "education_level"])
        """
        if len(between) != 2:
            raise ValueError("Two-way ANOVA requires exactly 2 independent variables")
        
        df_pandas = df.to_pandas()
        
        # Use pingouin ANOVA
        formula = f"{dv} ~ C({between[0]}) * C({between[1]})"
        model = ols(formula, data=df_pandas).fit()
        
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        logger.info(f"Two-way ANOVA: {dv} by {between}")
        
        return pl.DataFrame(anova_table)
    
    def post_hoc_test(
        self,
        df: pl.DataFrame,
        dv: str,
        between: str,
        method: str = "tukey"
    ) -> pl.DataFrame:
        """
        Post-hoc test after ANOVA with significant result

        Args:
            method: "tukey", "bonferroni", "holm"
        """
        df_pandas = df.to_pandas()
        
        if method == "tukey":
            result = pg.pairwise_tukey(data=df_pandas, dv=dv, between=between)
        else:
            result = pg.pairwise_tests(data=df_pandas, dv=dv, between=between, padjust=method)
        
        logger.info(f"Post-hoc test ({method}): {dv} by {between}")
        
        return pl.DataFrame(result)
    
    # ==========================================
    # 🔗 CORRELATION ANALYSIS
    # ==========================================
    
    def correlation(
        self,
        df: pl.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "pearson"
    ) -> pl.DataFrame:
        """
        Correlation matrix between variables

        Args:
            method: "pearson", "spearman", "kendall"
        """
        if columns is None:
            columns = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        
        df_pandas = df.select(columns).to_pandas()
        
        corr_matrix = df_pandas.corr(method=method)
        
        logger.info(f"Correlation matrix ({method}): {columns}")
        
        return pl.DataFrame(corr_matrix)
    
    def partial_correlation(
        self,
        df: pl.DataFrame,
        x: str,
        y: str,
        covariates: List[str]
    ) -> Dict:
        """
        Partial correlation: Correlation between x and y controlling for covariates

        Usage:
            result = analyzer.partial_correlation(
                df, "depression_score", "gpa",
                covariates=["age", "gender"]
            )
        """
        df_pandas = df.to_pandas()
        
        result = pg.partial_corr(
            data=df_pandas,
            x=x,
            y=y,
            covar=covariates
        )
        
        logger.info(f"Partial correlation: {x} and {y} (controlling for {covariates})")
        logger.info(f"r = {result['r'].values[0]:.4f}, p = {result['p-val'].values[0]:.6f}")
        
        return result.to_dict()
    
    # ==========================================
    # 📈 REGRESSION ANALYSIS
    # ==========================================
    
    def linear_regression(
        self,
        df: pl.DataFrame,
        dv: str,
        predictors: List[str]
    ) -> Dict:
        """
        Multiple linear regression

        Usage:
            result = analyzer.linear_regression(
                df,
                dv="depression_score",
                predictors=["age", "gpa", "sleep_hours", "stress_level"]
            )
        """
        df_pandas = df.to_pandas()

        # Create formula
        formula = f"{dv} ~ {' + '.join(predictors)}"

        # Fit model
        model = ols(formula, data=df_pandas).fit()
        
        # Summarize results
        summary = {
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "f_statistic": model.fvalue,
            "f_p_value": model.f_pvalue,
            "coefficients": pl.DataFrame({
                "predictor": model.params.index.tolist(),
                "coefficient": model.params.values,
                "std_error": model.bse.values,
                "t_value": model.tvalues.values,
                "p_value": model.pvalues.values
            })
        }
        
        logger.info(f"Linear regression: {dv} ~ {predictors}")
        logger.info(f"R² = {model.rsquared:.4f}, Adj. R² = {model.rsquared_adj:.4f}")
        logger.info(f"F({model.df_model:.0f}, {model.df_resid:.0f}) = {model.fvalue:.2f}, p < 0.001")
        
        return summary
    
    def logistic_regression(
        self,
        df: pl.DataFrame,
        dv: str,
        predictors: List[str]
    ) -> Dict:
        """
        Logistic regression (for binary outcome)

        Usage:
            # Predict depression (score >= 16) or not
            df_with_binary = df.with_columns(
                (pl.col("depression_score") >= 16).alias("depressed")
            )
            result = analyzer.logistic_regression(
                df_with_binary,
                dv="depressed",
                predictors=["age", "gender", "sleep_hours"]
            )
        """
        df_pandas = df.to_pandas()
        
        # Fit logistic regression
        from scipy import stats
        
        X = df_pandas[predictors]
        y = df_pandas[dv]
        
        X = sm.add_constant(X)
        model = sm.Logit(y, X).fit(disp=False)
        
        summary = {
            "log_likelihood": model.llf,
            "pseudo_r_squared": model.prsquared,
            "converged": model.mle_retcode == 1,
            "coefficients": pl.DataFrame({
                "predictor": model.params.index.tolist(),
                "coefficient": model.params.values,
                "std_error": model.bse.values,
                "z_value": model.tvalues.values,
                "p_value": model.pvalues.values,
                "odds_ratio": np.exp(model.params.values)
            })
        }
        
        logger.info(f"Logistic regression: {dv} ~ {predictors}")
        logger.info(f"Pseudo R² = {model.prsquared:.4f}")
        logger.info(f"Converged: {model.mle_retcode == 1}")
        
        return summary
    
    # ==========================================
    # 🔍 FACTOR ANALYSIS
    # ==========================================
    
    def factor_analysis(
        self,
        df: pl.DataFrame,
        columns: List[str],
        n_factors: int = 3,
        rotation: str = "varimax"
    ) -> Dict:
        """
        Exploratory Factor Analysis: Find latent factors

        Usage:
            result = analyzer.factor_analysis(
                df,
                columns=["sleep_hours", "exercise", "social_support",
                        "stress", "anxiety", "loneliness"],
                n_factors=3
            )
        """
        from factor_analyzer import FactorAnalyzer
        
        df_pandas = df.select(columns).to_pandas()
        
        # Fit factor analysis
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
        fa.fit(df_pandas)
        
        # Get results
        loadings = pl.DataFrame(fa.loadings_, columns=[f"Factor{i+1}" for i in range(n_factors)])
        eigenvalues = fa.get_eigenvalues()
        
        summary = {
            "loadings": loadings,
            "eigenvalues": pl.DataFrame({
                "factor": [f"Factor{i+1}" for i in range(n_factors)],
                "eigenvalue": eigenvalues[0][:n_factors].tolist(),
                "variance_proportion": eigenvalues[1][:n_factors].tolist()
            }),
            "n_factors": n_factors,
            "rotation": rotation
        }
        
        logger.info(f"Factor Analysis: {n_factors} factors, {rotation} rotation")
        logger.info(f"Eigenvalues: {eigenvalues[0][:n_factors]}")
        
        return summary
    
    # ==========================================
    # 📊 NORMALITY & ASSUMPTION TESTS
    # ==========================================
    
    def test_normality(
        self,
        df: pl.DataFrame,
        columns: List[str]
    ) -> pl.DataFrame:
        """
        Normality tests (Shapiro-Wilk, Kolmogorov-Smirnov)
        """
        df_pandas = df.to_pandas()
        
        results = []
        
        for col in columns:
            if col not in df.columns:
                continue
            
            data = df_pandas[col].dropna()
            
            if len(data) < 3:
                continue
            
            # Shapiro-Wilk
            try:
                shapiro_result = pg.normality(data, method="shapiro")
                shapiro_stat = shapiro_result.get('W', shapiro_result.get('Statistic', pd.Series([None]))).values[0]
                shapiro_p = shapiro_result.get('p_val', shapiro_result.get('p', pd.Series([None]))).values[0]
            except:
                shapiro_stat, shapiro_p = None, None

            # Kolmogorov-Smirnov
            try:
                ks_result = pg.normality(data, method="kstest")
                ks_stat = ks_result.get('D', ks_result.get('Statistic', pd.Series([None]))).values[0]
                ks_p = ks_result.get('p_val', ks_result.get('p', pd.Series([None]))).values[0]
            except:
                ks_stat, ks_p = None, None
            
            results.append({
                "column": col,
                "shapiro_stat": shapiro_stat,
                "shapiro_p": shapiro_p,
                "ks_stat": ks_stat,
                "ks_p": ks_p,
                "is_normal": bool(shapiro_p) > 0.05 if shapiro_p is not None else False
            })
        
        return pl.DataFrame(results)
    
    def test_homogeneity(
        self,
        df: pl.DataFrame,
        dv: str,
        group: str
    ) -> Dict:
        """
        Homogeneity of variance test (Levene's test)
        """
        df_pandas = df.to_pandas()
        
        groups = df_pandas.groupby(group)[dv].apply(list).values
        
        stat, p_value = pg.homoscedasticity(df_pandas, dv=dv, group=group)
        
        result = {
            "statistic": stat,
            "p_value": p_value,
            "is_homogeneous": p_value > 0.05
        }
        
        logger.info(f"Levene's test: {dv} by {group}")
        logger.info(f"Statistic: {stat:.4f}, p = {p_value:.6f}")
        
        return result
