"""
Imbalanced Data Handling Module - SMOTE-based
Handle imbalanced data in depression classification

Usage:
    from src.ml_models import ImbalancedDataHandler

    handler = ImbalancedDataHandler()
    X_resampled, y_resampled = handler.apply_smote(X_train, y_train)
"""

import numpy as np
import polars as pl
from typing import Dict, Optional, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class ImbalancedDataHandler:
    """
    Handle imbalanced dataset - common problem in depression data
    (often fewer severe depression cases than normal)
    """
    
    def __init__(self):
        self.original_distribution = {}
        self.resampled_distribution = {}
        logger.info("ImbalancedDataHandler initialized")
    
    # ==========================================
    # 📊 ANALYSIS
    # ==========================================
    
    def analyze_imbalance(
        self,
        y: np.ndarray
    ) -> Dict:
        """
        Analyze imbalance level

        Returns:
            Dict with distribution, imbalance ratio
        """
        counter = Counter(y)
        
        # Statistics
        n_samples = len(y)
        n_classes = len(counter)
        class_counts = dict(counter)
        class_ratios = {k: v / n_samples for k, v in counter.items()}
        
        # Imbalance ratio
        max_count = max(counter.values())
        min_count = min(counter.values())
        imbalance_ratio = max_count / min_count
        
        # Store original
        self.original_distribution = {
            'counts': class_counts,
            'ratios': class_ratios,
            'imbalance_ratio': imbalance_ratio
        }
        
        logger.info(f"Class Distribution:")
        for cls, count in sorted(class_counts.items()):
            ratio = class_ratios[cls] * 100
            logger.info(f"  Class {cls}: {count:,} ({ratio:.1f}%)")
        logger.info(f"Imbalance Ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 10:
            logger.warning("⚠️  Severe imbalance detected!")
        elif imbalance_ratio > 3:
            logger.warning("⚠️  Moderate imbalance")
        else:
            logger.info("✅ Balance is acceptable")
        
        return self.original_distribution
    
    def visualize_distribution(
        self,
        y: np.ndarray,
        title: str = "Class Distribution"
    ):
        """
        Plot bar chart of class distribution
        """
        import matplotlib.pyplot as plt
        
        counter = Counter(y)
        classes = sorted(counter.keys())
        counts = [counter[c] for c in classes]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(classes, counts, color='skyblue', edgecolor='navy')
        
        # Add labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   f'{count:,}', ha='center', va='bottom', fontsize=12)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    # ==========================================
    # 🔧 RESAMPLING METHODS
    # ==========================================
    
    def apply_smote(
        self,
        X: np.ndarray,
        y: np.ndarray,
        strategy: str = 'auto',
        k_neighbors: int = 5,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique)

        Args:
            X: Features
            y: Labels
            strategy: 'auto' (balance), 'minority', or dict {class: count}
            k_neighbors: Number of neighbors for SMOTE
            random_state: Seed

        Returns:
            X_resampled, y_resampled
        """
        try:
            from imblearn.over_sampling import SMOTE
            
            if strategy == 'auto':
                smote = SMOTE(
                    k_neighbors=k_neighbors,
                    random_state=random_state
                )
            elif strategy == 'minority':
                smote = SMOTE(
                    k_neighbors=k_neighbors,
                    random_state=random_state,
                    sampling_strategy='minority'
                )
            elif isinstance(strategy, dict):
                smote = SMOTE(
                    k_neighbors=k_neighbors,
                    random_state=random_state,
                    sampling_strategy=strategy
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Log results
            original_count = len(y)
            new_count = len(y_resampled)
            added_count = new_count - original_count
            
            logger.info(f"SMOTE applied:")
            logger.info(f"  Original samples: {original_count:,}")
            logger.info(f"  Resampled samples: {new_count:,}")
            logger.info(f"  Added {added_count:,} synthetic samples ({added_count/original_count*100:.1f}%)")
            
            # Store distribution
            self.resampled_distribution = dict(Counter(y_resampled))
            
            return X_resampled, y_resampled
            
        except ImportError:
            logger.error("imbalanced-learn not installed. Run: uv add imbalanced-learn")
            raise
    
    def apply_smote_tomek(
        self,
        X: np.ndarray,
        y: np.ndarray,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        SMOTE + Tomek Links (oversampling + cleaning)

        SMOTE creates new samples, Tomek links removes noisy samples
        """
        try:
            from imblearn.combine import SMOTETomek
            
            smote_tomek = SMOTETomek(
                random_state=random_state
            )
            
            X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
            
            logger.info(f"SMOTE+Tomek applied:")
            logger.info(f"  Original: {len(y):,} → Resampled: {len(y_resampled):,}")
            
            self.resampled_distribution = dict(Counter(y_resampled))
            
            return X_resampled, y_resampled
            
        except ImportError:
            logger.error("imbalanced-learn not installed.")
            raise
    
    def apply_smote_enn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        SMOTE + ENN (Edited Nearest Neighbors)

        More aggressive than SMOTE+Tomek in removing noise
        """
        try:
            from imblearn.combine import SMOTEENN
            
            smote_enn = SMOTEENN(
                random_state=random_state
            )
            
            X_resampled, y_resampled = smote_enn.fit_resample(X, y)
            
            logger.info(f"SMOTE+ENN applied:")
            logger.info(f"  Original: {len(y):,} → Resampled: {len(y_resampled):,}")
            
            self.resampled_distribution = dict(Counter(y_resampled))
            
            return X_resampled, y_resampled
            
        except ImportError:
            logger.error("imbalanced-learn not installed.")
            raise
    
    # ==========================================
    # ⚖️ UNDERSAMPLING
    # ==========================================
    
    def apply_random_undersampling(
        self,
        X: np.ndarray,
        y: np.ndarray,
        strategy: str = 'auto',
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random undersampling of majority class

        Note: Loses data, only use when dataset is large
        """
        try:
            from imblearn.under_sampling import RandomUnderSampler
            
            rus = RandomUnderSampler(
                sampling_strategy=strategy,
                random_state=random_state
            )
            
            X_resampled, y_resampled = rus.fit_resample(X, y)
            
            logger.info(f"Random Undersampling applied:")
            logger.info(f"  Original: {len(y):,} → Resampled: {len(y_resampled):,}")
            
            self.resampled_distribution = dict(Counter(y_resampled))
            
            return X_resampled, y_resampled
            
        except ImportError:
            logger.error("imbalanced-learn not installed.")
            raise
    
    # ==========================================
    # 🎯 COMPARISON
    # ==========================================
    
    def compare_methods(
        self,
        X: np.ndarray,
        y: np.ndarray,
        methods: Optional[list] = None
    ) -> pl.DataFrame:
        """
        Compare resampling methods

        Returns:
            DataFrame comparison
        """
        if methods is None:
            methods = ['original', 'smote', 'smote_tomek', 'smote_enn']
        
        results = []
        
        for method in methods:
            if method == 'original':
                X_res, y_res = X, y
            elif method == 'smote':
                X_res, y_res = self.apply_smote(X, y)
            elif method == 'smote_tomek':
                X_res, y_res = self.apply_smote_tomek(X, y)
            elif method == 'smote_enn':
                X_res, y_res = self.apply_smote_enn(X, y)
            else:
                continue
            
            counter = Counter(y_res)
            results.append({
                'method': method,
                'n_samples': len(y_res),
                'n_features': X_res.shape[1],
                'class_distribution': str(dict(counter)),
                'imbalance_ratio': max(counter.values()) / min(counter.values()) if min(counter.values()) > 0 else float('inf')
            })
        
        df_comparison = pl.DataFrame(results)
        
        logger.info("Resampling Methods Comparison:")
        print(df_comparison)
        
        return df_comparison
    
    # ==========================================
    # 🔄 PIPELINE INTEGRATION
    # ==========================================
    
    def prepare_balanced_data(
        self,
        df: pl.DataFrame,
        feature_cols: list,
        target_col: str = "depression_score",
        threshold: int = 16,
        method: str = "smote",
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple:
        """
        Complete pipeline: prepare data + apply resampling

        Returns:
            X_train_res, X_test, y_train_res, y_test
        """
        from sklearn.model_selection import train_test_split

        # Extract features and target
        X = df.select(feature_cols).to_numpy()

        # Binary classification
        if df[target_col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            y = (df[target_col].to_numpy() >= threshold).astype(int)
        else:
            y = df[target_col].to_numpy().astype(int)

        # Train/test split BEFORE resampling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        # Apply resampling only on TRAIN set
        if method == "smote":
            X_train_res, y_train_res = self.apply_smote(X_train, y_train, random_state=random_state)
        elif method == "smote_tomek":
            X_train_res, y_train_res = self.apply_smote_tomek(X_train, y_train, random_state=random_state)
        elif method == "smote_enn":
            X_train_res, y_train_res = self.apply_smote_enn(X_train, y_train, random_state=random_state)
        else:
            X_train_res, y_train_res = X_train, y_train
        
        logger.info(f"Data prepared with {method} resampling:")
        logger.info(f"  Train: {len(y_train_res):,} samples")
        logger.info(f"  Test: {len(y_test):,} samples (unseen)")
        
        return X_train_res, X_test, y_train_res, y_test
    
    # ==========================================
    # 📊 REPORT
    # ==========================================
    
    def generate_report(
        self,
        y_original: np.ndarray,
        y_resampled: np.ndarray
    ) -> str:
        """
        Generate report about resampling
        """
        original_counter = Counter(y_original)
        resampled_counter = Counter(y_resampled)
        
        report = []
        report.append("=" * 80)
        report.append(" 📊 IMBALANCED DATA HANDLING REPORT")
        report.append("=" * 80)
        report.append("")
        
        report.append("📋 ORIGINAL DISTRIBUTION:")
        for cls, count in sorted(original_counter.items()):
            ratio = count / len(y_original) * 100
            report.append(f"  Class {cls}: {count:,} ({ratio:.1f}%)")
        report.append(f"  Total: {len(y_original):,}")
        report.append("")
        
        report.append("🔄 RESAMPLED DISTRIBUTION:")
        for cls, count in sorted(resampled_counter.items()):
            ratio = count / len(y_resampled) * 100
            report.append(f"  Class {cls}: {count:,} ({ratio:.1f}%)")
        report.append(f"  Total: {len(y_resampled):,}")
        report.append("")
        
        samples_added = len(y_resampled) - len(y_original)
        report.append(f"📈 SAMPLES ADDED: {samples_added:,} ({samples_added/len(y_original)*100:.1f}%)")
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        print(report_text)
        
        return report_text
