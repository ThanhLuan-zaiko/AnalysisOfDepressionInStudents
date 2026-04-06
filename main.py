"""
Main Pipeline - Depression Analysis in Students
Integrates all modules into a complete workflow

Usage:
    uv run python main.py
    uv run python main.py --sample  # Run with sample data
"""

import sys
import argparse
import logging
from pathlib import Path
import polars as pl
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Setup logging
from src.utils import setup_logging, Timer, print_device_info
logger = setup_logging(level="INFO", log_file="logs/analysis.log")


def main(use_sample: bool = True):
    """
    Main analysis pipeline
    
    Args:
        use_sample: If True, use sample data (no CSV file needed)
    """
    print("=" * 80)
    print(" 🧠 DEPRESSION ANALYSIS IN STUDENTS")
    print("=" * 80)
    print()
    
    # ==========================================
    # 1. DEVICE INFO
    # ==========================================
    print_device_info()
    print()
    
    # ==========================================
    # 2. DATA LOADING & PROCESSING
    # ==========================================
    with Timer("Data Loading & Processing"):
        from src.data_processing import DataProcessor, load_sample_data
        
        processor = DataProcessor()
        
        if use_sample:
            logger.info("Using sample data")
            df = load_sample_data(n_rows=10000)
            print(f"✅ Sample data created: {df.height:,} rows, {df.width} columns")
        else:
            # Load from CSV file
            try:
                df = processor.load_csv("depression_data.csv", lazy=False)
                print(f"✅ Data loaded from CSV: {df.height:,} rows")
            except FileNotFoundError:
                logger.error("CSV file not found. Use --sample flag or add data file.")
                print("❌ File not found. Use: python main.py --sample")
                return
        
        # Data cleaning
        print("\n🧹 Cleaning data...")
        df_clean = processor.clean_data(df)
        
        # Calculate depression categories
        print("📊 Calculating depression categories...")
        df_clean = processor.calculate_depression_categories(df_clean)
        
        # Data profile
        print("\n" + "=" * 80)
        print(" 📊 DATA PROFILE")
        print("=" * 80)
        print(f"Total rows: {df_clean.height:,}")
        print(f"Total columns: {df_clean.width}")
        print(f"\nColumns: {df_clean.columns}")
        
        # Show category distribution
        if "depression_category" in df_clean.columns:
            category_counts = df_clean.group_by("depression_category").agg(
                pl.count().alias("count")
            ).sort("count", descending=True)
            
            print("\nDepression Category Distribution:")
            print(category_counts)
    
    # ==========================================
    # 3. VISUALIZATION
    # ==========================================
    with Timer("Visualization"):
        from src.visualization import DepressionVisualizer
        
        viz = DepressionVisualizer()
        
        print("\n" + "=" * 80)
        print(" 📈 VISUALIZATION")
        print("=" * 80)
        
        # Distribution plot
        print("Creating score distribution plot...")
        fig_dist = viz.plot_score_distribution(df_clean)
        viz.save_figure(fig_dist, "score_distribution", format="html")
        
        # Box plot by gender
        if "gender" in df_clean.columns:
            print("Creating box plot by gender...")
            fig_box = viz.plot_box_by_category(df_clean, category_col="gender")
            viz.save_figure(fig_box, "score_by_gender", format="html")
        
        # Category distribution
        if "depression_category" in df_clean.columns:
            print("Creating category distribution plot...")
            fig_cat = viz.plot_category_distribution(df_clean)
            viz.save_figure(fig_cat, "category_distribution", format="html")
        
        # Correlation heatmap
        print("Creating correlation heatmap...")
        fig_corr = viz.plot_correlation_heatmap(df_clean)
        viz.save_figure(fig_corr, "correlation_matrix", format="html")
        
        # Dashboard
        print("Creating comprehensive dashboard...")
        fig_dashboard = viz.create_dashboard(df_clean)
        viz.save_figure(fig_dashboard, "dashboard", format="html")
        
        print("✅ Visualizations saved to: results/visualizations/")
    
    # ==========================================
    # 4. STATISTICAL ANALYSIS
    # ==========================================
    with Timer("Statistical Analysis"):
        from src.statistical_analysis import StatisticalAnalyzer
        
        analyzer = StatisticalAnalyzer()
        
        print("\n" + "=" * 80)
        print(" 📊 STATISTICAL ANALYSIS")
        print("=" * 80)
        
        # Descriptive statistics
        print("\n📋 Descriptive Statistics:")
        desc_stats = analyzer.descriptive_stats(df_clean)
        print(desc_stats)
        
        # T-test (gender difference)
        if "gender" in df_clean.columns and df_clean["gender"].n_unique() == 2:
            print("\n🔍 T-test: Depression Score by Gender")
            t_test_result = analyzer.t_test(df_clean, "depression_score", "gender")
            print(t_test_result)
        
        # Correlation analysis
        print("\n🔗 Correlation Analysis:")
        corr_matrix = analyzer.correlation(df_clean)
        print(corr_matrix.head())
        
        # Normality test
        numeric_cols = df_clean.select(pl.col(pl.NUMERIC_DTYPES)).columns[:5]
        print(f"\n📏 Normality Test (first 5 numeric columns):")
        normality = analyzer.test_normality(df_clean, numeric_cols)
        print(normality)
    
    # ==========================================
    # 5. MACHINE LEARNING
    # ==========================================
    with Timer("Machine Learning"):
        from src.ml_models import DepressionPredictor, HyperparameterOptimizer, SHAPExplainer, ImbalancedDataHandler
        from src.evaluation import ModelEvaluator

        predictor = DepressionPredictor(use_gpu=True)
        optimizer = HyperparameterOptimizer()
        explainer = SHAPExplainer()
        imbalance_handler = ImbalancedDataHandler()
        evaluator = ModelEvaluator()

        print("\n" + "=" * 80)
        print(" 🤖 MACHINE LEARNING")
        print("=" * 80)

        # Prepare data
        feature_cols = [col for col in df_clean.columns
                       if col not in ["depression_score", "depression_category"]]

        # Check and handle imbalanced data
        print("\n📊 Analyzing class distribution...")
        X_temp = df_clean.select(feature_cols).to_numpy()
        y_temp = (df_clean["depression_score"].to_numpy() >= 16).astype(int)
        imbalance_handler.analyze_imbalance(y_temp)

        # Apply SMOTE if needed
        print("\n⚖️  Handling imbalanced data with SMOTE...")
        from sklearn.model_selection import train_test_split
        
        X_train_raw, X_test, y_train_raw, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.2,
            random_state=42,
            stratify=y_temp
        )
        
        X_train, y_train = imbalance_handler.apply_smote(X_train_raw, y_train_raw)
        
        print(f"  Train set (after SMOTE): {len(y_train):,} samples")
        print(f"  Test set (unseen): {len(y_test):,} samples")

        # Train XGBoost with default params
        print("\n🌲 Training XGBoost (default)...")
        xgb_model_default = predictor.train_xgboost(X_train, y_train, use_gpu=True)
        xgb_metrics_default = predictor.evaluate(xgb_model_default, X_test, y_test)
        print(f"  Accuracy: {xgb_metrics_default['accuracy']:.4f}")
        print(f"  F1: {xgb_metrics_default['f1']:.4f}")

        # Train LightGBM with default params
        print("\n💡 Training LightGBM (default)...")
        lgb_model_default = predictor.train_lightgbm(X_train, y_train, use_gpu=True)
        lgb_metrics_default = predictor.evaluate(lgb_model_default, X_test, y_test)
        print(f"  Accuracy: {lgb_metrics_default['accuracy']:.4f}")
        print(f"  F1: {lgb_metrics_default['f1']:.4f}")

        # Optuna Optimization (quick run with 30 trials)
        print("\n🎯 Hyperparameter Optimization with Optuna...")
        print("  (Running 30 trials to balance performance and time)")
        
        try:
            xgb_opt_result = optimizer.optimize_xgboost(X_train, y_train, n_trials=30, timeout=600)
            print(f"\n  ✅ Best XGBoost F1: {xgb_opt_result['best_score']:.4f}")
            print(f"  Best params: {xgb_opt_result['best_params']}")
            
            # Train model with best params
            xgb_model = xgb.XGBClassifier(**xgb_opt_result['best_params'], random_state=42)
            xgb_model.fit(X_train, y_train)
            xgb_metrics = predictor.evaluate(xgb_model, X_test, y_test)
            print(f"  Optimized - Accuracy: {xgb_metrics['accuracy']:.4f}, F1: {xgb_metrics['f1']:.4f}")
        except Exception as e:
            logger.warning(f"Optuna optimization failed: {e}. Using default model.")
            xgb_model = xgb_model_default
            xgb_metrics = xgb_metrics_default

        # SHAP Explainability
        print("\n🔍 SHAP Explainability Analysis...")
        shap_explanation = explainer.explain_tree_model(
            xgb_model, X_train, X_test, feature_names=feature_cols
        )
        
        # Global interpretation
        global_interp = explainer.global_interpretation(
            shap_explanation['shap_values'],
            X_test,
            feature_cols
        )
        
        print("\n✅ Top 5 Most Important Features:")
        print(global_interp.head(5))

        # Compare models
        print("\n🏆 Model Comparison:")
        comparison = evaluator.compare_models({
            "XGBoost (default)": xgb_metrics_default,
            "LightGBM (default)": lgb_metrics_default,
            "XGBoost (optimized)": xgb_metrics
        })

        # Save models
        print("\n💾 Saving models...")
        predictor.save_model(xgb_model_default, "xgboost_default.pkl")
        predictor.save_model(lgb_model_default, "lightgbm_default.pkl")
        predictor.save_model(xgb_model, "xgboost_optimized.pkl")
        print("✅ Models saved to: models/")
    
    # ==========================================
    # 6. DEEP LEARNING
    # ==========================================
    with Timer("Deep Learning"):
        from src.deep_learning import DepressionNN
        
        nn_trainer = DepressionNN()
        
        print("\n" + "=" * 80)
        print(" 🔥 DEEP LEARNING")
        print("=" * 80)
        
        # Prepare data
        train_loader, test_loader = nn_trainer.prepare_dataloaders(
            df_clean,
            feature_cols=feature_cols,
            target_col="depression_score",
            threshold=16,
            batch_size=64
        )
        
        # Create model
        input_dim = len(feature_cols)
        model = nn_trainer.create_model(input_dim, hidden_dim=128, model_type="basic")
        
        # Train
        print("\n🏋️ Training Neural Network...")
        history = nn_trainer.train(
            model,
            train_loader,
            val_loader=test_loader,
            epochs=50,
            learning_rate=0.001,
            early_stopping_patience=10
        )
        
        # Evaluate
        nn_metrics = nn_trainer.evaluate(model, test_loader)
        print(f"  Accuracy: {nn_metrics['accuracy']:.4f}")
        print(f"  F1: {nn_metrics['f1']:.4f}")
        print(f"  ROC AUC: {nn_metrics['roc_auc']:.4f}")
        
        # Save model
        print("\n💾 Saving neural network model...")
        nn_trainer.save_model(model, "neural_network_depression.pth")
        print("✅ Neural network saved to: models/")
    
    # ==========================================
    # 7. FINAL COMPARISON
    # ==========================================
    print("\n" + "=" * 80)
    print(" 🏆 FINAL MODEL COMPARISON")
    print("=" * 80)

    all_metrics = {
        "XGBoost (default)": xgb_metrics_default,
        "LightGBM (default)": lgb_metrics_default,
        "XGBoost (optimized)": xgb_metrics,
        "Neural Network": nn_metrics
    }

    comparison = evaluator.compare_models(all_metrics)
    print("\n")
    print(comparison)

    # Rank models
    ranked = evaluator.rank_models(all_metrics, primary_metric='f1')
    print(f"\n🥇 Best Model: {ranked[0]}")
    print(f"📈 Improvement from optimization: {(xgb_metrics['f1'] - xgb_metrics_default['f1'])*100:.2f}%")
    
    # ==========================================
    # 8. SUMMARY
    # ==========================================
    print("\n" + "=" * 80)
    print(" ✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print("\n📁 Results saved:")
    print("  • Visualizations: results/visualizations/")
    print("  • Models: models/")
    print("  • Logs: logs/analysis.log")
    print("\n📊 Summary:")
    print(f"  • Dataset: {df_clean.height:,} samples, {df_clean.width} features")
    print(f"  • SMOTE: Applied (handled imbalanced data)")
    print(f"  • Optuna: Hyperparameter optimization completed")
    print(f"  • SHAP: Explainability analysis completed")
    print(f"  • Best Model: {ranked[0]}")
    print(f"  • Best F1 Score: {all_metrics[ranked[0]]['f1']:.4f}")
    print("\n🔑 Key Insights:")
    print(f"  • Top features: {', '.join(global_interp['feature'].head(3).to_list())}")
    print(f"  • Optimization improved F1 by: {(xgb_metrics['f1'] - xgb_metrics_default['f1'])*100:.2f}%")
    print("\n" + "=" * 80)
    print(" 🎉 Thank you for using the analysis tool!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depression Analysis Pipeline")
    parser.add_argument("--sample", action="store_true", help="Run with sample data")
    parser.add_argument("--no-sample", action="store_true", help="Run with real data (requires CSV file)")
    
    args = parser.parse_args()
    
    # Default: use sample data
    use_sample = True
    if args.no_sample:
        use_sample = False
    
    try:
        main(use_sample=use_sample)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")
        print("📋 Details in logs/analysis.log")
        sys.exit(1)
