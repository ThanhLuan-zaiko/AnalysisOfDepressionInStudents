"""
Machine Learning Models Module
Train depression risk prediction models

Theo kế hoạch khoa học:
  - Logistic Regression (trung tâm, giải thích được)
  - GAM (linh hoạt, vẫn diễn giải được)
  - CatBoost (dự báo bổ sung)
  - Dummy baseline
  - Fairness, Subgroup, Robustness analysis
"""

from .predictor import DepressionPredictor
from .optimizer import HyperparameterOptimizer
from .explainer import SHAPExplainer
from .imbalanced import ImbalancedDataHandler
from .risk_model import DepressionRiskModeler
from .leakage_check import LabelLeakageInvestigator
from .famd import FAMDAnalyzer
from .stratified_split import StratifiedSplitter
from .gam_model import GAMClassifier
from .model_comparator import ModelComparator
from .fairness_analysis import FairnessAnalyzer
from .subgroup_analysis import SubgroupAnalyzer
from .robustness import RobustnessAnalyzer

__all__ = [
    "DepressionPredictor",
    "HyperparameterOptimizer",
    "SHAPExplainer",
    "ImbalancedDataHandler",
    "DepressionRiskModeler",
    "LabelLeakageInvestigator",
    "FAMDAnalyzer",
    "StratifiedSplitter",
    "GAMClassifier",
    "ModelComparator",
    "FairnessAnalyzer",
    "SubgroupAnalyzer",
    "RobustnessAnalyzer",
]
