"""
Machine Learning Models Module

Keep this package light on import. Heavy optional dependencies such as SHAP/IPython
should only load when the corresponding symbol is actually requested.
"""

from __future__ import annotations

from importlib import import_module

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

_SYMBOL_TO_MODULE = {
    "DepressionPredictor": ".predictor",
    "HyperparameterOptimizer": ".optimizer",
    "SHAPExplainer": ".explainer",
    "ImbalancedDataHandler": ".imbalanced",
    "DepressionRiskModeler": ".risk_model",
    "LabelLeakageInvestigator": ".leakage_check",
    "FAMDAnalyzer": ".famd",
    "StratifiedSplitter": ".stratified_split",
    "GAMClassifier": ".gam_model",
    "ModelComparator": ".model_comparator",
    "FairnessAnalyzer": ".fairness_analysis",
    "SubgroupAnalyzer": ".subgroup_analysis",
    "RobustnessAnalyzer": ".robustness",
}


def __getattr__(name: str):
    if name not in _SYMBOL_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_SYMBOL_TO_MODULE[name], __name__)
    return getattr(module, name)
