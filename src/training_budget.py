from __future__ import annotations

from typing import Any


TrainingBudgetMode = str


def resolve_training_budget(
    *,
    mode: TrainingBudgetMode,
    family: str,
    preset: str,
    train_rows: int,
) -> dict[str, dict[str, Any]]:
    family_name = family.lower()
    preset_name = preset.lower()

    if family_name not in {"modern", "legacy"}:
        raise ValueError(f"Unsupported training budget family: {family}")
    if mode not in {"default", "auto"}:
        raise ValueError(f"Unsupported training budget mode: {mode}")

    if family_name == "modern":
        budget = _modern_defaults(preset_name)
    else:
        budget = _legacy_defaults()

    if mode == "auto":
        budget = _apply_auto_budget(budget, family=family_name, preset=preset_name, train_rows=train_rows)

    return budget


def _modern_defaults(preset: str) -> dict[str, dict[str, Any]]:
    return {
        "logistic": {"max_iter": 1500},
        "catboost": {"iterations": 300, "early_stopping_rounds": 30, "learning_rate": 0.05, "depth": 6},
        "gam": {
            "n_splines": 12 if preset == "research" else 10,
            "optimize_splines": bool(preset == "research"),
        },
    }


def _legacy_defaults() -> dict[str, dict[str, Any]]:
    return {
        "logistic": {"max_iter": 1000},
        "catboost": {"iterations": 500, "early_stopping_rounds": 30, "learning_rate": 0.05, "depth": 6},
        "gam": {"n_splines": 15, "optimize_splines": True},
    }


def _apply_auto_budget(
    budget: dict[str, dict[str, Any]],
    *,
    family: str,
    preset: str,
    train_rows: int,
) -> dict[str, dict[str, Any]]:
    scale = _size_scale(train_rows)
    tuned = {name: values.copy() for name, values in budget.items()}

    if family == "modern":
        tuned["logistic"]["max_iter"] = {0: 900, 1: 1300, 2: 1800}[scale]
        tuned["catboost"]["iterations"] = {0: 220, 1: 320, 2: 420}[scale]
        tuned["catboost"]["early_stopping_rounds"] = {0: 20, 1: 30, 2: 40}[scale]
        tuned["gam"]["n_splines"] = {
            ("quick", 0): 8,
            ("quick", 1): 10,
            ("quick", 2): 12,
            ("research", 0): 10,
            ("research", 1): 12,
            ("research", 2): 14,
        }[(preset, scale)]
        tuned["gam"]["optimize_splines"] = bool(preset == "research" or scale >= 1)
    else:
        tuned["logistic"]["max_iter"] = {0: 800, 1: 1100, 2: 1500}[scale]
        tuned["catboost"]["iterations"] = {0: 260, 1: 420, 2: 650}[scale]
        tuned["catboost"]["early_stopping_rounds"] = {0: 20, 1: 35, 2: 50}[scale]
        tuned["gam"]["n_splines"] = {0: 10, 1: 13, 2: 17}[scale]
        tuned["gam"]["optimize_splines"] = True

    return tuned


def _size_scale(train_rows: int) -> int:
    if train_rows < 400:
        return 0
    if train_rows < 1500:
        return 1
    return 2
