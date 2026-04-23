from __future__ import annotations

import argparse
import sys
from typing import Any, Callable


def run_modern_pipeline_cli(
    dataset_path: str,
    preset: str = "quick",
    profile: str = "safe",
    compare: bool = False,
    export_html: bool = False,
    console_only: bool = False,
    output_dir: str = "results/app",
) -> None:
    from src.app import ArtifactPolicy, RunPreset, RunProfile
    from src.app import compare_profiles, load_dataset as load_dataset_v2
    from src.app import profile_dataset as profile_dataset_v2
    from src.app import run_pipeline as run_pipeline_v2

    artifact_policy = ArtifactPolicy.CONSOLE_ONLY if console_only else (
        ArtifactPolicy.FULL_EXPORT if export_html else ArtifactPolicy.JSON
    )
    bundle = load_dataset_v2(dataset_path)
    profile_report = profile_dataset_v2(
        bundle=bundle,
        artifact_policy=artifact_policy,
        export_html=export_html,
        output_dir=output_dir,
    )

    print("\n" + "=" * 80)
    print(" MODERN PIPELINE - HOLDOUT-FIRST")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"   OK {profile_report.summary['rows']:,} rows x {profile_report.summary['cols']} cols")
    print(f"   OK Positive rate: {profile_report.summary['target_positive_rate']}%")
    print(f"   OK Cache: {'used' if profile_report.summary['loaded_from_cache'] else 'fresh load'}")

    if profile_report.warnings:
        print("\nProfile warnings:")
        for warning in profile_report.warnings:
            print(f"   - {warning}")

    if compare:
        comparison = compare_profiles(
            bundle=bundle,
            preset=RunPreset(preset),
            artifact_policy=artifact_policy,
            output_dir=output_dir,
        )
        print("\nSafe vs Full comparison:")
        for model_name, summary in comparison.summary.items():
            print(f"   - {model_name}:")
            print(f"      safe ROC-AUC = {summary['safe_roc_auc']}")
            print(f"      full ROC-AUC = {summary['full_roc_auc']}")
            print(f"      delta        = {summary['roc_auc_delta_full_minus_safe']}")
            print(f"      safe F1      = {summary['safe_f1']}")
            print(f"      full F1      = {summary['full_f1']}")
        if comparison.artifacts:
            print("\nArtifacts:")
            for name, path in comparison.artifacts.items():
                print(f"   - {name}: {path}")
        return

    report = run_pipeline_v2(
        bundle=bundle,
        profile=RunProfile(profile),
        preset=RunPreset(preset),
        artifact_policy=artifact_policy,
        output_dir=output_dir,
    )
    print(f"\nRun config: profile={report.config['profile']} preset={report.config['preset']}")
    print(f"   models: {', '.join(report.config['models'])}")
    print(f"   selected columns: {', '.join(report.config['selected_columns'])}")
    rust_engine = report.config.get("rust_engine")
    if rust_engine:
        if rust_engine.get("available"):
            print(f"   rust engine: available ({rust_engine.get('version')})")
        else:
            print(f"   rust engine: fallback ({rust_engine.get('error')})")

    for model_name, result in report.models.items():
        print(f"\n{model_name.upper()}:")
        if result.metadata.get("engine"):
            print(f"   Engine: {result.metadata['engine']}")
        print(f"   OOF ROC-AUC:     {result.oof.get('roc_auc')}")
        print(f"   Holdout ROC-AUC: {result.holdout.get('roc_auc')}")
        print(f"   Holdout F1:      {result.holdout.get('f1')}")
        print(f"   Holdout Recall:  {result.holdout.get('recall')}")
        print(f"   Holdout Brier:   {result.holdout.get('brier_score')}")
        print(f"   Best F1 threshold: {result.thresholds['best_f1']['threshold']}")

    print("\nTimings:")
    for stage, seconds in report.timings.items():
        print(f"   - {stage}: {seconds}s")

    if report.warnings:
        print("\nRun warnings:")
        for warning in report.warnings:
            print(f"   - {warning}")

    if report.artifacts:
        print("\nArtifacts:")
        for name, path in report.artifacts.items():
            print(f"   - {name}: {path}")


def build_main_parser(default_dataset: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Depression analysis legacy pipeline plus modern holdout-first pipeline"
    )
    parser.add_argument("--eda", action="store_true", help="Legacy stage 1: EDA")
    parser.add_argument("--stats", action="store_true", help="Legacy stage 4: statistical analysis")
    parser.add_argument("--models", action="store_true", help="Legacy stage 7-8: modeling")
    parser.add_argument("--leakage", action="store_true", help="Legacy leakage investigation")
    parser.add_argument("--full", action="store_true", help="Run the full legacy pipeline")
    parser.add_argument("--no-ethical", action="store_true", help="Skip the ethical banner")
    parser.add_argument("--conservative", action="store_true", help="Legacy safe profile without suicidal-thoughts")
    parser.add_argument("--review", action="store_true", help="Legacy data review")
    parser.add_argument("--standardize", action="store_true", help="Legacy standardization")
    parser.add_argument("--famd", action="store_true", help="Legacy FAMD analysis")
    parser.add_argument("--split", action="store_true", help="Legacy stratified split report")
    parser.add_argument("--fairness", action="store_true", help="Legacy fairness analysis")
    parser.add_argument("--subgroups", action="store_true", help="Legacy subgroup analysis")
    parser.add_argument("--robustness", action="store_true", help="Legacy robustness analysis")
    parser.add_argument("--analysis", action="store_true", help="Legacy fairness + subgroup + robustness")
    parser.add_argument("--report", action="store_true", help="Legacy report generation")
    parser.add_argument("--quick", action="store_true", help="Modern pipeline: quick preset")
    parser.add_argument("--research", action="store_true", help="Modern pipeline: research preset")
    parser.add_argument("--profile", type=str, default="safe", choices=["safe", "full"],
                        help="Modern pipeline profile")
    parser.add_argument("--compare-profiles", action="store_true",
                        help="Modern pipeline: compare safe vs full")
    parser.add_argument("--export-html", action="store_true",
                        help="Modern pipeline: export HTML EDA and artifacts")
    parser.add_argument("--console-only", action="store_true",
                        help="Modern pipeline: do not write JSON/HTML artifacts")
    parser.add_argument("--output-dir", type=str, default="results/app",
                        help="Modern pipeline artifact directory")
    parser.add_argument("--dataset", type=str, default=default_dataset, help="Path to dataset CSV")
    return parser


def dispatch_main_cli(
    legacy_main: Callable[..., Any],
    default_dataset: str,
    logger: Any,
) -> None:
    parser = build_main_parser(default_dataset)
    args = parser.parse_args()

    any_flag = (
        args.eda or args.stats or args.models or args.leakage or args.full
        or args.review or args.standardize or args.famd or args.split
        or args.fairness or args.subgroups or args.robustness or args.analysis or args.report
        or args.quick or args.research or args.compare_profiles
    )

    try:
        if args.quick or args.research or args.compare_profiles:
            run_modern_pipeline_cli(
                dataset_path=args.dataset,
                preset="research" if args.research else "quick",
                profile=args.profile,
                compare=args.compare_profiles,
                export_html=args.export_html,
                console_only=args.console_only,
                output_dir=args.output_dir,
            )
        elif args.full:
            legacy_main(
                dataset_path=args.dataset,
                run_ethical=not args.no_ethical,
                run_eda_flag=True,
                run_stats=True,
                run_models=True,
                run_review=True,
                run_standardize=True,
                run_famd=False,
                run_split=False,
                conservative=args.conservative,
                run_fairness=args.analysis,
                run_subgroups=args.analysis,
                run_robustness=args.analysis,
            )
        elif any_flag:
            legacy_main(
                dataset_path=args.dataset,
                run_ethical=not args.no_ethical,
                run_eda_flag=args.eda,
                run_stats=args.stats,
                run_models=args.models,
                run_leakage=args.leakage,
                run_review=args.review,
                run_standardize=args.standardize,
                run_famd=args.famd,
                run_split=args.split,
                conservative=args.conservative,
                run_fairness=args.fairness or args.analysis,
                run_subgroups=args.subgroups or args.analysis,
                run_robustness=args.robustness or args.analysis,
                run_report=args.report,
            )
        else:
            legacy_main(
                dataset_path=args.dataset,
                run_ethical=True,
                run_eda_flag=True,
                run_stats=False,
                run_models=False,
            )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Pipeline failed: {str(exc)}", exc_info=True)
        print(f"\nError: {str(exc)}")
        print("Details in logs/analysis.log")
        import traceback

        traceback.print_exc()
        sys.exit(1)
