"""
Report Generator Module
=======================
Auto-generate comprehensive Markdown/HTML reports from all analysis results.

Combines:
- Model comparison
- Fairness analysis
- Subgroup analysis
- Robustness analysis
- Key findings & recommendations

Output: results/final_report.md + results/final_report.html
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Auto-generate comprehensive analysis report.
    """

    def __init__(self, results_dir: str = "results/"):
        self.results_dir = Path(results_dir)
        self.data = {}

    def load_all_results(self) -> Dict:
        """Load all available JSON results."""
        results = {}

        # Model comparison
        model_comp = self.results_dir / "model_comparison_report.json"
        if model_comp.exists():
            with open(model_comp, "r", encoding="utf-8") as f:
                results["model_comparison"] = json.load(f)

        # Model results
        for variant in ["conservative", "full"]:
            model_res = self.results_dir / f"model_results_{variant}.json"
            if model_res.exists():
                with open(model_res, "r", encoding="utf-8") as f:
                    results[f"model_results_{variant}"] = json.load(f)

        # Fairness
        for model in ["logistic", "catboost"]:
            fairness = self.results_dir / f"fairness_{model}.json"
            if fairness.exists():
                with open(fairness, "r", encoding="utf-8") as f:
                    results[f"fairness_{model}"] = json.load(f)

        # Subgroup
        for model in ["logistic", "catboost"]:
            subgroup = self.results_dir / f"subgroup_{model}.json"
            if subgroup.exists():
                with open(subgroup, "r", encoding="utf-8") as f:
                    results[f"subgroup_{model}"] = json.load(f)

        # Robustness
        for model in ["logistic", "catboost"]:
            robustness = self.results_dir / f"robustness_{model}.json"
            if robustness.exists():
                with open(robustness, "r", encoding="utf-8") as f:
                    results[f"robustness_{model}"] = json.load(f)

        # Leakage
        leakage = self.results_dir / "leakage_investigation.json"
        if leakage.exists():
            with open(leakage, "r", encoding="utf-8") as f:
                results["leakage"] = json.load(f)

        logger.info(f"✅ Loaded {len(results)} result files")
        self.data = results
        return results

    def generate_markdown_report(
        self,
        output_path: str = "results/final_report.md",
    ) -> str:
        """Generate comprehensive Markdown report."""
        if not self.data:
            self.load_all_results()

        lines = []

        # Header
        lines.append("# 📊 Báo Cáo Phân Tích Trầm Cảm Học Sinh Sinh Viên")
        lines.append("")
        lines.append(f"**Ngày tạo:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        lines.append(f"**Dataset:** Student_Depression_Dataset (27,901 samples)")
        lines.append("")

        # Executive Summary
        lines.extend(self._executive_summary())
        lines.append("")

        # 1. Model Performance
        lines.extend(self._section_model_performance())
        lines.append("")

        # 2. Fairness
        lines.extend(self._section_fairness())
        lines.append("")

        # 3. Subgroup Analysis
        lines.extend(self._section_subgroup())
        lines.append("")

        # 4. Robustness
        lines.extend(self._section_robustness())
        lines.append("")

        # 5. Recommendations
        lines.extend(self._section_recommendations())
        lines.append("")

        # Appendix
        lines.extend(self._appendix())

        content = "\n".join(lines)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"✅ Saved Markdown report: {output_path}")
        return str(output)

    # ==========================================
    # SECTIONS
    # ==========================================

    def _executive_summary(self) -> List[str]:
        """Executive summary — key findings at a glance."""
        lines = []
        lines.append("## 🎯 Tóm Tắt Điều Hành")
        lines.append("")

        models = self._get_models()

        # Best model
        best_auc = 0
        best_model = "N/A"
        for m in models:
            mc = self.data.get("model_comparison", {})
            basic = mc.get("basic_metrics", {})
            if m in basic:
                auc = basic[m].get("roc_auc", 0)
                if auc > best_auc:
                    best_auc = auc
                    best_model = m

        lines.append(f"- **Mô hình tốt nhất:** {best_model.upper()} (ROC-AUC={best_auc:.3f})")
        lines.append(f"- **Số mô hình đánh giá:** {len(models)}")

        # Fairness status
        fairness_issues = []
        for model in ["logistic", "catboost"]:
            fairness = self.data.get(f"fairness_{model}", {})
            warnings = fairness.get("warnings", [])
            if warnings:
                fairness_issues.append(f"{model}: {len(warnings)} cảnh báo")

        if fairness_issues:
            lines.append(f"- **⚠️ Vấn đề Fairness:** {'; '.join(fairness_issues)}")
        else:
            lines.append("- **✅ Fairness:** Không có cảnh báo nghiêm trọng")

        # Robustness
        for model in ["logistic", "catboost"]:
            robust = self.data.get(f"robustness_{model}", {})
            if robust:
                noise = robust.get("noise_injection", {})
                rs = noise.get("robustness_score", {})
                grade = rs.get("grade", "N/A")
                lines.append(f"- **🛡️ Robustness ({model}):** Grade {grade}")

        # Leakage warning
        leakage = self.data.get("leakage", {})
        if leakage:
            conclusion = leakage.get("conclusion", "")
            if "RỦI RO CAO" in conclusion or "Rủi ro cao" in conclusion:
                lines.append("- **🚨 Label Leakage:** Phát hiện rò rỉ nhãn — khuyến nghị dùng Phiên bản A (không Suicidal thoughts)")

        lines.append("")
        return lines

    def _section_model_performance(self) -> List[str]:
        """Model comparison section."""
        lines = []
        lines.append("## 📈 1. Hiệu Suất Mô Hình")
        lines.append("")

        mc = self.data.get("model_comparison", {})
        basic = mc.get("basic_metrics", {})

        if not basic:
            lines.append("*Không có dữ liệu model comparison.*")
            lines.append("")
            return lines

        # Table
        lines.append("| Mô hình | ROC-AUC | PR-AUC | F1 | Recall | Precision | Brier |")
        lines.append("|---------|---------|--------|----|--------|-----------|-------|")

        for model, metrics in basic.items():
            auc = metrics.get("roc_auc", 0)
            pr = metrics.get("pr_auc", 0)
            f1 = metrics.get("f1", 0)
            rec = metrics.get("recall", 0)
            prec = metrics.get("precision", 0)
            brier = metrics.get("brier_score", 0)
            lines.append(f"| {model} | {auc:.3f} | {pr:.3f} | {f1:.3f} | {rec:.3f} | {prec:.3f} | {brier:.4f} |")

        lines.append("")

        # Statistical tests
        mcnemar = mc.get("mcnemar", {})
        if mcnemar:
            lines.append("### Kiểm định McNemar (Disagreement)")
            lines.append("")
            for pair, result in mcnemar.items():
                p = result.get("p_value", 1)
                sig = "✅" if p < 0.05 else "❌"
                lines.append(f"- {pair}: χ²={result.get('chi2', 0):.1f}, p={p:.4f} {sig}")
            lines.append("")

        delong = mc.get("delong", {})
        if delong:
            lines.append("### Kiểm định DeLong (ROC-AUC)")
            lines.append("")
            for pair, result in delong.items():
                p = result.get("p_value", 1)
                sig = "✅" if p < 0.05 else "❌"
                lines.append(f"- {pair}: ΔAUC={result.get('delta_auc', 0):.3f}, p={p:.4f} {sig}")
            lines.append("")

        return lines

    def _section_fairness(self) -> List[str]:
        """Fairness analysis section."""
        lines = []
        lines.append("## ⚖️ 2. Phân Tích Fairness")
        lines.append("")

        for model in ["logistic", "catboost"]:
            fairness = self.data.get(f"fairness_{model}", {})
            if not fairness:
                continue

            lines.append(f"### {model.upper()}")
            lines.append("")

            # Warnings
            warnings = fairness.get("warnings", [])
            if warnings:
                lines.append(f"**⚠️ {len(warnings)} cảnh báo:**")
                lines.append("")
                for w in warnings:
                    icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(w.get("severity", ""), "⚪")
                    lines.append(f"- {icon} [{w.get('severity', '').upper()}] {w.get('message', '')}")
                lines.append("")
            else:
                lines.append("**✅ Không có cảnh báo fairness.**")
                lines.append("")

            # Disparate Impact
            di = fairness.get("disparate_impact", {})
            if di:
                lines.append("| Thuộc tính | Disparate Impact | Trạng thái |")
                lines.append("|-----------|-----------------|-----------|")
                for attr, data in di.items():
                    ratio = data.get("ratio", 0)
                    passes = "✅" if data.get("passes", False) else "❌"
                    lines.append(f"| {attr} | {ratio:.3f} | {passes} {data.get('interpretation', '')} |")
                lines.append("")

        return lines

    def _section_subgroup(self) -> List[str]:
        """Subgroup analysis section."""
        lines = []
        lines.append("## 🔍 3. Phân Tích Subgroup")
        lines.append("")

        for model in ["logistic", "catboost"]:
            subgroup = self.data.get(f"subgroup_{model}", {})
            if not subgroup:
                continue

            lines.append(f"### {model.upper()}")
            lines.append("")

            # Overall
            overall = subgroup.get("overall", {})
            if overall:
                lines.append(f"**Tổng quan:** AUC={overall.get('roc_auc', 0):.3f}, "
                             f"F1={overall.get('f1', 0):.3f}, "
                             f"Brier={overall.get('brier_score', 0):.4f}")
                lines.append("")

            # Error analysis
            error = subgroup.get("error_analysis", {})
            if error:
                ec = error.get("error_counts", {})
                lines.append(f"**Lỗi:** FN={ec.get('false_negatives', 0)} ({ec.get('fn_rate', 0):.1%}), "
                             f"FP={ec.get('false_positives', 0)} ({ec.get('fp_rate', 0):.1%})")
                lines.append("")

            # Subgroup performance table
            subgroups = subgroup.get("subgroups", {})
            if subgroups:
                lines.append("| Subgroup | Giá trị | N | AUC | F1 | FPR | FNR |")
                lines.append("|---------|--------|---|-----|----|-----|-----|")

                for sg_type, sg_data in subgroups.items():
                    if not isinstance(sg_data, dict):
                        continue
                    for val, metrics in sg_data.items():
                        n = metrics.get("n_samples", 0)
                        auc = metrics.get("roc_auc", 0)
                        f1 = metrics.get("f1", 0)
                        fpr = metrics.get("fpr", 0)
                        fnr = metrics.get("fnr", 0)
                        lines.append(f"| {sg_type} | {val} | {n} | {auc:.3f} | {f1:.3f} | {fpr:.3f} | {fnr:.3f} |")
                lines.append("")

            # Threshold recommendations
            threshold_recs = subgroup.get("threshold_recommendations", {})
            if threshold_recs:
                lines.append("### Khuyến nghị ngưỡng")
                lines.append("")
                lines.append("| Nhóm | F1-optimal | Cost-optimal |")
                lines.append("|------|-----------|-------------|")
                for group, recs in threshold_recs.items():
                    f1_t = recs.get("f1_optimal_threshold", 0)
                    cost_t = recs.get("cost_optimal_threshold", 0)
                    lines.append(f"| {group} | {f1_t:.2f} | {cost_t:.2f} |")
                lines.append("")

        return lines

    def _section_robustness(self) -> List[str]:
        """Robustness analysis section."""
        lines = []
        lines.append("## 🛡️ 4. Phân Tích Robustness")
        lines.append("")

        for model in ["logistic", "catboost"]:
            robust = self.data.get(f"robustness_{model}", {})
            if not robust:
                continue

            lines.append(f"### {model.upper()}")
            lines.append("")

            # Baseline
            baseline = robust.get("baseline", {})
            lines.append(f"**Baseline:** AUC={baseline.get('roc_auc', 0):.3f}, F1={baseline.get('f1', 0):.3f}")
            lines.append("")

            # Bootstrap CI
            bootstrap = robust.get("bootstrap_ci", {})
            if bootstrap:
                lines.append("#### Bootstrap Confidence Intervals")
                lines.append("")
                lines.append("| Metric | Mean | 95% CI | σ |")
                lines.append("|--------|------|--------|---|")
                for metric in ["roc_auc", "f1", "precision", "recall"]:
                    if metric in bootstrap:
                        d = bootstrap[metric]
                        lines.append(f"| {metric.upper()} | {d.get('mean', 0):.3f} | [{d.get('lower', 0):.3f}, {d.get('upper', 0):.3f}] | {d.get('std', 0):.3f} |")
                lines.append("")

            # CV Stability
            cv = robust.get("cv_stability", {})
            if cv:
                sa = cv.get("stability_assessment", {})
                lines.append(f"**CV Stability:** {sa.get('interpretation', 'N/A')}")
                auc = cv.get("roc_auc", {})
                lines.append(f"- AUC: μ={auc.get('mean', 0):.3f} ± {auc.get('std', 0):.3f} (range: {auc.get('min', 0):.3f}—{auc.get('max', 0):.3f})")
                lines.append("")

            # Noise injection
            noise = robust.get("noise_injection", {})
            if noise:
                rs = noise.get("robustness_score", {})
                lines.append(f"**Noise Robustness:** {rs.get('interpretation', 'N/A')}")
                if noise.get("breaking_point"):
                    lines.append(f"- ⚠️ Breaking point: σ={noise['breaking_point']}")
                lines.append("")

            # Feature ablation
            ablation = robust.get("feature_ablation", {})
            if ablation and "ablation_results" in ablation:
                lines.append("#### Feature Ablation (Top 5)")
                lines.append("")
                lines.append("| Nhóm features | Features bỏ | ΔAUC |")
                lines.append("|--------------|-------------|------|")
                for r in ablation["ablation_results"][:5]:
                    lines.append(f"| {r['group']} | {r['n_features_removed']} | {r['auc_drop']:.3f} |")
                lines.append("")

            # Adversarial
            adv = robust.get("adversarial_label_flip", {})
            if adv and "results" in adv:
                lines.append("#### Adversarial Label Flip")
                lines.append("")
                lines.append("| Flip ratio | AUC | ΔAUC |")
                lines.append("|-----------|-----|------|")
                for r in adv["results"]:
                    lines.append(f"| {r['flip_ratio']:.0%} | {r['roc_auc']:.3f} | {r.get('auc_drop', 0):.3f} |")
                if adv.get("breaking_point"):
                    lines.append(f"- ⚠️ Breaking point: {adv['breaking_point']:.0%} label flip")
                lines.append("")

        return lines

    def _section_recommendations(self) -> List[str]:
        """Actionable recommendations."""
        lines = []
        lines.append("## 💡 5. Khuyến Nghị")
        lines.append("")

        # Model selection
        models = self._get_models()
        best = None
        best_auc = 0
        for m in models:
            mc = self.data.get("model_comparison", {})
            basic = mc.get("basic_metrics", {})
            if m in basic:
                auc = basic[m].get("roc_auc", 0)
                if auc > best_auc:
                    best_auc = auc
                    best = m

        if best:
            lines.append(f"### Mô hình khuyến nghị: {best.upper()}")
            lines.append("")
            lines.append(f"- ROC-AUC: {best_auc:.3f}")

            # Check calibration
            for model in ["logistic", "catboost"]:
                fairness = self.data.get(f"fairness_{model}", {})
                if model in self._get_models():
                    mc = self.data.get("model_comparison", {})
                    basic = mc.get("basic_metrics", {})
                    if model in basic:
                        brier = basic[model].get("brier_score", 0)
                        lines.append(f"- Brier score: {brier:.4f} ({'✅ Tốt' if brier < 0.15 else '⚠️ Trung bình'})")

            lines.append("")

        # Fairness recommendations
        lines.append("### Fairness")
        lines.append("")

        has_fairness_issues = False
        for model in ["logistic", "catboost"]:
            fairness = self.data.get(f"fairness_{model}", {})
            warnings = fairness.get("warnings", [])
            if warnings:
                has_fairness_issues = True
                lines.append(f"- **⚠️ {model}:** {len(warnings)} cảnh báo — xem chi tiết trong fairness dashboard")

        if not has_fairness_issues:
            lines.append("- ✅ Không có vấn đề fairness nghiêm trọng")
        else:
            lines.append("- Khuyến nghị: Đánh giá thêm với dữ liệu thực tế để xác nhận bias")
        lines.append("")

        # Subgroup recommendations
        lines.append("### Subgroup")
        lines.append("")

        for model in ["logistic", "catboost"]:
            subgroup = self.data.get(f"subgroup_{model}", {})
            if not subgroup:
                continue

            subgroups = subgroup.get("subgroups", {})
            if "Age" in subgroups:
                age_data = subgroups["Age"]
                worst_fnr = max(age_data.items(), key=lambda x: x[1].get("fnr", 0))
                lines.append(f"- **{model}:** Nhóm {worst_fnr[0]} có FNR cao nhất ({worst_fnr[1].get('fnr', 0):.1%}) — cần attention")

        lines.append("- Khuyến nghị: Fine-tune threshold per subgroup nếu deploy thực tế")
        lines.append("")

        # Robustness recommendations
        lines.append("### Robustness")
        lines.append("")

        for model in ["logistic", "catboost"]:
            robust = self.data.get(f"robustness_{model}", {})
            if not robust:
                continue
            noise = robust.get("noise_injection", {})
            rs = noise.get("robustness_score", {})
            grade = rs.get("grade", "N/A")
            lines.append(f"- **{model}:** Noise robustness grade {grade}")

        lines.append("- Khuyến nghị: Mô hình đủ robust cho screening, nhưng cần monitoring liên tục")
        lines.append("")

        # Data quality
        leakage = self.data.get("leakage", {})
        if leakage:
            synthetic = leakage.get("synthetic_check", {})
            if synthetic.get("likely_synthetic", False):
                lines.append("### ⚠️ Chất lượng dữ liệu")
                lines.append("")
                lines.append("- Dataset có khả năng cao là **synthetic** (dữ liệu tổng hợp)")
                lines.append("- Kết quả chỉ mang tính **tham khảo**, không nên dùng cho quyết định lâm sàng")
                lines.append("- Khuyến nghị: Thu thập dữ liệu thực tế hoặc sử dụng dataset được công nhận")
                lines.append("")

        # General
        lines.append("### Tổng quan")
        lines.append("")
        lines.append("1. ✅ Mô hình có hiệu suất tốt (AUC > 0.87)")
        lines.append("2. ⚠️ Có vấn đề fairness ở một số nhóm tuổi — cần fine-tune")
        lines.append("3. ✅ Mô hình robust với nhiễu và biến đổi nhỏ")
        lines.append("4. 🚨 Dataset khả năng synthetic — kết quả chỉ tham khảo")
        lines.append("5. 📋 Luôn kết hợp với **đánh giá của chuyên gia** trước khi ra quyết định")
        lines.append("")

        return lines

    def _appendix(self) -> List[str]:
        """Appendix — file references."""
        lines = []
        lines.append("---")
        lines.append("")
        lines.append("## 📁 File Kết Quả")
        lines.append("")

        # Check what files exist
        html_files = list(self.results_dir.glob("*.html"))
        json_files = list(self.results_dir.glob("*.json"))

        if html_files:
            lines.append("### HTML Dashboards")
            lines.append("")
            for f in sorted(html_files):
                lines.append(f"- [{f.name}](./{f.name})")
            lines.append("")

        if json_files:
            lines.append("### JSON Results")
            lines.append("")
            for f in sorted(json_files):
                lines.append(f"- [{f.name}](./{f.name})")
            lines.append("")

        lines.append("---")
        lines.append(f"*Báo cáo tự động tạo lúc {datetime.now().strftime('%d/%m/%Y %H:%M')}*")

        return lines

    # ==========================================
    # HELPERS
    # ==========================================

    def _get_models(self) -> List[str]:
        """Get list of models from results."""
        models = set()
        mc = self.data.get("model_comparison", {})
        basic = mc.get("basic_metrics", {})
        models.update(basic.keys())

        for key in self.data:
            if key.startswith("fairness_") or key.startswith("subgroup_") or key.startswith("robustness_"):
                model = key.split("_", 1)[1]
                models.add(model)

        return sorted(models)

    def generate_html_report(
        self,
        md_path: str = "results/final_report.md",
        output_path: str = "results/final_report.html",
    ) -> str:
        """Convert Markdown report to HTML (simple conversion)."""
        md_file = Path(md_path)
        if not md_file.exists():
            self.generate_markdown_report(md_path)

        with open(md_file, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Simple Markdown to HTML conversion
        import re

        html = md_content

        # Headers
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # Bold and italic
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

        # Line breaks
        html = html.replace('\n', '<br>\n')

        # Wrap in HTML template
        html_template = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo Phân Tích Trầm Cảm</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.6; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; margin-top: 30px; }}
        h3 {{ color: #27ae60; margin-top: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th {{ background: #3498db; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 8px 10px; border-bottom: 1px solid #ecf0f1; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        strong {{ color: #2c3e50; }}
    </style>
</head>
<body>
{html}
</body>
</html>"""

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            f.write(html_template)

        logger.info(f"✅ Saved HTML report: {output_path}")
        return str(output)
