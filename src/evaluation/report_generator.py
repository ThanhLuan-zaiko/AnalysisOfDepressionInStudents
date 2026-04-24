"""
Vietnamese HTML report generator for depression analysis results.

The generator prefers the modern holdout-first A/B artifact:
``results/app/compare_profiles_research.json`` or ``compare_profiles_quick.json``.
It falls back to legacy result JSON when modern artifacts are not available.
"""

from __future__ import annotations

import html
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


MODEL_LABELS = {
    "dummy": "Dummy baseline",
    "logistic": "Logistic Regression",
    "gam": "GAM",
    "catboost": "CatBoost",
}


class ReportGenerator:
    """Generate Markdown and HTML reports from available analysis artifacts."""

    def __init__(self, results_dir: str = "results/"):
        self.results_dir = Path(results_dir)
        self.app_dir = self.results_dir / "app"
        self.data: dict[str, Any] = {}
        self.generated_artifacts: dict[str, str] = {}

    def load_all_results(self) -> dict[str, Any]:
        results: dict[str, Any] = {}

        compare_path = self._latest_existing(
            [
                self.app_dir / "compare_profiles_research.json",
                self.app_dir / "compare_profiles_quick.json",
            ]
        )
        if compare_path is not None:
            results["modern_compare"] = self._read_json(compare_path)
            results["modern_compare_path"] = str(compare_path)

        for profile in ("safe", "full"):
            run_path = self._latest_existing(
                [
                    self.app_dir / f"run_{profile}_research.json",
                    self.app_dir / f"run_{profile}_quick.json",
                ]
            )
            if run_path is not None:
                results[f"modern_run_{profile}"] = self._read_json(run_path)

        legacy_comp = self.results_dir / "model_comparison_report.json"
        if legacy_comp.exists():
            results["legacy_model_comparison"] = self._read_json(legacy_comp)

        for variant in ("conservative", "full"):
            path = self.results_dir / f"model_results_{variant}.json"
            if path.exists():
                results[f"legacy_model_results_{variant}"] = self._read_json(path)

        famd_cluster = self.results_dir / "visualizations" / "famd_clustering_results.json"
        if famd_cluster.exists():
            results["famd_clustering"] = self._read_json(famd_cluster)

        self.data = results
        logger.info("Loaded %s result groups for final report", len(results))
        return results

    def generate_markdown_report(self, output_path: str = "results/final_report.md") -> str:
        if not self.data:
            self.load_all_results()

        selection = self._select_best_model()
        compare = self.data.get("modern_compare")
        lines = [
            "# Báo cáo phân tích trầm cảm học sinh sinh viên",
            "",
            f"Ngày tạo: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            "",
            "## Kết luận model",
            "",
            self._selection_sentence(selection),
            "",
        ]

        if compare:
            lines.extend(["## So sánh Safe A và Full B", ""])
            for model_name in self._model_order(compare.get("summary", {}).keys()):
                summary = compare["summary"][model_name]
                lines.append(
                    f"- {MODEL_LABELS.get(model_name, model_name)}: "
                    f"Safe ROC-AUC={self._fmt(summary.get('safe_roc_auc'))}, "
                    f"Full ROC-AUC={self._fmt(summary.get('full_roc_auc'))}, "
                    f"delta={self._fmt_signed(summary.get('roc_auc_delta_full_minus_safe'))}"
                )
            lines.append("")
        else:
            lines.extend(["## Ghi chú", "", "Chưa có artifact modern A/B; báo cáo dùng kết quả legacy nếu có.", ""])

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines), encoding="utf-8")
        return str(output)

    def generate_html_report(
        self,
        md_path: str = "results/final_report.md",
        output_path: str = "results/final_report.html",
    ) -> str:
        if not self.data:
            self.load_all_results()

        selection = self._select_best_model()
        self._write_best_selection(selection)
        self._generate_evidence_charts()

        html_content = self._build_html(selection)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html_content, encoding="utf-8")
        return str(output)

    # ------------------------------------------------------------------
    # HTML sections
    # ------------------------------------------------------------------

    def _build_html(self, selection: dict[str, Any]) -> str:
        compare = self.data.get("modern_compare")
        charts = self._chart_links_html()
        body = [
            "<h1>Báo cáo phân tích trầm cảm học sinh sinh viên</h1>",
            f"<p class='muted'>Tạo lúc {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>",
            "<div class='notice'>Đây là phân tích hỗ trợ sàng lọc, không phải công cụ chẩn đoán hay thay thế đánh giá chuyên môn.</div>",
            "<h2>Kết luận điều hành</h2>",
            f"<p>{html.escape(self._selection_sentence(selection))}</p>",
            charts,
        ]

        if compare:
            body.extend(
                [
                    "<h2>So sánh A/B trên cùng holdout split</h2>",
                    "<p><strong>Safe A</strong> loại cột suicidal thoughts để giảm leakage. "
                    "<strong>Full B</strong> giữ toàn bộ biến để xem mức tăng hiệu năng nghiên cứu.</p>",
                    self._ab_summary_table(compare),
                    self._profile_metric_table(compare, "safe", "Safe A - báo cáo chính"),
                    self._profile_metric_table(compare, "full", "Full B - chỉ dùng tham chiếu leakage"),
                    self._model_evidence_section(compare),
                    self._feature_importance_section(compare),
                    self._threshold_section(compare, selection),
                    self._fairness_section(compare),
                ]
            )
        else:
            body.extend(
                [
                    "<h2>Kết quả legacy</h2>",
                    "<p>Không tìm thấy artifact A/B hiện đại trong results/app. Bảng dưới dùng file legacy nếu có.</p>",
                    self._legacy_table(),
                    self._legacy_evidence_section(selection),
                ]
            )

        body.append(self._famd_clustering_section())
        body.append(self._appendix_section())

        return f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Báo cáo phân tích trầm cảm</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1180px; margin: 0 auto; padding: 24px; color: #1f2933; line-height: 1.55; }}
    h1, h2, h3 {{ color: #102a43; }}
    h1 {{ border-bottom: 3px solid #2f6f9f; padding-bottom: 10px; }}
    h2 {{ margin-top: 34px; border-bottom: 1px solid #d9e2ec; padding-bottom: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0 26px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 9px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #e6f6ff; color: #102a43; }}
    tr:nth-child(even) {{ background: #f8fafc; }}
    .notice {{ background: #fff7e6; border-left: 4px solid #d9822b; padding: 12px 14px; margin: 16px 0; }}
    .muted {{ color: #627d98; }}
    .good {{ color: #0f7b4f; font-weight: 700; }}
    .warn {{ color: #b54708; font-weight: 700; }}
    .bad {{ color: #b42318; font-weight: 700; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #d9e2ec; border-radius: 6px; padding: 12px 14px; }}
    a {{ color: #1d4ed8; }}
  </style>
</head>
<body>
{''.join(body)}
</body>
</html>"""

    def _ab_summary_table(self, compare: dict[str, Any]) -> str:
        rows = []
        for model_name in self._model_order(compare.get("summary", {}).keys()):
            summary = compare["summary"][model_name]
            delta = summary.get("roc_auc_delta_full_minus_safe")
            delta_class = "warn" if delta is not None and float(delta) > 0.02 else ""
            rows.append(
                "<tr>"
                f"<td>{MODEL_LABELS.get(model_name, model_name)}</td>"
                f"<td>{self._fmt(summary.get('safe_roc_auc'))}</td>"
                f"<td>{self._fmt(summary.get('full_roc_auc'))}</td>"
                f"<td class='{delta_class}'>{self._fmt_signed(delta)}</td>"
                f"<td>{self._fmt(summary.get('safe_f1'))}</td>"
                f"<td>{self._fmt(summary.get('full_f1'))}</td>"
                "</tr>"
            )
        return (
            "<table><thead><tr><th>Model</th><th>Safe ROC-AUC</th><th>Full ROC-AUC</th>"
            "<th>Delta Full-Safe</th><th>Safe F1</th><th>Full F1</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    def _profile_metric_table(self, compare: dict[str, Any], profile: str, title: str) -> str:
        models = compare.get("profiles", {}).get(profile, {}).get("models", {})
        rows = []
        for model_name in self._model_order(models.keys()):
            result = models[model_name]
            holdout = result.get("holdout", {})
            cm = holdout.get("confusion_matrix", {})
            rows.append(
                "<tr>"
                f"<td>{MODEL_LABELS.get(model_name, model_name)}</td>"
                f"<td>{self._fmt(holdout.get('roc_auc'))}</td>"
                f"<td>{self._fmt(holdout.get('pr_auc'))}</td>"
                f"<td>{self._fmt(holdout.get('f1'))}</td>"
                f"<td>{self._fmt(holdout.get('recall'))}</td>"
                f"<td>{self._fmt(holdout.get('precision'))}</td>"
                f"<td>{self._fmt(holdout.get('brier_score'))}</td>"
                f"<td>TN={cm.get('tn', 'N/A')}, FP={cm.get('fp', 'N/A')}, FN={cm.get('fn', 'N/A')}, TP={cm.get('tp', 'N/A')}</td>"
                "</tr>"
            )
        return (
            f"<h3>{html.escape(title)}</h3>"
            "<table><thead><tr><th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>F1</th>"
            "<th>Recall</th><th>Precision</th><th>Brier</th><th>Confusion matrix</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    def _model_evidence_section(self, compare: dict[str, Any]) -> str:
        safe_models = compare.get("profiles", {}).get("safe", {}).get("models", {})
        cards = []
        best_name = self._select_best_model().get("model")
        for model_name in self._model_order(safe_models.keys()):
            holdout = safe_models[model_name].get("holdout", {})
            reason = self._model_reason(model_name, holdout, best_name)
            cards.append(
                "<div class='card'>"
                f"<h3>{MODEL_LABELS.get(model_name, model_name)}</h3>"
                f"<p>{reason}</p>"
                f"<p>ROC-AUC={self._fmt(holdout.get('roc_auc'))}, F1={self._fmt(holdout.get('f1'))}, "
                f"Brier={self._fmt(holdout.get('brier_score'))}</p>"
                "</div>"
            )
        return "<h2>Bằng chứng model yếu/mạnh</h2><div class='cards'>" + "".join(cards) + "</div>"

    def _feature_importance_section(self, compare: dict[str, Any]) -> str:
        models = compare.get("profiles", {}).get("safe", {}).get("models", {})
        parts = ["<h2>Bằng chứng bằng feature importance</h2>"]
        if "model_feature_importance_safe.html" in self.generated_artifacts:
            parts.append("<p><a href='./model_feature_importance_safe.html'>Mở biểu đồ feature importance Safe A</a></p>")
        for model_name in self._model_order(models.keys()):
            importance = models[model_name].get("feature_importance", [])[:8]
            if not importance:
                continue
            rows = []
            for item in importance:
                value = item.get("importance", item.get("abs_coefficient", item.get("variance_importance")))
                rows.append(f"<tr><td>{html.escape(str(item.get('feature')))}</td><td>{self._fmt(value)}</td></tr>")
            parts.append(
                f"<h3>{MODEL_LABELS.get(model_name, model_name)}</h3>"
                "<table><thead><tr><th>Feature</th><th>Evidence value</th></tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table>"
            )
        return "".join(parts)

    def _threshold_section(self, compare: dict[str, Any], selection: dict[str, Any]) -> str:
        profile = selection.get("profile", "safe")
        model_name = selection.get("model")
        model = compare.get("profiles", {}).get(profile, {}).get("models", {}).get(model_name, {})
        thresholds = model.get("thresholds", {})
        best_f1 = thresholds.get("best_f1", {})
        screening = thresholds.get("screening", {})
        return (
            "<h2>Ngưỡng khuyến nghị</h2>"
            f"<p>Model được chọn: <strong>{MODEL_LABELS.get(str(model_name), model_name)}</strong> trên profile {profile}. "
            f"Ngưỡng F1 tốt nhất = <strong>{best_f1.get('threshold', 'N/A')}</strong> "
            f"(F1={self._fmt(best_f1.get('f1'))}, recall={self._fmt(best_f1.get('recall'))}). "
            f"Ngưỡng ưu tiên sàng lọc = <strong>{screening.get('threshold', 'N/A')}</strong> "
            f"(recall={self._fmt(screening.get('recall'))}, flagged={self._fmt(screening.get('flagged_pct'))}%).</p>"
        )

    def _fairness_section(self, compare: dict[str, Any]) -> str:
        safe_models = compare.get("profiles", {}).get("safe", {}).get("models", {})
        best = self._select_best_model().get("model")
        fairness = safe_models.get(best, {}).get("fairness", [])
        if not fairness:
            return "<h2>Fairness theo nhóm</h2><p>Chưa có bảng subgroup/fairness trong artifact.</p>"
        rows = []
        for item in fairness:
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('family')))}</td>"
                f"<td>{html.escape(str(item.get('label')))}</td>"
                f"<td>{item.get('n_samples')}</td>"
                f"<td>{self._fmt(item.get('roc_auc'))}</td>"
                f"<td>{self._fmt(item.get('f1'))}</td>"
                f"<td>{self._fmt(item.get('fnr'))}</td>"
                "</tr>"
            )
        return (
            "<h2>Fairness theo nhóm cho model được chọn</h2>"
            "<table><thead><tr><th>Nhóm</th><th>Giá trị</th><th>N</th><th>ROC-AUC</th><th>F1</th><th>FNR</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    def _famd_clustering_section(self) -> str:
        clustering = self.data.get("famd_clustering")
        if not clustering:
            return "<h2>FAMD clustering</h2><p>Chưa có kết quả clustering. Chạy <code>uv run python main.py --famd</code> để tạo.</p>"
        kmeans = clustering["kmeans"]
        dbscan = clustering["dbscan"]
        link = "visualizations/famd_clustering_report.html"
        return (
            "<h2>FAMD clustering</h2>"
            f"<p>K-Means chọn k={kmeans['best_k']} với silhouette={self._fmt(kmeans['silhouette'])}. "
            f"DBSCAN {'tìm được ' + str(dbscan['n_clusters']) + ' cụm' if dbscan['found_valid_clusters'] else 'không tìm được cụm mật độ rõ'}.</p>"
            f"<p><a href='./{link}'>Mở báo cáo FAMD clustering chi tiết</a></p>"
        )

    def _appendix_section(self) -> str:
        links = []
        for name, path in sorted(self.generated_artifacts.items()):
            links.append(f"<li><a href='./{Path(path).name}'>{html.escape(name)}</a></li>")
        for candidate in [
            self.results_dir / "visualizations" / "famd_clustering_report.html",
            self.results_dir / "visualizations" / "famd_clusters_kmeans.html",
            self.results_dir / "visualizations" / "famd_clusters_dbscan.html",
        ]:
            if candidate.exists():
                links.append(f"<li><a href='./visualizations/{candidate.name}'>{candidate.name}</a></li>")
        return "<h2>Phụ lục artifact</h2><ul>" + "".join(links) + "</ul>"

    def _legacy_table(self) -> str:
        rows = []
        for model_name, metrics in self._legacy_metrics().items():
            rows.append(
                "<tr>"
                f"<td>{MODEL_LABELS.get(model_name, model_name)}</td>"
                f"<td>{self._fmt(metrics.get('roc_auc'))}</td>"
                f"<td>{self._fmt(metrics.get('pr_auc'))}</td>"
                f"<td>{self._fmt(metrics.get('f1'))}</td>"
                f"<td>{self._fmt(metrics.get('brier_score'))}</td>"
                "</tr>"
            )
        if not rows:
            return "<p>Không có kết quả model legacy.</p>"
        return (
            "<table><thead><tr><th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>F1</th><th>Brier</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    def _legacy_evidence_section(self, selection: dict[str, Any]) -> str:
        metrics = self._legacy_metrics()
        if not metrics:
            return ""
        cards = []
        best_name = selection.get("model")
        for model_name in self._model_order(metrics.keys()):
            row = metrics[model_name]
            reason = self._model_reason(model_name, row, best_name)
            cards.append(
                "<div class='card'>"
                f"<h3>{MODEL_LABELS.get(model_name, model_name)}</h3>"
                f"<p>{reason}</p>"
                f"<p>ROC-AUC={self._fmt(row.get('roc_auc'))}, PR-AUC={self._fmt(row.get('pr_auc'))}, "
                f"F1={self._fmt(row.get('f1'))}, Brier={self._fmt(row.get('brier_score'))}</p>"
                "</div>"
            )
        return "<h2>Bằng chứng model yếu/mạnh</h2><div class='cards'>" + "".join(cards) + "</div>"

    # ------------------------------------------------------------------
    # Selection and charts
    # ------------------------------------------------------------------

    def _select_best_model(self) -> dict[str, Any]:
        compare = self.data.get("modern_compare")
        if compare:
            safe_models = compare.get("profiles", {}).get("safe", {}).get("models", {})
            ranked = []
            for model_name, result in safe_models.items():
                holdout = result.get("holdout", {})
                roc_auc = self._to_float(holdout.get("roc_auc"))
                if roc_auc is None:
                    continue
                ranked.append(
                    (
                        roc_auc,
                        self._to_float(holdout.get("pr_auc")) or 0.0,
                        self._to_float(holdout.get("f1")) or 0.0,
                        model_name,
                        holdout,
                    )
                )
            if ranked:
                ranked.sort(reverse=True)
                roc_auc, pr_auc, f1, model_name, holdout = ranked[0]
                return {
                    "profile": "safe",
                    "model": model_name,
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                    "f1": f1,
                    "brier_score": holdout.get("brier_score"),
                    "reason": "Chọn theo ROC-AUC holdout trên Safe A; PR-AUC và F1 dùng làm tie-breaker.",
                }

        legacy = self._legacy_metrics()
        if legacy:
            model_name, metrics = max(
                legacy.items(),
                key=lambda item: self._to_float(item[1].get("roc_auc")) or -1.0,
            )
            return {
                "profile": "legacy",
                "model": model_name,
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "f1": metrics.get("f1"),
                "brier_score": metrics.get("brier_score"),
                "reason": "Fallback: chọn theo ROC-AUC trong artifact legacy.",
            }

        return {"profile": "none", "model": None, "reason": "Chưa có kết quả model để chọn."}

    def _write_best_selection(self, selection: dict[str, Any]) -> None:
        path = self.results_dir / "best_model_selection.json"
        path.write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
        self.generated_artifacts["best_model_selection.json"] = str(path)

    def _selection_sentence(self, selection: dict[str, Any]) -> str:
        model = selection.get("model")
        if not model:
            return "Chưa đủ artifact để chọn model tốt nhất."
        return (
            f"Model được chọn là {MODEL_LABELS.get(model, model)} trên profile {selection.get('profile')} "
            f"với ROC-AUC={self._fmt(selection.get('roc_auc'))}, PR-AUC={self._fmt(selection.get('pr_auc'))}, "
            f"F1={self._fmt(selection.get('f1'))}, Brier={self._fmt(selection.get('brier_score'))}. "
            f"{selection.get('reason', '')}"
        )

    def _generate_evidence_charts(self) -> None:
        compare = self.data.get("modern_compare")
        try:
            if compare:
                self._plot_metric_evidence(compare)
                self._plot_feature_importance(compare)
            elif self._legacy_metrics():
                self._plot_legacy_metric_evidence()
        except Exception as exc:  # pragma: no cover - charts are helpful but non-critical
            logger.warning("Could not generate report evidence charts: %s", exc)

    def _plot_legacy_metric_evidence(self) -> None:
        import plotly.graph_objects as go
        import plotly.io as pio

        metrics = self._legacy_metrics()
        model_names = self._model_order(metrics.keys())
        labels = [MODEL_LABELS.get(name, name) for name in model_names]
        fig = go.Figure()
        for metric, title in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1", "F1")]:
            fig.add_trace(
                go.Bar(
                    name=title,
                    x=labels,
                    y=[metrics[name].get(metric) for name in model_names],
                )
            )
        fig.update_layout(
            title="Legacy model evidence: ROC-AUC, PR-AUC, F1",
            template="plotly_white",
            barmode="group",
            yaxis_title="Score",
            height=520,
        )
        path = self.results_dir / "model_evidence_metrics.html"
        pio.write_html(fig, str(path), full_html=True, include_plotlyjs=True)
        self.generated_artifacts["model_evidence_metrics.html"] = str(path)

    def _plot_metric_evidence(self, compare: dict[str, Any]) -> None:
        import plotly.graph_objects as go
        import plotly.io as pio

        safe_models = compare.get("profiles", {}).get("safe", {}).get("models", {})
        model_names = self._model_order(safe_models.keys())
        labels = [MODEL_LABELS.get(name, name) for name in model_names]
        fig = go.Figure()
        for metric, title in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1", "F1")]:
            fig.add_trace(
                go.Bar(
                    name=title,
                    x=labels,
                    y=[safe_models[name].get("holdout", {}).get(metric) for name in model_names],
                )
            )
        fig.update_layout(
            title="Safe A model evidence: ROC-AUC, PR-AUC, F1",
            template="plotly_white",
            barmode="group",
            yaxis_title="Score",
            height=520,
        )
        path = self.results_dir / "model_evidence_metrics.html"
        pio.write_html(fig, str(path), full_html=True, include_plotlyjs=True)
        self.generated_artifacts["model_evidence_metrics.html"] = str(path)

    def _plot_feature_importance(self, compare: dict[str, Any]) -> None:
        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.subplots import make_subplots

        models = compare.get("profiles", {}).get("safe", {}).get("models", {})
        plot_items = []
        for model_name in self._model_order(models.keys()):
            importance = models[model_name].get("feature_importance", [])[:12]
            if not importance:
                continue
            values = [
                item.get("importance", item.get("abs_coefficient", item.get("variance_importance", 0.0)))
                for item in importance
            ]
            features = [str(item.get("feature")) for item in importance]
            plot_items.append((model_name, features, values))

        if not plot_items:
            return

        fig = make_subplots(
            rows=len(plot_items),
            cols=1,
            subplot_titles=[MODEL_LABELS.get(item[0], item[0]) for item in plot_items],
            vertical_spacing=0.08,
        )
        for row_idx, (_, features, values) in enumerate(plot_items, 1):
            fig.add_trace(
                go.Bar(
                    x=list(reversed(values)),
                    y=list(reversed(features)),
                    orientation="h",
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )
        fig.update_layout(
            title="Safe A feature importance evidence",
            template="plotly_white",
            height=max(420, len(plot_items) * 360),
        )
        path = self.results_dir / "model_feature_importance_safe.html"
        pio.write_html(fig, str(path), full_html=True, include_plotlyjs=True)
        self.generated_artifacts["model_feature_importance_safe.html"] = str(path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _chart_links_html(self) -> str:
        links = []
        for filename in ("model_evidence_metrics.html", "model_feature_importance_safe.html"):
            if filename in self.generated_artifacts:
                links.append(f"<li><a href='./{filename}'>{filename}</a></li>")
        return "<h2>Biểu đồ bằng chứng</h2><ul>" + "".join(links) + "</ul>" if links else ""

    def _legacy_metrics(self) -> dict[str, dict[str, Any]]:
        legacy = self.data.get("legacy_model_comparison", {})
        basic = legacy.get("basic_metrics", {})
        if isinstance(basic, list):
            return {
                str(row.get("model")): row
                for row in basic
                if isinstance(row, dict) and row.get("model")
            }
        if isinstance(basic, dict):
            return basic

        for key in ("legacy_model_results_conservative", "legacy_model_results_full"):
            data = self.data.get(key, {})
            if data:
                return {
                    name: metrics
                    for name, metrics in data.items()
                    if isinstance(metrics, dict) and name != "metadata"
                }
        return {}

    def _model_reason(self, model_name: str, holdout: dict[str, Any], best_name: str | None) -> str:
        auc = self._to_float(holdout.get("roc_auc")) or 0.0
        brier = self._to_float(holdout.get("brier_score"))
        if model_name == "dummy":
            return "Baseline yếu: ROC-AUC quanh 0.5 nghĩa là gần như không học được tín hiệu ngoài tỷ lệ lớp."
        if model_name == "logistic":
            return "Mạnh về diễn giải và calibration; nếu thua CatBoost thì thường do quan hệ phi tuyến/tương tác chưa được bắt đủ."
        if model_name == "gam":
            if auc <= 0.6 or (brier is not None and brier > 0.25):
                return "Yếu trong lần chạy này: tín hiệu holdout/calibration thấp, nên chỉ dùng làm tham chiếu giải thích."
            return "Có giá trị giải thích phi tuyến, nhưng vẫn cần so với CatBoost trên holdout."
        if model_name == "catboost":
            if model_name == best_name:
                return "Mạnh nhất trên Safe A: xếp đầu theo ROC-AUC holdout và xử lý tốt dữ liệu bảng hỗn hợp."
            return "Mạnh cho dữ liệu bảng, nhưng không được chọn nếu holdout không đứng đầu hoặc calibration kém hơn."
        return f"ROC-AUC hiện tại là {auc:.3f}."

    def _model_order(self, names: Any) -> list[str]:
        order = ["dummy", "logistic", "gam", "catboost"]
        names_set = {str(name) for name in names}
        return [name for name in order if name in names_set] + sorted(names_set - set(order))

    def _latest_existing(self, candidates: list[Path]) -> Path | None:
        existing = [path for path in candidates if path.exists()]
        if not existing:
            return None
        return max(existing, key=lambda path: path.stat().st_mtime)

    def _read_json(self, path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _fmt(self, value: Any, digits: int = 3) -> str:
        number = self._to_float(value)
        return "N/A" if number is None else f"{number:.{digits}f}"

    def _fmt_signed(self, value: Any, digits: int = 3) -> str:
        number = self._to_float(value)
        return "N/A" if number is None else f"{number:+.{digits}f}"

    def _to_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
