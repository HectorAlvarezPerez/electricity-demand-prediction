"""Export unified resource benchmark outputs into the existing LaTeX report."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import FIGURES_DIR, METRICS_DIR, ensure_artifact_dirs


MAIN_REPORT_TEX = ROOT / "artifacts" / "reports" / "document_general_resultats_i_desenvolupament.tex"
OUT_FIG = FIGURES_DIR / "resource_benchmark_comparison.png"
OUT_INTERVAL_FIG = FIGURES_DIR / "resource_benchmark_intervals.png"
AUTO_START = "% BEGIN_RESOURCE_BENCHMARK_AUTO"
AUTO_END = "% END_RESOURCE_BENCHMARK_AUTO"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export the resource benchmark to a LaTeX fragment")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def load_payload(seed: int) -> dict:
    summary_json = METRICS_DIR / "resource_benchmark" / f"resource_benchmark_seed{seed}.json"
    if not summary_json.exists():
        raise FileNotFoundError(
            f"Missing benchmark summary: {summary_json}. Run src/run_resource_benchmark.py first."
        )
    with open(summary_json, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_rows(payload: dict) -> list[dict]:
    rows = []
    for model_name, model_payload in payload["models"].items():
        fit = model_payload["fit_metrics"]
        inference = model_payload["inference"]
        interval_metrics = model_payload.get("prediction_intervals", {}).get("metrics", {})
        source_calibrated = interval_metrics.get("source_val", {}).get("target_test", {}).get("mw", {})
        target_calibrated = interval_metrics.get("target_val", {}).get("target_test", {}).get("mw", {})
        pretty_name = {
            "xgboost": "XGBoost",
            "mlp": "MLP",
            "graphsage": "GraphSAGE",
        }.get(model_name, model_name)
        row = {
            "model_name": pretty_name,
            "source_test_mae": fit["source_test"]["mae"],
            "source_test_rmse": fit["source_test"]["rmse"],
            "target_test_mae": fit["target_test"]["mae"],
            "target_test_rmse": fit["target_test"]["rmse"],
            "target_test_mape": fit["target_test"]["mape"],
            "train_time_s": model_payload["train_time_s"],
            "peak_rss_mb": model_payload["peak_rss_mb"],
            "peak_vram_mb": model_payload.get("peak_vram_mb"),
            "model_size_mb": model_payload["model_size_mb"],
            "n_parameters": model_payload.get("n_parameters"),
            "target_inf_mean_ms": inference["target_test"]["mean_ms"],
            "target_inf_p95_ms": inference["target_test"]["p95_ms"],
            "target_throughput": inference["target_test"]["throughput_samples_s"],
            "source_calib_coverage_95": source_calibrated.get("coverage_95", float("nan")),
            "source_calib_mean_width_mw": source_calibrated.get("mean_width", float("nan")),
            "source_calib_interval_score_mw": source_calibrated.get("interval_score", float("nan")),
            "target_calib_coverage_95": target_calibrated.get("coverage_95", float("nan")),
            "target_calib_mean_width_mw": target_calibrated.get("mean_width", float("nan")),
            "target_calib_interval_score_mw": target_calibrated.get("interval_score", float("nan")),
        }
        rows.append(row)
    return rows


def write_figure(rows: list[dict]) -> None:
    labels = [r["model_name"] for r in rows]
    x = np.arange(len(labels))
    width = 0.24

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    metrics = [
        ("target_test_mae", "Target MAE", "#1f77b4"),
        ("train_time_s", "Train time (s)", "#d62728"),
        ("peak_rss_mb", "Peak RSS (MB)", "#2ca02c"),
        ("target_inf_mean_ms", "Inference mean (ms)", "#ff7f0e"),
    ]

    for ax, (key, title, color) in zip(axes.flat, metrics):
        vals = [r[key] for r in rows]
        bars = ax.bar(x, vals, color=color, width=0.6)
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        if key in {"train_time_s", "peak_rss_mb"}:
            ax.set_yscale("log")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), _fmt(val, 2), ha="center", va="bottom", fontsize=8)

    fig.suptitle("Unified resource benchmark comparison", fontsize=14, weight="bold")
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_interval_figure(rows: list[dict]) -> None:
    labels = [r["model_name"] for r in rows]
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    coverage_source = [r["source_calib_coverage_95"] * 100.0 for r in rows]
    coverage_target = [r["target_calib_coverage_95"] * 100.0 for r in rows]
    width_source = [r["source_calib_mean_width_mw"] for r in rows]
    width_target = [r["target_calib_mean_width_mw"] for r in rows]

    axes[0].bar(x - width / 2, coverage_source, width, label="Calibració source-val", color="#1f77b4")
    axes[0].bar(x + width / 2, coverage_target, width, label="Calibració target-val", color="#ff7f0e")
    axes[0].axhline(95.0, color="#333333", linestyle="--", linewidth=1.0, label="Cobertura nominal 95%")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, max(100.0, np.nanmax(coverage_source + coverage_target) * 1.08))
    axes[0].set_ylabel("Cobertura empírica (%)")
    axes[0].set_title("Cobertura sobre target test")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].bar(x - width / 2, width_source, width, label="Calibració source-val", color="#1f77b4")
    axes[1].bar(x + width / 2, width_target, width, label="Calibració target-val", color="#ff7f0e")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Amplada mitjana (MW)")
    axes[1].set_title("Amplada dels intervals 95%")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle("Conformal prediction intervals on the target test split", fontsize=13, weight="bold")
    fig.savefig(OUT_INTERVAL_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_report_block(rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append("\\paragraph{Resultats del benchmark.}")
    lines.append(
        "Les taules següents resumeixen els resultats del benchmark unificat de recursos per als tres models avaluats. "
        "La figura associada compara el cost d'entrenament, la memòria i la latència d'inferència amb l'error final sobre el test objectiu."
    )
    lines.append("\\begin{figure}[htbp]")
    lines.append("    \\centering")
    lines.append("    \\includegraphics[width=0.96\\textwidth]{../figures/resource_benchmark_comparison.png}")
    lines.append(
        "    \\caption{Comparativa del benchmark unificat de recursos per a XGBoost, MLP i GraphSAGE. "
        "S'hi mostra el compromís entre error sobre el domini objectiu, temps d'entrenament, memòria i latència d'inferència.}"
    )
    lines.append("    \\label{fig:resource_benchmark_comparison}")
    lines.append("\\end{figure}")
    lines.append("\\FloatBarrier")

    lines.append("\\paragraph{Intervals de predicció conformal.}")
    lines.append(
        "A més de la predicció puntual, el benchmark calcula intervals de predicció conformal al 95\\% "
        "a partir dels residus absoluts de validació per cada horitzó de sortida. "
        "La calibració amb \\texttt{source\\_val} conserva una lectura \\textit{zero-shot} estricta, "
        "mentre que la calibració amb \\texttt{target\\_val} utilitza validació d'Espanya i dona una estimació "
        "més operativa de la incertesa del domini objectiu, però ja no correspon a un protocol zero-shot pur."
    )
    lines.append("\\begin{figure}[htbp]")
    lines.append("    \\centering")
    lines.append("    \\includegraphics[width=0.96\\textwidth]{../figures/resource_benchmark_intervals.png}")
    lines.append(
        "    \\caption{Cobertura empírica i amplada mitjana dels intervals conformals al 95\\% sobre el test objectiu. "
        "Es comparen les calibracions amb validació font i amb validació del domini objectiu.}"
    )
    lines.append("    \\label{fig:resource_benchmark_intervals}")
    lines.append("\\end{figure}")
    lines.append("\\FloatBarrier")

    lines.append("\\begin{table}[htbp]")
    lines.append("    \\centering")
    lines.append("    \\small")
    lines.append("    \\resizebox{0.98\\textwidth}{!}{%")
    lines.append("    \\begin{tabular}{lrrrrrrr}")
    lines.append("        \\toprule")
    lines.append(
        "        Model & Target MAE & Cov. source-val & Width source-val (MW) & Score source-val & Cov. target-val & Width target-val (MW) & Score target-val \\\\"
    )
    lines.append("        \\midrule")
    for r in rows:
        lines.append(
            f"        {r['model_name']} & {_fmt(r['target_test_mae'])} & "
            f"{_fmt(r['source_calib_coverage_95'] * 100.0, 2)}\\% & {_fmt(r['source_calib_mean_width_mw'], 1)} & {_fmt(r['source_calib_interval_score_mw'], 1)} & "
            f"{_fmt(r['target_calib_coverage_95'] * 100.0, 2)}\\% & {_fmt(r['target_calib_mean_width_mw'], 1)} & {_fmt(r['target_calib_interval_score_mw'], 1)} \\\\"
        )
    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}%")
    lines.append("    }")
    lines.append(
        "    \\caption{Mètriques d'interval sobre el test objectiu. "
        "La cobertura indica el percentatge de valors reals dins l'interval i l'amplada mitjana mesura el cost d'incertesa en MW.}"
    )
    lines.append("    \\label{tab:resource_benchmark_intervals}")
    lines.append("\\end{table}")
    lines.append("\\FloatBarrier")

    lines.append("\\begin{table}[htbp]")
    lines.append("    \\centering")
    lines.append("    \\small")
    lines.append("    \\resizebox{0.98\\textwidth}{!}{%")
    lines.append("    \\begin{tabular}{lrrrrrrrr}")
    lines.append("        \\toprule")
    lines.append(
        "        Model & Target MAE & Target RMSE & Target MAPE & Train time (s) & Peak RSS (MB) & Inference mean (ms) & Throughput (samples/s) & Size (MB) \\\\"
    )
    lines.append("        \\midrule")
    for r in rows:
        lines.append(
            f"        {r['model_name']} & {_fmt(r['target_test_mae'])} & {_fmt(r['target_test_rmse'])} & {_fmt(r['target_test_mape'])} & "
            f"{_fmt(r['train_time_s'])} & {_fmt(r['peak_rss_mb'])} & {_fmt(r['target_inf_mean_ms'])} & {_fmt(r['target_throughput'])} & {_fmt(r['model_size_mb'])} \\\\"
        )
    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}%")
    lines.append("    }")
    lines.append(
        "    \\caption{Resum numèric del benchmark unificat de recursos i precisió sobre el test objectiu. "
        "Les mètriques de latència corresponen a la mitjana per batch i s'expressen en mil·lisegons.}"
    )
    lines.append("    \\label{tab:resource_benchmark_summary}")
    lines.append("\\end{table}")
    lines.append("\\FloatBarrier")

    lines.append("\\begin{table}[htbp]")
    lines.append("    \\centering")
    lines.append("    \\small")
    lines.append("    \\begin{tabular}{lccc}")
    lines.append("        \\toprule")
    lines.append("        Model & Source test MAE & Source test RMSE & Inference p95 (ms) \\\\")
    lines.append("        \\midrule")
    for r in rows:
        lines.append(
            f"        {r['model_name']} & {_fmt(r['source_test_mae'])} & {_fmt(r['source_test_rmse'])} & {_fmt(r['target_inf_p95_ms'])} \\\\")
    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append(
        "    \\caption{Mètriques addicionals per completar la lectura del benchmark. "
        "El p95 d'inferència ajuda a detectar pics de latència no visibles amb la mitjana.}"
    )
    lines.append("    \\label{tab:resource_benchmark_additional}")
    lines.append("\\end{table}")
    lines.append("\\FloatBarrier")

    return "\n".join(lines)


def update_main_report(block: str) -> None:
    if not MAIN_REPORT_TEX.exists():
        raise FileNotFoundError(f"Main report file not found: {MAIN_REPORT_TEX}")

    content = MAIN_REPORT_TEX.read_text(encoding="utf-8")
    start = content.find(AUTO_START)
    end = content.find(AUTO_END)
    if start == -1 or end == -1 or end < start:
        raise RuntimeError(
            "Could not find auto-update markers in the main report. "
            f"Expected markers: '{AUTO_START}' and '{AUTO_END}'."
        )

    replacement = f"{AUTO_START}\n{block}\n{AUTO_END}"
    new_content = content[:start] + replacement + content[end + len(AUTO_END):]
    MAIN_REPORT_TEX.write_text(new_content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_artifact_dirs()
    payload = load_payload(args.seed)
    rows = extract_rows(payload)
    write_figure(rows)
    write_interval_figure(rows)
    block = render_report_block(rows)
    update_main_report(block)
    print(f"Saved -> {OUT_FIG}")
    print(f"Saved -> {OUT_INTERVAL_FIG}")
    print(f"Updated -> {MAIN_REPORT_TEX}")


if __name__ == "__main__":
    main()
