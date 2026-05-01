"""Export the hourly renewables benchmark into the existing LaTeX report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import METRICS_DIR


MAIN_REPORT_TEX = ROOT / "artifacts" / "reports" / "document_general_resultats_i_desenvolupament.tex"
DATA_DIR = ROOT / "data" / "processed_renewables_hourly"
AUTO_START = "% BEGIN_RENEWABLES_BENCHMARK_AUTO"
AUTO_END = "% END_RENEWABLES_BENCHMARK_AUTO"

MODEL_ORDER = [
    "xgboost_no_external",
    "mlp_no_external",
    "graphsage_no_external",
    "xgboost_external",
    "mlp_external",
    "graphsage_external",
]
PRETTY_MODEL = {
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "graphsage": "GraphSAGE",
}
PRETTY_FEATURE_SET = {
    "no_external": "Sense externes",
    "external": "Amb externes",
}
TARGET_ORDER = [
    ("y_solar_mwh", "Solar"),
    ("y_wind_mwh", "Eòlica"),
    ("y_hydro_mwh", "Hidro"),
    ("y_renewable_total_mwh", "Total renovable"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export the renewables benchmark to a LaTeX fragment")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--multiseed", action="store_true", help="Use renewables_resource_benchmark_multiseed.json if available")
    return p.parse_args()


def _fmt(value: float, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}"


def _fmt_pm(mean: float, std: float, digits: int = 1) -> str:
    return f"{_fmt(mean, digits)} $\\pm$ {_fmt(std, digits)}"


def _fmt_ci(ci: dict | None, digits: int = 1) -> str:
    if not ci:
        return "--"
    return f"[{_fmt(ci.get('ci_low'), digits)}, {_fmt(ci.get('ci_high'), digits)}]"


def _fmt_int(value: int) -> str:
    return f"{int(value):,}".replace(",", ".")


def load_payload(seed: int) -> dict:
    summary_json = METRICS_DIR / "renewables_resource_benchmark" / f"renewables_resource_benchmark_seed{seed}.json"
    if not summary_json.exists():
        raise FileNotFoundError(
            f"Missing renewables benchmark summary: {summary_json}. Run src/run_renewables_benchmark.py first."
        )
    return json.loads(summary_json.read_text(encoding="utf-8"))


def load_multiseed_payload() -> dict:
    summary_json = METRICS_DIR / "renewables_resource_benchmark" / "renewables_resource_benchmark_multiseed.json"
    if not summary_json.exists():
        raise FileNotFoundError(
            f"Missing multi-seed renewables benchmark summary: {summary_json}. Run src/run_renewables_benchmark.py --seeds ... first."
        )
    return json.loads(summary_json.read_text(encoding="utf-8"))


def load_dataset_stats() -> dict[str, object]:
    frames = {
        split: pd.read_parquet(DATA_DIR / f"{split}.parquet")
        for split in ["train", "val", "test"]
    }
    full = pd.concat(frames.values(), ignore_index=True)
    return {
        "n_rows": len(full),
        "n_cols": len(full.columns),
        "input_min": pd.to_datetime(full["utc_timestamp"], utc=True).min(),
        "input_max": pd.to_datetime(full["utc_timestamp"], utc=True).max(),
        "target_min": pd.to_datetime(full["target_timestamp"], utc=True).min(),
        "target_max": pd.to_datetime(full["target_timestamp"], utc=True).max(),
    }


def extract_rows(payload: dict) -> list[dict]:
    rows: list[dict] = []
    for key in MODEL_ORDER:
        model_payload = payload["models"][key]
        if key.endswith("_no_external"):
            model_name = key[: -len("_no_external")]
            feature_set = "no_external"
        elif key.endswith("_external"):
            model_name = key[: -len("_external")]
            feature_set = "external"
        else:
            raise ValueError(f"Unexpected renewables model key: {key}")
        target_metrics = model_payload["fit_metrics_by_target_raw"]["target_test"]
        inference = model_payload["inference"]["target_test"]
        rows.append(
            {
                "key": key,
                "model_name": PRETTY_MODEL[model_name],
                "feature_set": PRETTY_FEATURE_SET[feature_set],
                "solar_mae": target_metrics["y_solar_mwh"]["mae"],
                "wind_mae": target_metrics["y_wind_mwh"]["mae"],
                "hydro_mae": target_metrics["y_hydro_mwh"]["mae"],
                "total_mae": target_metrics["y_renewable_total_mwh"]["mae"],
                "train_time_s": model_payload["train_time_s"],
                "peak_rss_mb": model_payload["peak_rss_mb"],
                "inference_mean_ms": inference["mean_ms"],
                "model_size_mb": model_payload["model_size_mb"],
            }
        )
    return rows


def _split_model_key(key: str) -> tuple[str, str]:
    if key.endswith("_no_external"):
        return key[: -len("_no_external")], "no_external"
    if key.endswith("_external"):
        return key[: -len("_external")], "external"
    raise ValueError(f"Unexpected renewables model key: {key}")


def extract_multiseed_rows(payload: dict) -> list[dict]:
    aggregate_by_key = {
        f"{row['model_name']}_{row['feature_set']}": row
        for row in payload["aggregate"]
    }
    bootstrap = payload.get("bootstrap_target_mae", {})
    rows: list[dict] = []
    for key in MODEL_ORDER:
        model_name, feature_set = _split_model_key(key)
        row = aggregate_by_key[key]
        rows.append(
            {
                "key": key,
                "model_name": PRETTY_MODEL[model_name],
                "feature_set": PRETTY_FEATURE_SET[feature_set],
                "n_seeds": row["n_seeds"],
                "solar_mae": row["target_raw_target_test_y_solar_mwh_mae_mean"],
                "solar_mae_std": row["target_raw_target_test_y_solar_mwh_mae_std"],
                "wind_mae": row["target_raw_target_test_y_wind_mwh_mae_mean"],
                "wind_mae_std": row["target_raw_target_test_y_wind_mwh_mae_std"],
                "hydro_mae": row["target_raw_target_test_y_hydro_mwh_mae_mean"],
                "hydro_mae_std": row["target_raw_target_test_y_hydro_mwh_mae_std"],
                "total_mae": row["target_raw_target_test_y_renewable_total_mwh_mae_mean"],
                "total_mae_std": row["target_raw_target_test_y_renewable_total_mwh_mae_std"],
                "train_time_s": row["train_time_s_mean"],
                "train_time_s_std": row["train_time_s_std"],
                "peak_rss_mb": row["peak_rss_mb_mean"],
                "peak_rss_mb_std": row["peak_rss_mb_std"],
                "inference_mean_ms": row["inference_target_test_mean_ms_mean"],
                "inference_mean_ms_std": row["inference_target_test_mean_ms_std"],
                "model_size_mb": row["model_size_mb_mean"],
                "model_size_mb_std": row["model_size_mb_std"],
                "total_mae_ci": bootstrap.get(key, {}).get("y_renewable_total_mwh"),
            }
        )
    return rows


def _best_row(rows: list[dict], metric: str) -> dict:
    return min(rows, key=lambda row: row[metric])


def _delta_text(rows: list[dict], model_name: str, metric: str) -> str:
    no_external = next(row for row in rows if row["key"] == f"{model_name}_no_external")
    external = next(row for row in rows if row["key"] == f"{model_name}_external")
    delta = external[metric] - no_external[metric]
    if delta < 0:
        return f"redueix {metric} en {abs(delta):.1f}"
    if delta > 0:
        return f"empitjora {metric} en {delta:.1f}"
    return f"manté {metric} sense canvis apreciables"


def render_report_block(
    rows: list[dict],
    dataset_stats: dict[str, object],
    *,
    multiseed: bool = False,
    seeds: list[int] | None = None,
) -> str:
    best_solar = _best_row(rows, "solar_mae")
    best_wind = _best_row(rows, "wind_mae")
    best_hydro = _best_row(rows, "hydro_mae")
    best_total = _best_row(rows, "total_mae")

    lines: list[str] = []
    lines.append(r"\section{Generació Renovable Horària}")
    lines.append(
        "Aquest segon cas d'estudi reutilitza el mateix marc experimental en una tasca on el senyal físic és més immediat. "
        "La demanda elèctrica depèn fortament de calendari i inèrcia autoregressiva; la generació renovable, en canvi, "
        "respon a patrons atmosfèrics i operatius que poden canviar d'una hora a la següent. Per això aquest bloc serveix "
        "com a contrast natural del cas de demanda."
    )

    lines.append(r"\subsection{Objectiu de l'experiment}")
    lines.append(
        "La tasca definida és una predicció horària a horitzó $H+1$. Per a cada país i hora $H$, el model rep informació "
        "disponible fins a aquell instant i prediu els valors agregats de la següent hora. Els objectius són:"
    )
    lines.append(r"\[")
    lines.append(r"\texttt{solar\_mwh},\quad")
    lines.append(r"\texttt{wind\_mwh},\quad")
    lines.append(r"\texttt{hydro\_mwh},\quad")
    lines.append(r"\texttt{renewable\_total\_mwh}.")
    lines.append(r"\]")

    lines.append(r"\subsection{Metodologia i construcció del dataset}")
    lines.append(
        "Les dades de generació provenen d'ENTSO-E, concretament de les sèries d'\\textit{Actual Generation per Production Type}. "
        "Cada registre horari o subhorari de potència es transforma primer a energia amb la durada real de l'interval, i "
        "després s'agrega a resolució horària:"
    )
    lines.append(r"\[")
    lines.append(r"\text{MWh}_{h} = \sum_{t \in h} \text{MW}_{t} \cdot \Delta t_{\text{hores}}.")
    lines.append(r"\]")
    lines.append(
        "Les tecnologies s'agrupen en categories interpretables. La solar correspon a \\texttt{Solar}, l'eòlica agrega "
        "\\texttt{Wind Onshore} i \\texttt{Wind Offshore}, i la hidràulica combina \\texttt{Hydro Run-of-river and poundage} "
        "i \\texttt{Hydro Water Reservoir}. El total renovable suma aquestes categories amb biomassa, geotèrmia, marina i altres "
        "renovables. S'exclou explícitament \\texttt{Hydro Pumped Storage} perquè representa emmagatzematge i no generació renovable primària."
    )
    lines.append(
        "El protocol de validació manté la filosofia de transferència del benchmark de demanda: Espanya actua com a domini objectiu i "
        "els altres set països com a dominis font. Els models s'entrenen només sobre fonts i s'avaluen en \\textit{zero-shot} sobre Espanya."
    )

    lines.append(r"\subsection{Diferències respecte al cas de demanda}")
    lines.append(
        "En renovables, el target no és només una prolongació autoregressiva. La solar depèn del perfil diürn i de la temperatura; "
        "l'eòlica respon més directament a variacions de vent; la hidràulica conserva una component d'inèrcia més gran. Això canvia la "
        "lectura del benchmark: aquí no n'hi ha prou amb veure si un model aprèn lags, sinó si també sap aprofitar covariables físiques "
        "quan realment afegeixen senyal útil."
    )

    lines.append(r"\subsection{Variables externes i hipòtesi meteorològica}")
    lines.append(r"\begin{itemize}")
    lines.append(r"    \item \textbf{Sense variables externes:} valors actuals, lags horaris, mitjanes mòbils, calendari local i país.")
    lines.append(
        r"    \item \textbf{Amb variables externes:} les mateixes variables més la temperatura horària i els agregats diaris "
        r"de temperatura del pipeline d'Open-Meteo."
    )
    lines.append(r"\end{itemize}")
    lines.append(
        "La informació externa s'alinea a l'hora objectiu $H+1$. Això equival a assumir una proxy perfecta d'una previsió meteorològica "
        "a una hora vista, de manera coherent amb l'objectiu d'aïllar el valor potencial de les covariables exògenes sense introduir "
        "l'error propi d'un sistema meteorològic operatiu."
    )

    lines.append(r"\subsection{Execució reproduïble}")
    lines.append(r"La construcció del dataset i el benchmark complet executat per a aquesta secció han estat:")
    lines.append(r"\begin{verbatim}")
    lines.append("python src/data/download_weather.py")
    lines.append("python src/data/preprocess_renewables.py --include_external")
    lines.append("python src/run_renewables_benchmark.py \\")
    lines.append("  --seeds 42 123 2024 \\")
    lines.append("  --models xgboost mlp graphsage \\")
    lines.append("  --feature_sets no_external external \\")
    lines.append("  --xgb_estimators 100 \\")
    lines.append("  --xgb_n_jobs 4 \\")
    lines.append("  --torch_epochs 300 \\")
    lines.append("  --torch_patience 20 \\")
    lines.append("  --batch_size 256 \\")
    lines.append("  --log_every 10")
    lines.append(r"\end{verbatim}")
    if multiseed:
        seed_text = ", ".join(str(seed) for seed in seeds or [])
        lines.append(
            "Per reforçar la robustesa estadística, el benchmark final s'ha executat amb tres llavors "
            f"(\\texttt{{{seed_text}}}). Les taules reporten mitjana $\\pm$ desviació estàndard i el total renovable inclou "
            "un interval de confiança bootstrap al 95\\% sobre el MAE del test objectiu. Aquesta anàlisi quantifica variabilitat "
            "experimental, però no s'interpreta com una comparació estadística exhaustiva de significança entre models."
        )
    lines.append(r"L'execució genera els fitxers:")
    lines.append(r"\begin{itemize}")
    lines.append(r"    \item \path{data/processed_renewables_hourly/train.parquet}")
    lines.append(r"    \item \path{data/processed_renewables_hourly/val.parquet}")
    lines.append(r"    \item \path{data/processed_renewables_hourly/test.parquet}")
    lines.append(r"    \item \path{artifacts/metrics/renewables_resource_benchmark/}: resum CSV i JSON de l'execució.")
    lines.append(r"\end{itemize}")
    lines.append(
        "El dataset resultant conté \\textbf{"
        + _fmt_int(dataset_stats["n_rows"])
        + "} mostres i \\textbf{"
        + _fmt_int(dataset_stats["n_cols"])
        + "} columnes. El rang temporal efectiu va de "
        + str(dataset_stats["input_min"])
        + " a "
        + str(dataset_stats["input_max"])
        + " com a timestamp d'entrada, amb objectius entre "
        + str(dataset_stats["target_min"])
        + " i "
        + str(dataset_stats["target_max"])
        + "."
    )

    lines.append(r"\subsection{Resultats}")
    lines.append(
        "La Taula~\\ref{tab:renewables_target_metrics} mostra l'error absolut mitjà sobre Espanya en escala física. "
        "A diferència de la demanda, aquí es reporten els objectius per separat perquè cadascuna de les tecnologies respon "
        "a mecanismes físics diferents i el total renovable resumeix el compromís agregat."
    )
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(r"    \small")
    lines.append(r"    \resizebox{0.98\textwidth}{!}{%")
    lines.append(r"    \begin{tabular}{llrrrrr}")
    lines.append(r"        \toprule")
    lines.append(r"        Model & Configuració & Solar MAE & Eòlica MAE & Hidro MAE & Total renovable MAE & Total MAE IC95 \\")
    lines.append(r"        \midrule")
    for row in rows:
        if multiseed:
            solar = _fmt_pm(row["solar_mae"], row["solar_mae_std"])
            wind = _fmt_pm(row["wind_mae"], row["wind_mae_std"])
            hydro = _fmt_pm(row["hydro_mae"], row["hydro_mae_std"])
            total = _fmt_pm(row["total_mae"], row["total_mae_std"])
            total_ci = _fmt_ci(row.get("total_mae_ci"))
        else:
            solar = _fmt(row["solar_mae"])
            wind = _fmt(row["wind_mae"])
            hydro = _fmt(row["hydro_mae"])
            total = _fmt(row["total_mae"])
            total_ci = "--"
        lines.append(
            "        "
            + row["model_name"]
            + " & "
            + row["feature_set"]
            + " & "
            + solar
            + " & "
            + wind
            + " & "
            + hydro
            + " & "
            + total
            + " & "
            + total_ci
            + r" \\"
        )
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}%")
    lines.append(r"    }")
    lines.append(
        r"    \caption{MAE sobre Espanya en el benchmark horari de renovables. Totes les columnes s'expressen en MWh.}"
    )
    lines.append(r"    \label{tab:renewables_target_metrics}")
    lines.append(r"\end{table}")
    lines.append(r"\FloatBarrier")

    lines.append(
        "La lectura per targets és menys uniforme que en la versió diària antiga. En aquesta execució, el millor MAE en solar correspon a "
        f"\\textbf{{{best_solar['model_name']}}} ({best_solar['feature_set'].lower()}), mentre que el millor resultat en eòlica i en total "
        f"renovable el dona \\textbf{{{best_wind['model_name']}}} amb externes. La hidràulica es manté més favorable a una formulació tabular "
        f"sense externes."
    )
    lines.append(
        "Això suggereix que l'efecte de les covariables externes depèn fortament del target. En XGBoost, afegir temperatura i estadístics "
        f"diaris millora sobretot l'eòlica i el total renovable; en GraphSAGE també redueix el MAE del total renovable de manera visible; "
        f"en MLP, en canvi, l'ajuda és més clara en eòlica i hidràulica que no pas en el total agregat."
    )

    lines.append(
        "La Taula~\\ref{tab:renewables_resource_metrics} resumeix el cost computacional de la mateixa execució. "
        "Totes les mesures s'han obtingut en CPU i la latència correspon al temps mitjà d'inferència per batch sobre el test objectiu."
    )
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(r"    \small")
    lines.append(r"    \resizebox{0.98\textwidth}{!}{%")
    lines.append(r"    \begin{tabular}{llrrrr}")
    lines.append(r"        \toprule")
    lines.append(r"        Model & Configuració & Train time (s) & Peak RSS (MB) & Inference mean (ms) & Model size (MB) \\")
    lines.append(r"        \midrule")
    for row in rows:
        if multiseed:
            train_time = _fmt_pm(row["train_time_s"], row["train_time_s_std"], 2)
            peak_rss = _fmt_pm(row["peak_rss_mb"], row["peak_rss_mb_std"], 1)
            inference = _fmt_pm(row["inference_mean_ms"], row["inference_mean_ms_std"], 3)
            size = _fmt_pm(row["model_size_mb"], row["model_size_mb_std"], 3)
        else:
            train_time = _fmt(row["train_time_s"], 2)
            peak_rss = _fmt(row["peak_rss_mb"], 1)
            inference = _fmt(row["inference_mean_ms"], 3)
            size = _fmt(row["model_size_mb"], 3)
        lines.append(
            "        "
            + row["model_name"]
            + " & "
            + row["feature_set"]
            + " & "
            + train_time
            + " & "
            + peak_rss
            + " & "
            + inference
            + " & "
            + size
            + r" \\"
        )
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}%")
    lines.append(r"    }")
    lines.append(
        r"    \caption{Cost computacional del benchmark horari de renovables. La latència d'inferència és la mitjana per batch sobre el test objectiu.}"
    )
    lines.append(r"    \label{tab:renewables_resource_metrics}")
    lines.append(r"\end{table}")
    lines.append(r"\FloatBarrier")

    lines.append(
        "La lectura conjunta reforça la tesi central del treball: la complexitat arquitectònica no garanteix el millor compromís final. "
        "XGBoost continua essent el millor referent global quan es mira el total renovable i el cost, perquè combina el millor MAE agregat "
        f"({best_total['feature_set'].lower()}) amb un temps d'entrenament molt inferior al de les alternatives neuronals. La MLP manté la "
        "inferència més lleugera i excel·leix en solar, però no domina el total. GraphSAGE millora respecte a la seva "
        "variant sense externes i queda competitiu en alguns targets, però ho fa amb una latència i un cost d'entrenament clarament més alts."
    )

    lines.append(r"\subsection{Lectura metodològica del bloc renovable}")
    lines.append(
        "El cas renovable horari matisa la conclusió obtinguda en demanda. Quan el target depèn més directament de factors físics, les "
        "variables externes deixen de ser un refinament marginal i poden modificar el rànquing segons la tecnologia. Tanmateix, aquesta "
        "dependència no fa desaparèixer la necessitat d'un baseline tabular fort: l'XGBoost continua sent la millor referència sobre el total renovable."
    )
    lines.append(
        "La lectura també ajuda a entendre el paper dels models. La MLP aprofita bé un problema més local i més instantani com la solar, "
        "GraphSAGE guanya robustesa en el total quan rep externes, i XGBoost manté el millor equilibri entre precisió agregada i cost. "
        "Això encaixa amb el fil conductor del TFG: abans de justificar una arquitectura més complexa, cal comprovar si el guany és estable "
        "i prou gran per compensar l'increment de recursos."
    )

    return "\n".join(lines)


def update_main_report(block: str) -> None:
    content = MAIN_REPORT_TEX.read_text(encoding="utf-8")
    start = content.find(AUTO_START)
    end = content.find(AUTO_END)
    if start == -1 or end == -1 or end < start:
        raise RuntimeError(
            "Could not find auto-update markers in the main report. "
            f"Expected markers: '{AUTO_START}' and '{AUTO_END}'."
        )
    replacement = f"{AUTO_START}\n{block}\n{AUTO_END}"
    MAIN_REPORT_TEX.write_text(content[:start] + replacement + content[end + len(AUTO_END):], encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.multiseed:
        payload = load_multiseed_payload()
        rows = extract_multiseed_rows(payload)
    else:
        payload = load_payload(args.seed)
        rows = extract_rows(payload)
    dataset_stats = load_dataset_stats()
    block = render_report_block(rows, dataset_stats, multiseed=args.multiseed, seeds=payload.get("seeds"))
    update_main_report(block)
    print(f"Updated -> {MAIN_REPORT_TEX}")


if __name__ == "__main__":
    main()
