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


def _fmt_date(value: object) -> str:
    return pd.to_datetime(value, utc=True).strftime("%d-%m-%Y")


def _fmt_optional_pm(mean: float, std: float | None = None, digits: int = 1) -> str:
    if std is None:
        return _fmt(mean, digits)
    return _fmt_pm(mean, std, digits)


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


def _load_demand_h24_predictions(seed: int) -> pd.DataFrame:
    path = METRICS_DIR / "resource_benchmark" / f"xgb_seed{seed}_prediction_intervals.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing demand prediction intervals: {path}")

    frame = pd.read_parquet(
        path,
        columns=[
            "country_code",
            "split",
            "calibration",
            "horizon",
            "forecast_timestamp",
            "y_true_mw",
            "pred_mw",
        ],
    )
    frame = frame[
        (frame["country_code"] == "ES")
        & (frame["split"] == "target_test")
        & (frame["calibration"] == "source_val")
        & (frame["horizon"] == 24)
    ].copy()
    frame = frame.rename(
        columns={
            "forecast_timestamp": "target_timestamp",
            "y_true_mw": "demand_true",
            "pred_mw": "demand_pred",
        }
    )
    return frame[["target_timestamp", "demand_true", "demand_pred"]]


def _load_renewable_total_predictions(seed: int, model_key: str) -> pd.DataFrame:
    model_name, feature_set = _split_model_key(model_key)
    path = (
        METRICS_DIR
        / "renewables_resource_benchmark"
        / f"{model_name}_{feature_set}_seed{seed}_target_test_predictions.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing renewables target-test predictions: {path}")

    frame = pd.read_parquet(
        path,
        columns=["country_code", "target_timestamp", "target", "y_true", "pred"],
    )
    frame = frame[
        (frame["country_code"] == "ES")
        & (frame["target"] == "y_renewable_total_mwh")
    ].copy()
    frame = frame.rename(columns={"y_true": "renewable_true", "pred": "renewable_pred"})
    return frame[["target_timestamp", "renewable_true", "renewable_pred"]]


def _compute_balance_metrics(demand: pd.DataFrame, renewables: pd.DataFrame) -> dict[str, float]:
    aligned = demand.merge(renewables, on="target_timestamp", how="inner")
    if aligned.empty:
        raise ValueError("No aligned timestamps between demand and renewables predictions")

    demand_true = aligned["demand_true"].to_numpy(dtype=float)
    demand_pred = aligned["demand_pred"].to_numpy(dtype=float)
    renewable_true = aligned["renewable_true"].to_numpy(dtype=float)
    renewable_pred = aligned["renewable_pred"].clip(lower=0).to_numpy(dtype=float)

    actual_residual = (demand_true - renewable_true).clip(min=0)
    predicted_residual = (demand_pred - renewable_pred).clip(min=0)
    actual_share = renewable_true / demand_true * 100.0
    predicted_share = renewable_pred / demand_pred * 100.0

    return {
        "n_samples": float(len(aligned)),
        "demand_true_gwh": float(demand_true.mean() / 1000.0),
        "renewable_true_gwh": float(renewable_true.mean() / 1000.0),
        "actual_share_pct": float(actual_share.mean()),
        "actual_residual_gwh": float(actual_residual.mean() / 1000.0),
        "demand_pred_gwh": float(demand_pred.mean() / 1000.0),
        "renewable_pred_gwh": float(renewable_pred.mean() / 1000.0),
        "predicted_share_pct": float(predicted_share.mean()),
        "predicted_residual_gwh": float(predicted_residual.mean() / 1000.0),
        "residual_mae_gwh": float(abs(actual_residual - predicted_residual).mean() / 1000.0),
        "share_mae_pct_points": float(abs(actual_share - predicted_share).mean()),
    }


def _aggregate_balance(rows: list[dict[str, float]]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    out: dict[str, float] = {}
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        out[column] = float(values.mean()) if len(values) else float("nan")
        out[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return out


def load_operational_balance_rows(seeds: list[int]) -> tuple[dict[str, float], list[dict]]:
    demand_by_seed = {seed: _load_demand_h24_predictions(seed) for seed in seeds}

    observed_seed_rows = [
        _compute_balance_metrics(demand_by_seed[seed], _load_renewable_total_predictions(seed, "xgboost_external"))
        for seed in seeds
    ]
    observed = _aggregate_balance(observed_seed_rows)

    rows: list[dict] = []
    for key in MODEL_ORDER:
        model_name, feature_set = _split_model_key(key)
        seed_rows = [
            _compute_balance_metrics(demand_by_seed[seed], _load_renewable_total_predictions(seed, key))
            for seed in seeds
        ]
        aggregate = _aggregate_balance(seed_rows)
        aggregate.update(
            {
                "key": key,
                "model_name": PRETTY_MODEL[model_name],
                "feature_set": PRETTY_FEATURE_SET[feature_set],
            }
        )
        rows.append(aggregate)
    return observed, rows


def render_report_block(
    rows: list[dict],
    dataset_stats: dict[str, object],
    operational_balance: tuple[dict[str, float], list[dict]] | None = None,
    *,
    multiseed: bool = False,
    seeds: list[int] | None = None,
) -> str:
    best_solar = _best_row(rows, "solar_mae")
    best_wind = _best_row(rows, "wind_mae")
    best_hydro = _best_row(rows, "hydro_mae")
    best_total = _best_row(rows, "total_mae")

    lines: list[str] = []
    lines.append(r"\section{Predicció de Generació Renovable}")
    lines.append(r"\subsection{Plantejament i Diferències Respecte a Demanda}")
    lines.append(
        "El segon cas d'estudi aplica el mateix marc experimental a una tasca amb una dependència física més directa. "
        "Mentre que la demanda elèctrica està molt marcada pel calendari i per la inèrcia autoregressiva, la generació renovable "
        "canvia amb les condicions atmosfèriques i amb factors operatius que poden variar d'una hora a la següent."
    )
    lines.append(
        "La tasca definida és una predicció horària a horitzó $H+24$. Per a cada país i hora $H$, el model rep informació "
        "disponible fins a aquell instant i prediu els valors agregats vint-i-quatre hores després. Els objectius són la generació "
        "solar, eòlica, hidràulica i el total renovable. Com a nota metodològica, el percentatge renovable es deriva "
        "posteriorment a partir de la producció agregada i no s'entrena com una sortida pròpia del model."
    )
    lines.append(
        "En renovables, l'objectiu no és només una prolongació autoregressiva. La solar, l'eòlica i la hidràulica responen a mecanismes "
        "físics diferents, de manera que la configuració externa incorpora un paquet meteorològic més ampli que en demanda: temperatura, "
        "humitat, precipitació, nuvolositat, radiació solar, durada d'insolació i vent a 10 i 100 metres, juntament amb agregats diaris "
        "coherents amb aquestes variables. Amb aquesta configuració es pot veure fins a quin punt la meteorologia afegeix informació "
        "sobre els desfasaments, les mitjanes mòbils i el calendari."
    )

    lines.append(r"\subsection{Dades, Variables i Protocol Experimental}")
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
        "El protocol de validació manté la filosofia de transferència de la comparativa de demanda: Espanya actua com a domini objectiu i "
        "els altres set països com a dominis font. Els models s'entrenen només sobre fonts i s'avaluen en transferència directa sobre Espanya."
    )
    lines.append(r"\begin{itemize}")
    lines.append(r"    \item \textbf{Sense variables externes:} valors actuals, desfasaments horaris, mitjanes mòbils, calendari local i país.")
    lines.append(
        r"    \item \textbf{Amb variables externes:} les mateixes variables més covariables d'Open-Meteo sobre temperatura, "
        r"humitat, precipitació, nuvolositat, radiació, insolació i vent, agregades per país i alineades amb l'hora objectiu."
    )
    lines.append(r"\end{itemize}")
    lines.append(
        "La informació externa s'alinea a l'hora objectiu $H+24$. Això equival a assumir una proxy perfecta d'una previsió meteorològica "
        "a vint-i-quatre hores vista, de manera coherent amb l'objectiu d'aïllar el valor potencial d'aquestes covariables sense introduir "
        "l'error propi d'un sistema meteorològic operatiu."
    )
    lines.append(
        "El protocol segueix el mateix criteri del cas de demanda: comparar XGBoost, MLP i GraphSAGE amb el mateix tall temporal, "
        "el mateix domini objectiu i les mateixes famílies de variables. Per separar l'efecte de la informació externa disponible, cada model "
        "s'avalua en dues configuracions: una només amb senyal temporal i autoregressiu, i una altra amb covariables meteorològiques "
        "alineades amb l'horitzó $H+24$."
    )
    if multiseed:
        seed_text = ", ".join(str(seed) for seed in seeds or [])
        lines.append(
            "L'avaluació final utilitza tres llavors "
            f"({seed_text}). Les taules reporten mitjana $\\pm$ desviació estàndard i el total renovable inclou "
            "un interval de confiança bootstrap al 95\\% sobre el MAE del conjunt de prova objectiu. Aquesta presentació recull la variabilitat "
            "entre execucions, sense plantejar una prova formal de significança entre models."
        )
    else:
        lines.append(
            "En aquesta versió d'una sola llavor, les taules mostren el resultat puntual del model i ofereixen una primera comparació "
            "del comportament relatiu entre arquitectures."
        )
    lines.append(
        "El conjunt final conté \\textbf{"
        + _fmt_int(dataset_stats["n_rows"])
        + "} mostres i \\textbf{"
        + _fmt_int(dataset_stats["n_cols"])
        + "} columnes. Les particions es defineixen segons l'instant objectiu $H+24$; per això, en validació i prova, "
        "l'entrada pot començar vint-i-quatre hores abans del tall formal del període. El rang efectiu d'objectius va del "
        + _fmt_date(dataset_stats["target_min"])
        + " al "
        + _fmt_date(dataset_stats["target_max"])
        + "."
    )

    lines.append(r"\subsection{Resultats i Interpretació}")
    lines.append(
        "La Taula~\\ref{tab:renewables_target_metrics} mostra l'error absolut mitjà sobre Espanya en escala física. "
        "A diferència de la demanda, aquí es reporten els objectius per separat perquè cadascuna de les tecnologies respon "
        "a mecanismes físics diferents i el total renovable resumeix el resultat agregat."
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
        r"    \caption{MAE sobre Espanya en la comparativa horària de renovables. Totes les columnes s'expressen en MWh.}"
    )
    lines.append(r"    \label{tab:renewables_target_metrics}")
    lines.append(r"\end{table}")
    lines.append(r"\FloatBarrier")

    lines.append(
        "Els resultats per objectiu són heterogenis. En aquest escenari horari, el millor MAE en solar correspon a "
        f"\\textbf{{{best_solar['model_name']}}} ({best_solar['feature_set'].lower()}), mentre que el millor resultat en eòlica i en total "
        f"renovable el dona \\textbf{{{best_wind['model_name']}}} amb externes. La hidràulica es manté més favorable a una formulació tabular "
        f"sense externes."
    )
    lines.append(
        "Això suggereix que l'efecte del paquet meteorològic depèn fortament de l'objectiu i del model. En XGBoost, afegir variables externes "
        f"millora la solar, l'eòlica i el total renovable; en GraphSAGE redueix lleugerament el MAE del total renovable però empitjora altres "
        f"objectius; en MLP, en canvi, les variables externes no aporten cap millora agregada: tant la solar com el total empitjoren respecte de la configuració sense externes. "
        "Per tant, les variables externes no s'han d'interpretar com un guany universal, sinó com una informació addicional que modifica la classificació en alguns objectius."
    )

    lines.append(
        "La Taula~\\ref{tab:renewables_resource_metrics} resumeix el cost computacional del mateix protocol. "
        "Totes les mesures s'han obtingut en CPU i la latència correspon al temps mitjà d'inferència per lot sobre el conjunt de prova objectiu."
    )
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(r"    \small")
    lines.append(r"    \resizebox{0.98\textwidth}{!}{%")
    lines.append(r"    \begin{tabular}{llrrrr}")
    lines.append(r"        \toprule")
    lines.append(r"        Model & Configuració & Temps entr. (s) & Pic RSS (MB) & Inferència mitjana (ms) & Mida model (MB) \\")
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
        r"    \caption{Cost computacional de la comparativa horària de renovables. La latència d'inferència és la mitjana per lot sobre el conjunt de prova objectiu.}"
    )
    lines.append(r"    \label{tab:renewables_resource_metrics}")
    lines.append(r"\end{table}")
    lines.append(r"\FloatBarrier")

    lines.append(
        "En conjunt, la complexitat arquitectònica no es tradueix automàticament en un millor resultat pràctic. "
        "XGBoost continua essent el referent global quan es mira el total renovable i el cost, perquè combina el millor MAE agregat "
        f"({best_total['feature_set'].lower()}) amb un temps d'entrenament molt inferior al de les alternatives neuronals. La MLP manté la "
        "inferència més lleugera, però no supera l'XGBoost en cap objectiu i no domina el total. GraphSAGE millora respecte a la seva "
        "variant sense externes en el total renovable i queda competitiu en alguns objectius, però ho fa amb una latència i un cost d'entrenament clarament més alts."
    )
    lines.append(
        "El cas renovable matisa el que s'havia observat en demanda. Quan l'objectiu depèn més directament de factors físics, un "
        "paquet meteorològic més ric pot modificar la classificació segons la tecnologia, però el seu efecte no és uniforme. En aquest escenari, "
        "l'XGBoost continua sent la referència més sòlida sobre el total renovable."
    )
    lines.append(
        "La MLP no treu profit de la meteorologia, ja que empitjora en afegir-la; GraphSAGE no transforma la informació meteorològica en un guany uniforme, i XGBoost manté "
        "el millor equilibri entre precisió agregada i cost. El resultat és coherent amb el criteri general del treball: una arquitectura "
        "més complexa només és preferible si el guany és prou estable per compensar l'increment de recursos."
    )

    if operational_balance is not None:
        observed_balance, balance_rows = operational_balance
        best_balance = min(balance_rows, key=lambda row: row["residual_mae_gwh"])
        xgb_external_balance = next(row for row in balance_rows if row["key"] == "xgboost_external")

        lines.append(r"\subsection{Balanç Demanda-Renovables}")
        lines.append(
            "Per connectar la predicció renovable amb la necessitat real del sistema, es creua el total renovable $H+24$ amb la "
            "predicció de demanda $H+24$ de la comparativa anterior. Com que la demanda es mesura com a potència mitjana horària, en "
            "aquest càlcul s'interpreta com energia equivalent d'una hora i es compara amb la generació renovable horària."
        )
        lines.append(r"\[")
        lines.append(r"\begin{aligned}")
        lines.append(r"\text{Cobertura renovable}_{t} &= \frac{\text{Renovable}_{t}}{\text{Demanda}_{t}} \cdot 100,\\")
        lines.append(r"\text{No renovable necessària}_{t} &= \max(\text{Demanda}_{t} - \text{Renovable}_{t}, 0).")
        lines.append(r"\end{aligned}")
        lines.append(r"\]")
        lines.append(
            r"La Taula~\ref{tab:renewables_operational_balance} no substitueix les mètriques de predicció anteriors; les tradueix a una pregunta operativa: "
            "quanta demanda quedaria per cobrir amb generació no renovable o altres recursos gestionables si es prenguessin "
            "les prediccions com a base de programació."
        )
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"    \centering")
        lines.append(r"    \small")
        lines.append(r"    \resizebox{0.98\textwidth}{!}{%")
        lines.append(r"    \begin{tabular}{llrrrrr}")
        lines.append(r"        \toprule")
        lines.append(
            r"        Escenari & Configuració & Demanda & Renovable & Cobertura renovable & No renovable necessària & MAE no renov. \\"
        )
        lines.append(r"        \midrule")
        lines.append(
            "        Observat & Test ES & "
            + _fmt(observed_balance["demand_true_gwh"], 2)
            + " & "
            + _fmt(observed_balance["renewable_true_gwh"], 2)
            + " & "
            + _fmt(observed_balance["actual_share_pct"], 1)
            + r"\% & "
            + _fmt(observed_balance["actual_residual_gwh"], 2)
            + r" & -- \\"
        )
        for row in balance_rows:
            if multiseed:
                demand = _fmt_optional_pm(row["demand_pred_gwh"], row["demand_pred_gwh_std"], 2)
                renewable = _fmt_optional_pm(row["renewable_pred_gwh"], row["renewable_pred_gwh_std"], 2)
                share = _fmt_optional_pm(row["predicted_share_pct"], row["predicted_share_pct_std"], 1)
                residual = _fmt_optional_pm(row["predicted_residual_gwh"], row["predicted_residual_gwh_std"], 2)
                residual_mae = _fmt_optional_pm(row["residual_mae_gwh"], row["residual_mae_gwh_std"], 2)
            else:
                demand = _fmt(row["demand_pred_gwh"], 2)
                renewable = _fmt(row["renewable_pred_gwh"], 2)
                share = _fmt(row["predicted_share_pct"], 1)
                residual = _fmt(row["predicted_residual_gwh"], 2)
                residual_mae = _fmt(row["residual_mae_gwh"], 2)
            lines.append(
                "        "
                + row["model_name"]
                + " & "
                + row["feature_set"]
                + " & "
                + demand
                + " & "
                + renewable
                + " & "
                + share
                + r"\% & "
                + residual
                + " & "
                + residual_mae
                + r" \\"
            )
        lines.append(r"        \bottomrule")
        lines.append(r"    \end{tabular}%")
        lines.append(r"    }")
        lines.append(
            r"    \caption{Balanç operatiu entre demanda horària i generació renovable prevista sobre Espanya. "
            r"Demanda, renovable, generació no renovable necessària i el seu MAE s'expressen en GWh equivalents per hora.}"
        )
        lines.append(r"    \label{tab:renewables_operational_balance}")
        lines.append(r"\end{table}")
        lines.append(r"\FloatBarrier")
        lines.append(
            "En les dades observades del test, la generació renovable cobreix de mitjana el "
            + _fmt(observed_balance["actual_share_pct"], 1)
            + "\\% de la demanda espanyola, deixant una generació no renovable necessària mitjana de "
            + _fmt(observed_balance["actual_residual_gwh"], 2)
            + " GWh per hora. Amb la combinació XGBoost de demanda i XGBoost amb externes per renovables, la cobertura prevista és del "
            + _fmt(xgb_external_balance["predicted_share_pct"], 1)
            + "\\% i la generació no renovable necessària prevista és de "
            + _fmt(xgb_external_balance["predicted_residual_gwh"], 2)
            + " GWh per hora. El menor error en aquest residual correspon a \\textbf{"
            + best_balance["model_name"]
            + "} ("
            + best_balance["feature_set"].lower()
            + "), amb un MAE de "
            + _fmt(best_balance["residual_mae_gwh"], 2)
            + " GWh."
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
        seeds = payload.get("seeds") or []
    else:
        payload = load_payload(args.seed)
        rows = extract_rows(payload)
        seeds = [args.seed]
    dataset_stats = load_dataset_stats()
    operational_balance = load_operational_balance_rows([int(seed) for seed in seeds])
    block = render_report_block(
        rows,
        dataset_stats,
        operational_balance=operational_balance,
        multiseed=args.multiseed,
        seeds=seeds,
    )
    update_main_report(block)
    print(f"Updated -> {MAIN_REPORT_TEX}")


if __name__ == "__main__":
    main()
