# Predicció de demanda elèctrica i generació renovable

Repositori del TFG sobre predicció horària de demanda elèctrica i generació renovable, amb `ES` com a domini objectiu operatiu en la pipeline actual.

Títol de treball:
`Prediccio de Demanda Electrica i Generacio Renovable amb Comparativa de Models`.

## Objectiu

Aquest projecte reconstrueix la narrativa per demostrar experimentalment quan *no compensa usar una xarxa neuronal*, comparant `XGBoost` davant d'una única GNN simple i reproduïble en els eixos de: precisió, cost computacional i latència.

S'estructura en dos benchmarks paral·lels:
1. **Benchmark de Demanda**: Previsió de demanda horària.
2. **Benchmark de Renovables**: Previsió de generació (solar, eòlica, hidro, etc).

Per a cadascun d'ells s'avalua el millor baseline tabular (`XGBoost`) al costat d'una `MLP` tabular i d'un baseline senzill d'aprenentatge sobre grafs (`GraphSAGE`), incloent-hi variables meteorològiques quan el protocol ho requereix.

## Estat actual

Actualment hi ha implementat:

- descàrrega de demanda i generació des d’ENTSO-E,
- descàrrega de temperatura històrica i meteorologia enriquida des d’Open-Meteo,
- preprocessat a format llarg amb lags explícits i horitzó de 24 hores (demanda),
- preprocessat horari de generació renovable amb targets `solar_mwh`, `wind_mwh`, `hydro_mwh` i `renewable_total_mwh`,
- splits temporals `train` / `val` / `test`,
- baselines tabulars `Daily Naive`, `Ridge` i `XGBoost` (demanda),
- baseline neuronal `MLP` (demanda),
- benchmark renovable horari amb `XGBoost`, `MLP` i `GraphSAGE`, amb i sense externes,
- fine-tuning few-shot per a `MLP` i `XGBoost`,
- scripts de visualització i comparació de resultats.

Propers passos possibles:

- afegir figures específiques del benchmark renovable,
- estudiar fonts meteorològiques de forecast històric per substituir la proxy perfecta.

Benchmark unificat de recursos per demanda:

```bash
.venv/bin/python src/run_resource_benchmark.py --seed 42 --xgb_n_jobs 4
```

Aquest flux genera comparatives de `XGBoost`, `MLP` i `GraphSAGE` sobre les mateixes particions i el mateix conjunt de features, guardant resultats detallats a `artifacts/metrics/resource_benchmark/`.

El mateix runner actualitza directament la secció corresponent dins del document existent i genera la figura resum. Després només cal recompilar el PDF:

```bash
cd artifacts/reports && pdflatex document_general_resultats_i_desenvolupament.tex
```

## Estructura del repositori

```text
.
├── data/
│   ├── raw/
│   └── processed_long/
├── artifacts/
│   ├── figures/
│   ├── metrics/
│   └── models/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── benchmarking/
│   ├── data/
│   ├── models/
│   ├── visualization/
│   ├── run_resource_benchmark.py
│   ├── run_renewables_benchmark.py
│   ├── run_graph_topk_sweep.py
│   ├── run_finetune_sweep.py
│   └── run_finetune_sweep_xgb.py
├── tests/
├── requirements.txt
└── pyproject.toml
```

## Dominis i dades

Dominis font:

- `BE`
- `DE`
- `FR`
- `GR`
- `IT`
- `NL`
- `PT`

Domini objectiu actual:

- `ES`

El codi d'entrenament i avaluació treballa amb `ES` com a domini objectiu.

## Dataset processat

La pipeline principal genera un dataset en format llarg, una fila per `(utc_timestamp, country_code)`, amb:

- demanda actual,
- lags autoregressius `lag_1` a `lag_24`, més `lag_48` i `lag_168`,
- variables temporals cícliques en hora local,
- variables meteorològiques si estan disponibles,
- horitzó multisalida de 24 passos `y_h1 ... y_h24`,
- codificació one-hot del país.

Splits temporals definits al codi:

- `train`: 2015-01-01 a 2022-12-31
- `val`: 2023-01-01 a 2023-12-31
- `test`: des de 2024-01-01

## Requisits

Versió recomanada:

- Python 3.11 o superior

Instal·lació:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

El fitxer `requirements.txt` ja inclou les dependències necessàries per a:

- entrenament amb `PyTorch`,
- baselines amb `scikit-learn` i `XGBoost`,
- lectura i escriptura de fitxers parquet,
- tracking d’experiments amb `MLflow`.

## Variables d’entorn

Crea un fitxer `.env` a l’arrel del repositori amb les credencials necessàries:

```env
ENTSOE_TOKEN_KEY=el_teu_token_entsoe
```

## Flux recomanat

### 1. Descarregar demanda europea

```bash
python src/data/download_entsoe.py
```

Sortides:

- `data/raw/europe/demand/entsoe_demand_{country}.csv`
- `data/raw/europe/generation/entsoe_generation_{country}.csv`

### 2. Descarregar meteorologia històrica

```bash
python src/data/download_weather.py
```

Sortida:

- `data/raw/weather/weather_{country}.csv`

### 3. Construir el dataset processat

```bash
python src/data/preprocess.py
```

Sortides:

- `data/processed_long/train.parquet`
- `data/processed_long/val.parquet`
- `data/processed_long/test.parquet`

### 3b. Benchmark horari de renovables

Objectiu:

- Predir generació renovable horària a horitzó `H+1`.
- Mantenir el mateix protocol de transferència que demanda: entrenament en països font i avaluació `zero-shot` sobre Espanya (`ES`).
- Comparar `XGBoost`, `MLP` i `GraphSAGE` amb i sense variables meteorològiques externes.

Targets:

- `solar_mwh`
- `wind_mwh`
- `hydro_mwh`
- `renewable_total_mwh`

Metodologia:

- ENTSO-E aporta la generació per tecnologia.
- La potència se transforma a energia per interval amb `MWh = MW * durada_interval_h` i després s'agrega a resolució horària.
- `wind_mwh` agrega eòlica terrestre i marina.
- `hydro_mwh` agrega hidràulica fluvial i d'embassament.
- `renewable_total_mwh` exclou bombeig (`Hydro Pumped Storage`) i residus (`Waste`).
- Les variables externes provenen d'Open-Meteo, agregant diverses ciutats representatives per país.
- La meteorologia de l'hora objectiu `H+1` s'usa com a proxy perfecta d'una previsió a una hora vista.

Sense variables externes:

```bash
python src/data/preprocess_renewables.py
```

Amb meteorologia horària d'Open-Meteo com a proxy perfecta a `H+1`:

```bash
python src/data/download_weather.py
python src/data/preprocess_renewables.py --include_external
```

Sortides:

- `data/processed_renewables_hourly/train.parquet`
- `data/processed_renewables_hourly/val.parquet`
- `data/processed_renewables_hourly/test.parquet`

Execució completa del benchmark:

```bash
python src/run_renewables_benchmark.py \
  --models xgboost mlp graphsage \
  --feature_sets no_external external \
  --xgb_estimators 100 \
  --torch_epochs 100 \
  --torch_patience 10 \
  --batch_size 256 \
  --log_every 10
```

Prova curta de funcionament:

```bash
python src/run_renewables_benchmark.py \
  --models xgboost \
  --feature_sets no_external \
  --xgb_estimators 2
```

Resultats:

- `artifacts/metrics/renewables_resource_benchmark/renewables_resource_benchmark_seed42.csv`
- `artifacts/metrics/renewables_resource_benchmark/renewables_resource_benchmark_seed42.json`
- models entrenats sota `artifacts/models/renewables_*`

Les mètriques principals són:

- MAE/RMSE/MAPE normalitzats i en escala original.
- MAE per target (`solar`, `wind`, `hydro`, `renewable_total`).
- temps d'entrenament, memòria, mida de model, latència i throughput.

### 3c. Sensibilitat de la topologia GraphSAGE

Per comparar diverses densitats del graf de demanda:

```bash
python src/run_graph_topk_sweep.py \
  --top_k_values 1 2 3 5 7 \
  --seeds 42 123 2024 \
  --epochs 500 \
  --patience 20 \
  --output_dir artifacts/metrics/graph_topk_sweep \
  --skip_existing
```

Sortides:

- `artifacts/metrics/graph_topk_sweep/graph_topk_sweep_rows.csv`
- `artifacts/metrics/graph_topk_sweep/graph_topk_sweep_aggregate.csv`
- `artifacts/metrics/graph_topk_sweep/graph_topk_sweep_summary.json`

### 4. Entrenar baselines clàssics

```bash
python src/models/baselines.py
```

Genera:

- `artifacts/models/baseline_ridge.joblib`
- `artifacts/models/baseline_xgb.json`
- `artifacts/models/baseline_xgb_features.json`
- `artifacts/metrics/baseline_metrics.json`

### 5. Entrenar el baseline MLP

```bash
python src/train.py --seed 42 --epochs 30
```

Genera:

- `artifacts/models/mlp_tabular_long_seed42.pt`
- `artifacts/metrics/mlp_metrics_seed42.json`

### 6. Fine-tuning few-shot del MLP

```bash
python src/train_finetune.py \
  --pretrained_model artifacts/models/mlp_tabular_long_seed42.pt \
  --target_fraction 0.05 \
  --epochs 15
```

### 7. Sweep few-shot del MLP

```bash
python src/run_finetune_sweep.py \
  --pretrained_model artifacts/models/mlp_tabular_long_seed42.pt \
  --fractions 0.01 0.02 0.05 0.10
```

### 8. Fine-tuning few-shot de XGBoost

```bash
python src/train_finetune_xgb.py \
  --pretrained_model artifacts/models/baseline_xgb.json \
  --metadata_path artifacts/models/baseline_xgb_features.json \
  --target_fraction 0.05
```

### 9. Sweep few-shot de XGBoost

```bash
python src/run_finetune_sweep_xgb.py \
  --pretrained_model artifacts/models/baseline_xgb.json \
  --metadata_path artifacts/models/baseline_xgb_features.json \
  --fractions 0.01 0.02 0.05 0.10
```

## Visualitzacions

Exemples:

```bash
python src/visualization/plot_baselines.py
python src/visualization/plot_feature_importance.py
python src/visualization/plot_mlp_vs_xgboost.py
python src/visualization/plot_day_ahead_benchmark.py
python src/visualization/plot_graph_topology.py
python src/visualization/plot_gnn_architecture.py
```

Les figures es guarden a `artifacts/figures/`.

## Tracking d’experiments

Els scripts d’entrenament fan servir `mlflow`.

Si vols centralitzar el tracking en una base SQLite local compatible amb `src/visualization/plot_mlp_vs_xgboost.py`, exporta:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

Si no es defineix, `MLflow` farà servir la configuració per defecte.

## Scripts principals

### Dades

- `src/data/download_entsoe.py`: descàrrega de demanda i generació d’ENTSO-E.
- `src/data/download_entsoe_forecast.py`: descàrrega del forecast day-ahead d’ENTSO-E.
- `src/data/download_weather.py`: descàrrega de temperatura històrica des d’Open-Meteo.
- `src/data/preprocess.py`: construcció del dataset final en format llarg.

### Models

- `src/models/baselines.py`: entrena `Daily Naive`, `Ridge` i `XGBoost`.
- `src/models/graphsage.py`: defineix el baseline `GraphSAGE`.
- `src/models/mlp_baseline.py`: defineix el baseline `MLP`.
- `src/train.py`: entrena el `MLP` sobre source i avalua zero-shot sobre target.
- `src/train_gnn.py`: entrena el `GraphSAGE` diagnòstic sobre demanda.
- `src/train_finetune.py`: fine-tuning few-shot del `MLP`.
- `src/train_finetune_xgb.py`: fine-tuning few-shot de `XGBoost`.
- `src/benchmarking/`: perfils reproduïbles de recursos per `XGBoost`, `MLP`, `GraphSAGE` i renovables.

### Anàlisi

- `src/models/ablation_features.py`: ablació de grups de variables.
- `src/models/ablation_weather.py`: impacte de la informació meteorològica.
- `src/models/evaluate_day_ahead_benchmark.py`: benchmark day-ahead.
- `src/run_resource_benchmark.py`: comparativa unificada de recursos en demanda.
- `src/run_renewables_benchmark.py`: comparativa unificada de recursos en renovables.
- `src/run_graph_topk_sweep.py`: sensibilitat del graf GraphSAGE al nombre de veïns.
- `notebooks/eda.ipynb`: anàlisi exploratòria inicial.

## Limitacions conegudes

- el projecte manté una estructura de codi de recerca, no de producte,
- no s'ha incorporat una cerca exhaustiva d'hiperparàmetres equivalent per totes les famílies de models,
- les dades, mètriques i figures generades es mantenen fora de Git per evitar pujar artefactes pesants.

## Llicència

Aquest repositori es distribueix sota la llicència inclosa a [`LICENSE`](LICENSE).
