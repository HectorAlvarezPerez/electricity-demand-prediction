# Predicció de demanda elèctrica i adaptació de domini a Catalunya

Repositori del TFG sobre predicció horària de demanda elèctrica i generació renovable, amb Catalunya com a motivació principal i `ES` com a domini objectiu operatiu en la pipeline actual.

Títol de treball:
`Prediccio de Demanda Electrica i Adaptacio de Domini a Catalunya amb OT`.

## Objectiu

Aquest projecte reconstrueix la narrativa per demostrar experimentalment quan *no compensa usar una xarxa neuronal*, comparant `XGBoost` davant d'una única GNN simple i reproduïble en els eixos de: precisió, cost computacional i latència.

S'estructura en dos benchmarks paral·lels:
1. **Benchmark de Demanda**: Previsió de demanda horària.
2. **Benchmark de Renovables**: Previsió de generació (solar, eòlica, hidro, etc).

Per a cadascun d'ells la intenció és avaluar el millor baseline tabular (`XGBoost`) amb el baseline neuronal senzill d'aprenentatge sobre grafs (`GraphSAGE`), incloent-hi traces d'informació meteorològica prèvia.

## Estat actual

Actualment hi ha implementat:

- descàrrega de demanda i generació des d’ENTSO-E,
- descàrrega de demanda de Catalunya des de REE/ESIOS,
- descàrrega de temperatura històrica des d’Open-Meteo,
- preprocessat a format llarg amb lags explícits i horitzó de 24 hores (demanda),
- splits temporals `train` / `val` / `test`,
- baselines tabulars `Daily Naive`, `Ridge` i `XGBoost` (demanda),
- baseline neuronal `MLP` (demanda),
- fine-tuning few-shot per a `MLP` i `XGBoost`,
- scripts de visualització i comparació de resultats.

Els propers passos de desenvolupament (Roadmap):

- marc unificat d'avaluació de hardware/recursos (latència, throughput, pesos),
- construcció de dades tipus graf associant per país segons correlació a `train`,
- implementació del benchmark final per demanda amb `GraphSAGE`,
- introducció del preprocessat diferenciat per a les generacions renovables,
- entrenament `XGBoost` i `GraphSAGE` paral·lel sobre les renovables,
- i figures d'anàlisi de resultats.

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
│   ├── data/
│   ├── models/
│   ├── visualization/
│   ├── train.py
│   ├── train_finetune.py
│   ├── train_finetune_xgb.py
│   ├── run_finetune_sweep.py
│   └── run_finetune_sweep_xgb.py
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

Punt important: tot i que el TFG està orientat a Catalunya, el codi d’entrenament i avaluació que hi ha implementat ara mateix treballa amb `ES` com a domini target. La part catalana existeix a la capa de descàrrega de dades, però encara no està integrada com a target final dins de la pipeline principal.

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
REE_TOKEN=el_teu_token_esios
```

També s’accepta `ESIOS_TOKEN_KEY` com a alternativa a `REE_TOKEN`.

## Flux recomanat

### 1. Descarregar demanda europea

```bash
python src/data/download_entsoe.py
```

Sortides:

- `data/raw/europe/demand/entsoe_demand_{country}.csv`
- `data/raw/europe/generation/entsoe_generation_{country}.csv`

### 2. Descarregar demanda de Catalunya i alguns indicadors de generació

```bash
python src/data/download_esios.py
```

Sortides:

- `data/raw/catalonia/demand/esios_demand_real.csv`
- `data/raw/catalonia/demand/esios_demand_scheduled.csv`
- fitxers addicionals sota `data/raw/catalonia/generation/`

El valor per defecte del `start_year` ja està configurat per reconstruir històric des de `2015`.

### 3. Descarregar meteorologia històrica

```bash
python src/data/download_weather.py
```

Sortida:

- `data/raw/weather/weather_{country}.csv`

### 4. Construir el dataset processat

```bash
python src/data/preprocess.py
```

Sortides:

- `data/processed_long/train.parquet`
- `data/processed_long/val.parquet`
- `data/processed_long/test.parquet`

### 5. Entrenar baselines clàssics

```bash
python src/models/baselines.py
```

Genera:

- `artifacts/models/baseline_ridge.joblib`
- `artifacts/models/baseline_xgb.json`
- `artifacts/models/baseline_xgb_features.json`
- `artifacts/metrics/baseline_metrics.json`

### 6. Entrenar el baseline MLP

```bash
python src/train.py --seed 42 --epochs 30
```

Genera:

- `artifacts/models/mlp_tabular_long_seed42.pt`
- `artifacts/metrics/mlp_metrics_seed42.json`

### 7. Fine-tuning few-shot del MLP

```bash
python src/train_finetune.py \
  --pretrained_model artifacts/models/mlp_tabular_long_seed42.pt \
  --target_fraction 0.05 \
  --epochs 15
```

### 8. Sweep few-shot del MLP

```bash
python src/run_finetune_sweep.py \
  --pretrained_model artifacts/models/mlp_tabular_long_seed42.pt \
  --fractions 0.01 0.02 0.05 0.10
```

### 9. Fine-tuning few-shot de XGBoost

```bash
python src/train_finetune_xgb.py \
  --pretrained_model artifacts/models/baseline_xgb.json \
  --metadata_path artifacts/models/baseline_xgb_features.json \
  --target_fraction 0.05
```

### 10. Sweep few-shot de XGBoost

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
- `src/data/download_esios.py`: descàrrega de demanda i indicadors per a Catalunya.
- `src/data/download_weather.py`: descàrrega de temperatura històrica des d’Open-Meteo.
- `src/data/preprocess.py`: construcció del dataset final en format llarg.

### Models

- `src/models/baselines.py`: entrena `Daily Naive`, `Ridge` i `XGBoost`.
- `src/models/mlp_baseline.py`: defineix el baseline `MLP`.
- `src/train.py`: entrena el `MLP` sobre source i avalua zero-shot sobre target.
- `src/train_finetune.py`: fine-tuning few-shot del `MLP`.
- `src/train_finetune_xgb.py`: fine-tuning few-shot de `XGBoost`.

### Anàlisi

- `src/models/ablation_features.py`: ablació de grups de variables.
- `src/models/ablation_weather.py`: impacte de la informació meteorològica.
- `src/models/evaluate_day_ahead_benchmark.py`: benchmark day-ahead.
- `notebooks/eda.ipynb`: anàlisi exploratòria inicial.

## Limitacions conegudes

- la pipeline principal encara no fa servir Catalunya com a domini target final,
- no hi ha encara implementació d’OT ni de GNN,
- no hi ha suite de tests automatitzada,
- el projecte continua tenint estructura de codi de recerca, no de producte.

## Llicència

Aquest repositori es distribueix sota la llicència inclosa a [`LICENSE`](LICENSE).
