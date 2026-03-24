# MLEChurn — Telco churn (production-style)

End-to-end binary churn model: **sklearn `Pipeline`**, **LightGBM**, custom feature engineering, **FastAPI** scoring, optional **MLflow** and **Optuna**.

## Architecture

- **`src/`** — installable application code (import as `src.*`). The repo adds `churn-ml-system/` (or `.`) to `PYTHONPATH` so `src` is the top-level package (no `src/src` nesting).
- **Data** — `src.data.loading` resolves paths from `CHURN_PROJECT_ROOT` / `--project-root` and `CHURN_DATA_DIR` / `configs/train.yaml`.
- **Training pipeline** (sequential, single learning graph):
  1. **Table prep** — drop `id`, coerce numeric columns (`src.models.preprocessing`).
  2. **Design matrix** — `ChurnTableToDesignMatrixTransformer`: domain features (`ChurnFeatureEngineer`) **then** column-bind with imputed numerics + one-hot categoricals (`src.models.design_matrix`).
  3. **Model** — `LGBMClassifier` (`src.models.pipeline`).
- **Inference** — `joblib` load + `predict_proba` positive class (`src.inference.predictor`).
- **API** — `src.api.main` FastAPI app; model path from `CHURN_MODEL_PATH` / `CHURN_PROJECT_ROOT`.
- **Config** — `configs/train.yaml` holds defaults (seeds, paths, model hyperparameters, CV, Optuna, MLflow, logging). CLI flags override YAML.

Reproducibility: **`random_seed`** in YAML drives CV splits, LightGBM, and Optuna sampler. For stable runs with multithreaded LightGBM, set **`training.lgbm_n_jobs: 1`** in YAML.

## Local setup

```bash
cd /path/to/MLEChurn
python -m venv churn-ml-system/.venv
churn-ml-system/.venv/bin/pip install -r requirements.txt
# optional dev:
churn-ml-system/.venv/bin/pip install -r requirements-dev.txt
```

Place training CSV under `./datasets/train.csv` (gitignored) or set `CHURN_DATA_DIR` / `paths.data_dir` in YAML.

## Train

From the **repository root** (so `configs/train.yaml` resolves):

```bash
churn-ml-system/.venv/bin/python train.py --project-root . --no-mlflow
```

Common overrides:

```bash
python train.py --project-root . --config configs/train.yaml --nrows 5000 --no-mlflow
python train.py --project-root . --optuna-trials 15 --no-mlflow
```

Artifacts default to **`models/churn_pipeline.joblib`** (see `paths.model_output` in YAML).

Logging: stderr. Level from **`CHURN_LOG_LEVEL`** or **`logging.level`** in YAML or `--log-level`. Set **`CHURN_LOG_FORMAT=json`** for JSON lines (good for log aggregation). Use **`src.common.logging_config.log_event`** for key-value context on important events.

## Run API

```bash
export PYTHONPATH=churn-ml-system
export CHURN_PROJECT_ROOT="$(pwd)"
churn-ml-system/.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

`GET /health`, `POST /predict` with body `{"records": [{...feature columns...}]}` → `{"churn_probability": [...]}`.

Docker (from repo root): `docker build -t churn-api -f Dockerfile .` then mount `./models` to `/app/models`.

## Monitoring & batch scoring

Generated artifacts go under **`reports/`** (gitignored) with UTC timestamps in filenames.

- **Drift** — `compute_numeric_drift_report` + `save_drift_report` write **`numeric_drift_<timestamp>.json`** (full payload + `generated_at_utc`) and **`.csv`** (one row per feature). From batch, pass **`--drift-reference path/to/train_sample.csv`**.
- **SHAP** — `design_matrix_as_array`, `compute_batch_shap_values`, `explain_global` / `explain_instance` (clear names; no hidden helpers). Batch: **`--with-shap`** writes **`batch_shap_values_<run_id>.csv`** (row index, optional `id`, SHAP columns) and **`batch_shap_global_<run_id>.json`** (mean |SHAP| for the batch).
- **Batch CLI** — `PYTHONPATH=churn-ml-system python -m src.batch.run_batch -i in.csv -o out.csv --project-root . [--with-shap] [--drift-reference train.csv] [--reports-dir reports]`.

## Tests

```bash
churn-ml-system/.venv/bin/python -m pytest
```

## Layout

```
MLEChurn/
  configs/train.yaml
  train.py
  churn-ml-system/
    src/           # Python package root name: src
      api/
      common/
      config/
      data/
      features/
      inference/
      models/
      monitoring/
      batch/
      tracking/
  tests/
```
