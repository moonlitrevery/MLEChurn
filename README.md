# 🚀 Churn Prediction ML System

```{=html}
<p align="center">
```
`<b>`{=html}Production-oriented Machine Learning project focused on
real-world deployment, monitoring, and explainability.`</b>`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## 🧠 Overview

This project implements an **end-to-end Machine Learning system** to
predict customer churn.

Instead of focusing only on model performance, the goal was to simulate
how ML systems are built in real-world environments.

------------------------------------------------------------------------

## 🏗️ Architecture

    Raw Data
       ↓
    Feature Engineering
       ↓
    Preprocessing
       ↓
    Model (LightGBM)
       ↓
    Predictions
       ↓
     ┌───────────────┬───────────────┐
     │   FastAPI     │   Batch Job   │
     └───────────────┴───────────────┘
              ↓
       Monitoring Layer
     (Drift + Explainability)

------------------------------------------------------------------------

## ⚙️ Tech Stack

-   Python
-   scikit-learn
-   LightGBM
-   MLflow
-   Optuna
-   FastAPI
-   SHAP

------------------------------------------------------------------------

## 🚀 How to Run

``` bash
pip install -r requirements.txt
```

``` bash
kaggle competitions download -c playground-series-s6e3
unzip playground-series-s6e3.zip -d data/
```

``` bash
python train.py
```

``` bash
uvicorn src.api.main:app --reload
```

``` bash
python src/batch/run_batch.py --with-shap
```

------------------------------------------------------------------------

## 🧠 What This Project Demonstrates

✔ End-to-end ML system design\
✔ Production-oriented thinking\
✔ Monitoring and reliability\
✔ Explainability integration\
✔ Real-world ML engineering practices

------------------------------------------------------------------------

## 💡 Final Thoughts

This project focuses on building **robust, production-ready ML
systems**, not just models.
