<div align="center">

# 🚀 Churn Prediction ML System

**Production-grade Machine Learning system focused on real-world deployment, monitoring, and explainability.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![ML](https://img.shields.io/badge/Machine%20Learning-LightGBM-FF6F00?style=for-the-badge)
![MLOps](https://img.shields.io/badge/MLOps-MLflow-blue?style=for-the-badge)
![API](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-Pytest-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production--Oriented-success?style=for-the-badge)

</div>

---

## 🧠 Overview

This project implements an **end-to-end Machine Learning system** to predict customer churn.

Instead of focusing only on model performance, the goal was to simulate how ML systems are built in **real-world production environments**.

---

## 🏗️ System Architecture

```
Raw Data → Feature Engineering → Preprocessing → Model → Predictions
                                      ↓
                        ┌─────────────┬─────────────┐
                        │   FastAPI   │   Batch Job │
                        └─────────────┴─────────────┘
                                      ↓
                          Monitoring Layer (Drift + SHAP)
```

---

## ⚙️ Tech Stack

- Python
- scikit-learn
- LightGBM
- MLflow
- Optuna
- FastAPI
- SHAP
- Pytest

---

## 🔍 Key Features

- 📊 Churn prediction model (LightGBM)
- 🔁 Batch + real-time inference
- 📈 SHAP explainability
- 📉 Data drift monitoring
- 🧪 Unit tests
- 📦 MLflow experiment tracking
- 🧠 Feature engineering pipeline

---

## 🚀 Quickstart

```bash
pip install -r requirements.txt
```

```bash
kaggle competitions download -c playground-series-s6e3
unzip playground-series-s6e3.zip -d data/
```

```bash
python train.py
```

```bash
uvicorn src.api.main:app --reload
```

```bash
python src/batch/run_batch.py --with-shap
```

---

## 📁 Project Structure

```
src/
├── api/            # FastAPI service
├── batch/          # Batch inference pipeline
├── features/       # Feature engineering
├── models/         # Training & inference logic
├── monitoring/     # Drift + logging
```

---

## 📊 Example Output

```json
{
  "prediction": 1,
  "probability": 0.87,
  "top_features": ["tenure", "contract_type", "monthly_charges"]
}
```

---

## 🧠 What This Project Demonstrates

✔ End-to-end ML system design  
✔ Production-oriented thinking  
✔ Monitoring and reliability  
✔ Explainability integration  
✔ Real-world ML engineering practices  

---

## 💡 Final Thoughts

This project focuses on building **robust, production-ready ML systems**, not just models.

It represents the transition from:
> Data Analyst → Machine Learning Engineer

---

## 🤝 Connect

Feel free to connect or reach out on LinkedIn!

