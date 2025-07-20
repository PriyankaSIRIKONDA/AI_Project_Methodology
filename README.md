# Churn Prediction Project

## Overview
This project predicts customer churn using machine learning, following best practices for code structure, reproducibility, experiment tracking, and deployment. The workflow is based on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and is organized for production-readiness and collaboration.

---

## Project Structure
```
churn-prediction/
├── data/
│   ├── raw/           # Original data
│   ├── processed/     # Cleaned and feature-engineered data
│   └── ...
├── src/
│   ├── data/          # Data preprocessing scripts
│   ├── features/      # Feature engineering scripts
│   ├── models/        # Model training and prediction scripts
│   ├── api/           # FastAPI deployment script
│   └── ...
├── models/            # Trained model artifacts
├── notebooks/         # Jupyter notebooks for EDA and analysis
├── docs/              # Sphinx documentation
├── requirements.txt   # Python dependencies
├── conda.yaml         # Conda environment for MLflow Project
├── MLproject          # MLflow Project configuration
├── README.md          # This file
└── ...
```

---

## Setup Instructions

1. **Clone the repository**
2. **Install dependencies**:
   - With pip:
     ```bash
     pip install -r requirements.txt
     ```
   - Or with conda (recommended for MLflow Projects):
     ```bash
     conda env create -f conda.yaml
     conda activate churn-prediction
     ```
3. **Download the dataset** from Kaggle and place it in `data/raw/`.

---

## Step-by-Step Workflow

### 1. Data Preprocessing
- **Script:** `src/data/preprocess.py`
- **Purpose:** Cleans raw data, handles missing values, encodes categorical variables.
- **Run:**
  ```bash
  python src/data/preprocess.py
  ```
- **Output:** `data/processed/cleaned_telco.csv`

### 2. Feature Engineering
- **Script:** `src/features/build_features.py`
- **Purpose:** Adds tenure groups, service count, and other engineered features.
- **Run:**
  ```bash
  python src/features/build_features.py
  ```
- **Output:** `data/processed/featured_telco.csv`

### 3. Model Training & Experiment Tracking
- **Script:** `src/models/train_model.py`
- **Purpose:** Trains a Random Forest model, evaluates performance, and logs experiments with MLflow.
- **Run:**
  ```bash
  python src/models/train_model.py
  ```
- **Output:** `models/rf_telco_churn.pkl`, MLflow logs
- **MLflow UI:**
  ```bash
  mlflow ui
  ```
  Visit [http://localhost:5000](http://localhost:5000) to view experiment runs.

### 4. Model Deployment as an API
- **Script:** `src/api/predict_api.py`
- **Purpose:** Serves the trained model via a REST API using FastAPI.
- **Run:**
  ```bash
  uvicorn src.api.predict_api:app --reload
  ```
- **Test:**
  Visit [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI and try the `/predict` endpoint.

### 5. MLflow Project Packaging
- **Files:** `MLproject`, `conda.yaml`
- **Purpose:** Enables full reproducibility and one-command runs.
- **Run:**
  ```bash
  mlflow run .
  ```
- **Custom parameters:**
  ```bash
  mlflow run . -P n_estimators=200 -P max_depth=10
  ```

### 6. (Optional) Cloud Deployment Proposal
- **Proposal:**
  - Containerize the API and model with Docker.
  - Deploy on Google Cloud Run or AI Platform for scalable serving.
  - Host MLflow server on a GCP VM for remote experiment tracking.

---

## Code Quality & Documentation
- **PEP8 compliance:** Enforced with `black` and `flake8`.
- **Docstrings:** Present in all scripts for clarity.
- **Sphinx:** Initial setup in `docs/` for auto-generated documentation.
- **README.md:** This file provides a full project overview and usage guide.

---

## References
- [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [cookiecutter-data-science](https://drivendata.github.io/cookiecutter-data-science/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Sphinx Documentation](https://www.sphinx-doc.org/en/master/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
