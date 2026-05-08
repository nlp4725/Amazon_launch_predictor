## Amazon Launch Predictor — ML End-to-End Project

## Project Overview

Amazon Launch Predictor is an end-to-end machine learning pipeline that predicts whether a newly launched Amazon product will accumulate more than 10 reviews within 180 days of listing. The project targets third-party sellers in the $15–$100 price range launching standalone products. It follows ML engineering best practices with modular pipelines, containerization, Google Cloud deployment, and a REST API with Streamlit dashboard for interactive predictions.

## Architecture

The codebase is organized into distinct pipelines following the flow:
`Load → Preprocess → Feature Engineering → Train → Evaluate → Inference → Serve`

### Core Modules

- **`src/data_collection_pipeline/`**: Raw data ingestion from source
  - `ingest.py`: Loads product listing data from SQLite into a flat DataFrame

- **`src/feature_pipeline/`**: Data loading, preprocessing, and feature engineering
  - `load.py`: Loads raw parquet or SQLite data
  - `preprocessing.py`: Extracts price, seller, and title from raw Keepa JSON; parses time-series review and sales rank histories using Keepa timestamp format
  - `feature_engineering.py`: TF-IDF vectorization of product titles, ColumnTransformer for numeric and categorical features, train/test split, label generation (>10 reviews at 180 days)

- **`src/training_pipeline/`**: Model training and evaluation
  - `train.py`: LGBMClassifier with `class_weight='balanced'` to handle class imbalance (~12% success rate); saves model to `src/serving/model/`
  - `evaluate.py`: Computes ROC-AUC, PR-AUC, precision, recall, and F1 at threshold 0.6; saves results to parquet

- **`src/inference_pipeline/`**: Production inference
  - `inference.py`: `predict()` takes title, price, and category directly; `fetch_product_from_keepa()` queries the Keepa API to resolve an ASIN; `predict_from_asin()` combines both

- **`src/serving/model/`**: Committed model artifacts
  - `lgbm_tfidf_model.pkl`: Trained LightGBM classifier
  - `preprocessor.pkl`: Fitted ColumnTransformer (TF-IDF + numeric pipeline)

### Web Applications

- **`main.py`**: FastAPI service
  - `GET /` — health check
  - `POST /predict/asin` — fetches product from Keepa by ASIN, returns prediction
  - `POST /predict/url` — parses ASIN from an Amazon product URL, returns prediction
  - Keepa token guard — returns 503 if fewer than 10 API tokens remain

- **`app.py`**: Streamlit dashboard
  - Accepts ASIN or full Amazon product URL
  - Displays launch probability and Success/Failure label
  - Reads `API_URL` from environment (defaults to `http://localhost:8000` for local dev)

### Cloud Infrastructure & Deployment

- **Google Cloud Run**: Two separate services — FastAPI (port 8000) and Streamlit (port 8501)
- **Google Secret Manager**: Stores `KEEPA_API_KEY`; injected into Cloud Run at deploy time via Terraform
- **Cloud Build**: CI/CD trigger on push to `main` — builds both Docker images, pushes to Artifact Registry, deploys both services
- **Terraform**: All infrastructure defined as code in `terraform/main.tf`

#### Cloud Run Services:
- **fastapi-service**: FastAPI backend — `https://fastapi-service-o3kqyzyx6q-uc.a.run.app`
- **streamlit-service**: Streamlit dashboard — `https://streamlit-service-o3kqyzyx6q-uc.a.run.app`

## Common Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Local Development
```bash
# Copy and populate your API key
cp .env.example .env   # then add KEEPA_API_KEY

# Start FastAPI
uvicorn main:app --reload

# Start Streamlit (separate terminal)
streamlit run app.py
```

### Full ML Pipeline (retrain from scratch)
```bash
python pipeline.py
```

### Run Individual Pipeline Steps
```bash
# 1. Preprocess raw data
python src/feature_pipeline/preprocessing.py

# 2. Feature engineering + train/test split
python src/feature_pipeline/feature_engineering.py

# 3. Train model
python src/training_pipeline/train.py

# 4. Evaluate model
python src/training_pipeline/evaluate.py
```

### Inference
```bash
# Manual input
python src/inference_pipeline/inference.py

# Via API
curl -X POST "http://localhost:8000/predict/asin?asin=B0XXXXXXXX"
curl -X POST "http://localhost:8000/predict/url?url_str=https://www.amazon.com/dp/B0XXXXXXXX"
```

### Testing
```bash
# Run all tests
pytest

# Run specific modules
pytest tests/test_preprocessing.py
pytest tests/test_feature_engineering.py
pytest tests/test_train.py
pytest tests/test_evaluate.py
pytest tests/test_inference.py

# Verbose output
pytest -v
```

### Docker
```bash
# Build FastAPI container
docker build -f Dockerfile.fastapi -t amazon-launch-fastapi .

# Build Streamlit container
docker build -f Dockerfile.streamlit -t amazon-launch-streamlit .

# Run FastAPI container
docker run -p 8000:8000 --env-file .env amazon-launch-fastapi

# Run Streamlit container
docker run -p 8501:8501 -e API_URL=http://localhost:8000 amazon-launch-streamlit
```

### Infrastructure
```bash
# Provision Google Cloud infrastructure
cd terraform
terraform init
terraform apply

# Trigger a manual Cloud Build deploy
gcloud builds submit --config cloudbuild.yaml
```

## Key Design Patterns

### Pipeline Modularity
Each pipeline stage is independently runnable. Parquet checkpoints (`preprocessed.parquet`, `features_train.parquet`, etc.) allow resuming from any step without rerunning upstream stages.

### Keepa Time-Series Parsing
Keepa returns price, review count, and sales rank as flat arrays alternating `[keepa_timestamp, value, keepa_timestamp, value, ...]`. `preprocessing.py` converts these into `{days_since_launch: value}` dicts, enabling windowed feature extraction (e.g. reviews at day 30, 90, 180).

### Class Imbalance Handling
Only ~12% of standalone listings accumulate >10 reviews in 180 days. LGBMClassifier is trained with `class_weight='balanced'` and evaluated at threshold 0.6 — tuned for precision over recall since a false positive wastes a seller's launch investment.

### Secret Management
`KEEPA_API_KEY` is stored in Google Secret Manager and injected into Cloud Run at startup via Terraform's `secret_key_ref`. Locally, it is read from a gitignored `.env` file. The key never appears in source code or git history.

### CI/CD Flow
Push to `main` → Cloud Build fetches the FastAPI Cloud Run URL dynamically → builds and pushes both images → deploys both services with the correct `API_URL` env var wired between them. No manual deploy steps.

### Model Artifact Versioning
Model `.pkl` files are committed to `src/serving/model/` so Cloud Run containers are fully self-contained — no GCS bucket required at inference time. Retraining runs `python pipeline.py` locally, then a push to `main` redeploys automatically.

## Dependencies

Key production dependencies (see `requirements.txt`):
- **ML/Data**: `lightgbm==4.6.0`, `scikit-learn==1.8.0`, `pandas==3.0.2`, `numpy==2.4.4`
- **API**: `fastapi==0.136.1`, `uvicorn==0.46.0`, `pydantic==2.13.3`
- **Dashboard**: `streamlit==1.57.0`
- **Data sourcing**: `requests==2.33.1` (Keepa REST API)
- **Config**: `python-dotenv==1.2.2`, `joblib==1.5.3`

## File Structure Notes

- **`data/raw/`**: Source parquet and SQLite — gitignored, not committed
- **`data/processed/`**: Pipeline checkpoint parquets — gitignored
- **`src/serving/model/`**: Trained model and preprocessor pkl — committed (exception to gitignore)
- **`notebooks/`**: Jupyter notebooks for EDA and experimentation
- **`tests/`**: Unit and integration tests for each pipeline component
- **`terraform/main.tf`**: All Google Cloud infrastructure as code
- **`cloudbuild.yaml`**: CI/CD pipeline — build, push, deploy both services
- **`Dockerfile.fastapi`** / **`Dockerfile.streamlit`**: Separate containers per service
