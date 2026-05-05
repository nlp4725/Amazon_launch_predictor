# Amazon Launch Predictor — Notebook to Script Migration

## Overview

This document tracks the process of converting an exploratory Jupyter notebook into a
production-ready Python script, with version control set up from the start.

---

## Step 1: Set Up Git Repository and Folders

Create a GitHub repository, then clone it to my local machine:

```bash
git clone <repo-url>
cd Amazon_launch_predictor
```

Move the existing notebook into the project folder:

```bash
mv launch_predictor.ipynb ../Amazon_launch_predictor/
```

Under the correct directory, create project structure. -p is used to make subfolders:

```bash
mkdir -p src/{data,features,models,serving,validation} tests notebooks data/{raw,processed} models
```

| Folder / File | Purpose |
|---|---|
| `data/` | raw and processed .db .npy .csv files |
| `models/` | saved .pkl files from training runs (messy) |
| `notebooks/` | exploratory .ipynb files |
| `tests/` | automated tests to verify code correctness |
| `src/data/` | data loading and preprocessing scripts |
| `src/features/` | feature engineering scripts |
| `src/models/` | training, tuning, and evaluation scripts |
| `src/serving/` | inference / API logic |
| `src/serving/model/` | final .pkl model and scaler/encoder bundled for the API |
| `src/validation/` | data quality checks |
| `pipeline.py` | runs full ML pipeline end-to-end (load → preprocess → train → evaluate) |
| `main.py` | FastAPI app — exposes model as an API endpoint |
| `app.py` | Streamlit/Gradio UI — visual interface for predictions |
| `Dockerfile` | packages the app into a container for deployment |



Create files under root directory:

```bash
touch requirements.txt README.md .gitignore main.py pipeline.py Dockerfile app.py
```

Move raw database and embedding numpy files to data/raw:

```bash
mv product_launch.db title_embedding_all.npy title_embedding.npy data/raw
```

Move .ipynb file to notebooks folder:

```bash
mv *.ipynb notebooks
```

Update .gitignore (**DO not commit things before push .gitignore**):

Add content to .gitignore
```bash
# Virtual environments
.venv/
venv/
env/

# Data
data/raw/
data/processed/

# Models & artifacts ignore training runs
models/ 
mlruns/
artifacts/


# Python
__pycache__/
*.py[cod]
.env
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store

# BUT keep our bundled serving model in the repo
!src/serving/model/
!src/serving/model/**
```

```bash
git add .gitignore
git commit -m "add .gitignore
git add .
git commit -m "created folders and subfolders
```

To unstage:

```bash
git reset Head
```

Fix commited database. stop them from being tracked in the future:

```bash
git rm --cached data/raw/product_launch.db data/raw/title_embedding.npy data/raw/title_embedding_all.npy
git add .gitignore
```

To check if they are still being checked:

```bash
git ls-files data/raw/
```

Handling mistake: DB too big to push. 

Wipe local history clean:

```bash
rm -rf .git
```

Start fresh:

``bash
git init 
git branch -M main
```

Add ignore first

```bash
git add .gitignore
git commit -m "add .gitignore"
git add .
git commit -m "project setup"
```

Connect to github and force push

```bash
git remote add origin https://github.com/nlp4725/Amazon_launch_predictor.git
git push -f origin main
```





Enviroment setup:

Have conda working. skip.

```bash
conda create -n amazon_predictor python==3.10
conda activate amazon_predictor
pip install pandas scikit-learn xgboost mlflow fastapi uvicorn streamlit gradio
```

Generate requrement.txt

## Step2 EDA

Have finished this step. skip.

## Step3 Modularize: Notebook to script 

Add subfolders

```bash
mkdir src/{data_collection_pipeline,feature_pipeline}
mkdir -p src/{training_pipeline,inference_pipeline}
```

Add __init__.py

```bash
touch src/__init__.py src/feature_pipeline/__init__.py src/training_pipeline/__init__.py src/inference_pipeline/__init__.py src/data_collection_pipeline/__init__.py tests/__init__.py
```

### Data_Collection_Pipeline

Raw data was collected from the Keepa API covering newly launched Amazon products between January 2024 and December 2025. The scraper is in `src/data_collection_pipeline/ingest.py`.

**Collection flow (per month):**

| Main Function | What it does |
|---|---|
| `get_or_cache_asins()` | Queries the Keepa product finder with filters (private label sellers only, excludes Amazon) to get ASINs launched that month. Saves results to `asin_list` table — skips the API call if that month is already cached. |
| `fetch_and_save_products()` | Fetches full product data (title, review count, price, buy box history) for each ASIN in batches of 100. Saves to `products` table with `(asin, month)` as primary key. Only fetches ASINs not already in the database to handle partial runs. |
| `save_data_to_db()` | Entry point. Creates the two SQLite tables and iterates through each month calling the two functions above. |

**Token management:** Keepa refills at 20 tokens/min. If remaining < 300, waits `(300 - remaining) / 20` mins before the next batch.

**SQLite tables created:**

```
asin_list  — (month, asin)
products   — (asin, month, title, cat, raw_data, buybox, rating)
```

**SQLite tables update and save:**

```
conn.commit() - every month or every 100 asins
```

### Feature_Pipeline

```bash
touch src/feature_pipeline/load.py src/feature_pipeline/preprocessing.py src/feature_pipeline/feature_engineering.py
```

---

#### `load.py`

Connects to SQLite, queries the `products` table, sorts by month, saves a parquet checkpoint, and returns the DataFrame for in-memory pipeline chaining.

| Function | In | Out |
|---|---|---|
| `load_data(db_path, output_dir)` | `product_launch.db` | DataFrame + `data/raw/product_launch.parquet` |

**Why save parquet?** Allows skipping `load_data()` on re-runs — read directly from parquet instead of querying SQLite again.

To inspect columns after loading:
```bash
python3 -c "from src.feature_pipeline.load import load_data; df = load_data(); print(df.dtypes)"
```

**Testing logic (`tests/test_load.py`):**
- Create a fake SQLite db using `tmp_path` (pytest built-in temporary folder)
- Insert one dummy row into `products` table
- Call `load_data()` with fake db path
- Assert df is not empty, has `asin` column, and parquet file was saved

```bash
pytest tests/test_load.py -v
```

---

#### `preprocessing.py`

Extracts structured columns from raw JSON fields in `raw_data` and `buybox`. Returns a clean DataFrame suitable for EDA or feature engineering. Labeling and splitting happen in `feature_engineering.py`.

**Helper functions:**

| Function | In | Out |
|---|---|---|
| `convert_history_to_dict(original_list, list_time)` | Keepa time-series list `[time, value, ...]`, listing timestamp | `{days_since_launch: value}` dict or None |
| `sales_rank_at_months(original_list, start_days, end_days, list_time)` | Raw sales rank list, day window, listing timestamp | Average sales rank within the window or None |
| `extraction(row)` | Single `raw_data` JSON string | `pd.Series` with review, monthly sold, and sales rank columns |

**Main functions:**

| Function | In | Out |
|---|---|---|
| `add_price_seller_title(df)` | df with `raw_data`, `buybox` columns | df + `listed_price`, `price`, `title`, `seller` |
| `add_review_n_monthly_sold(df)` | df with `raw_data` column | df joined with all `extraction()` output columns |
| `run_preprocess(df, output_dir)` | raw df from `load_data()` | clean df + `data/processed/preprocessed.parquet` |

**Testing logic (`tests/test_preprocessing.py`):**
- Build a minimal fake DataFrame with realistic JSON strings in `raw_data` and `buybox`
- `@pytest.fixture` defines `sample_df` once and injects it into each test
- Test each function independently — no real database needed
- `test_run_preprocess` uses `tmp_path` to verify parquet is saved without touching real `data/`

```bash
pytest tests/test_preprocessing.py -v
```

---

#### `feature_engineering.py`

Prepares the clean DataFrame for model training. Filters, labels, splits, and encodes. Title text is vectorized with TF-IDF — no sentence-transformer or GPU required.

**Pipeline order:**
```
filter_and_label()      ← filter rows, create label
drop_and_extract_date() ← select columns, extract month/year
split()                 ← time-based train/test split
fit_and_transform()     ← fit OHE + TF-IDF on train only, apply to test
```

| Function | In | Out |
|---|---|---|
| `filter_and_label(df)` | df with `most_recent_review_time`, `most_recent_review` | filtered df + `label` column (1 = >10 reviews, 0 = failure) |
| `drop_and_extract_date(df)` | labeled df | df with selected columns + `month` (int), `year`, `date`, `title` |
| `split(df)` | df with `date` and `label` | `X_train`, `y_train`, `X_test`, `y_test` |
| `fit_and_transform(X_train, X_test, models_dir)` | train/test DataFrames | transformed np.arrays + `models/preprocessor.pkl` |
| `run_feature_engineering(df, output_dir)` | clean df from `run_preprocess()` | `features_train.parquet`, `features_test.parquet`, `y_train.parquet`, `y_test.parquet` |

**Key design decisions:**
- `fit_transform()` on train, `.transform()` only on test — prevents data leakage
- `TfidfVectorizer(max_features=200)` replaces sentence embeddings — lightweight, no torch/GPU needed
- `preprocessor.pkl` saved to `models/` so inference applies the same transformations to new data without refitting

**Testing logic (`tests/test_feature_engineering.py`):**
- `sample_df` fixture includes a `title` column with fake product title strings
- Each function tested independently with small fake DataFrames
- `test_fit_and_transform` skips gracefully if fixture doesn't produce both train and test rows
- `test_run_feature_engineering` checks all four parquet files are saved to `tmp_path`

```bash
pytest tests/test_feature_engineering.py -v
```

Run all tests:
```bash
pytest tests/ -v
```
**Critical pattern — `feature_engineering.py` and `test_feature_engineering.py`:** All directory and file paths (`output_dir`, `models_dir`, `emb_path`) must be passed as function parameters with sensible defaults. This lets tests override them with `tmp_path` so no real files in `data/` or `models/` are touched during testing. Never hardcode paths inside functions.

---

### Training_Pipeline

```bash
touch src/training_pipeline/train.py src/training_pipeline/evaluate.py
```

---

#### `train.py`

Fits `LGBMClassifier` on the encoded training features and saves the model to disk.

| Function | In | Out |
|---|---|---|
| `train(X_train, y_train, models_dir)` | encoded np.array features and binary labels | fitted model + `models/lgbm_tfidf_model.pkl` |

**Key design decisions:**
- `class_weight='balanced'` — dataset has ~12% success rate, this prevents the model from predicting failure for everything
- Returns the fitted model so `pipeline.py` can pass it directly to `evaluate()` without reloading from disk

**Testing logic (`tests/test_train.py`):**
- Random numpy arrays as `X_train`, `y_train` — no real data needed since the model just needs valid input shapes
- Asserts model is returned and `.pkl` file is saved to `tmp_path`

```bash
pytest tests/test_train.py -v
```

---

#### `evaluate.py`

Loads the trained model, computes classification metrics on test data, and saves results.

| Function | In | Out |
|---|---|---|
| `evaluate(X_test, y_test, model, models_dir, output_dir)` | test features and labels | dict of metrics + `data/processed/evaluation_results.parquet` |

**Key design decisions:**
- `model=None` by default — if not passed, loads `lgbm_tfidf_model.pkl` from `models_dir`. In tests, a dummy trained model is passed directly so no `.pkl` file is needed.
- Uses `predict_proba` (not `predict`) for ROC-AUC and PR-AUC — these metrics need probabilities, not hard class predictions.
- Metrics saved as `evaluation_results.parquet` so results are reproducible and can be compared across runs.
- Returns the metrics dict so callers (tests, `pipeline.py`) can use the values without re-reading the parquet.

**Testing logic (`tests/test_evaluate.py`):**
- Fixture trains a real `LGBMClassifier` on random data in-memory — no `.pkl` loading needed.
- Passes the trained model directly to `evaluate()` via `model=` parameter.
- Passes `output_dir=tmp_path` so parquet saves to a temp folder, not `data/processed/`.
- Asserts returned dict has the expected keys and all values are between 0 and 1.

```bash
pytest tests/test_evaluate.py -v
```

---

### Inference_Pipeline

```bash
touch src/inference_pipeline/inference.py
```

---

#### `inference.py`

Takes a new product's ASIN or Amazon URL, fetches product data from Keepa, and returns a launch success prediction.

| Function | In | Out |
|---|---|---|
| `fetch_product_from_keepa(asin)` | ASIN string | dict with `title`, `price`, `cat`, `seller` |
| `predict(title, price, cat, seller, models_dir)` | product fields | dict with `predicted_probability`, `predicted_label` |
| `predict_from_asin(asin, models_dir)` | ASIN string | product fields + `predicted_probability`, `predicted_label` |

**Key design decisions:**
- `seller` defaults to `"unknown"` for manual input — OHE was fitted with `handle_unknown='ignore'` so unseen sellers encode as all zeros
- `model=None` pattern — model is passed directly in tests to avoid loading `.pkl` from disk
- `{**product, **result}` merges product info and prediction into one response dict

**Testing logic (`tests/test_inference.py`):**
- `fitted_models` fixture builds a minimal preprocessor and model saved to `tmp_path` — no real `.pkl` files needed
- `fetch_product_from_keepa` tested by mocking `requests.get` — no real Keepa API calls made
- `predict_from_asin` tested by mocking `fetch_product_from_keepa` — isolates prediction logic from HTTP calls

```bash
pytest tests/test_inference.py -v
```

---

### `pipeline.py`

Chains all pipeline steps end to end. Run this to retrain the model from scratch.

```
load → preprocess → feature_engineering → train → evaluate
```

**Skip logic:** If `product_launch.parquet` or `preprocessed.parquet` already exist, those steps are skipped — avoids slow SQLite and JSON parsing on re-runs.

```bash
python pipeline.py
```

---

### `main.py`

FastAPI app — exposes inference as HTTP endpoints. Users input an ASIN or Amazon URL and get back product information plus a launch success prediction.

**User flow:**
```
User pastes ASIN or Amazon URL
        ↓
main.py checks Keepa token balance (raises 503 if < 10 tokens)
        ↓
inference.py fetches title, price, cat, seller from Keepa
        ↓
preprocessor.pkl transforms the input row
        ↓
lgbm_tfidf_model.pkl predicts probability
        ↓
Returns: title, price, cat, seller, predicted_probability, predicted_label
```

| Endpoint | Input | Output |
|---|---|---|
| `GET /` | — | health check message |
| `POST /predict/asin` | `asin: str` | product info + prediction |
| `POST /predict/url` | `url_str: str` | parses ASIN from URL, product info + prediction |

**URL parsing:** uses regex `r'/dp/([A-Z0-9]{10})'` to extract the ASIN — works regardless of URL format, no string splitting needed.

**Token check:** Keepa refills at 20 tokens/min. If < 10 tokens, returns HTTP 503 so the user knows to retry later rather than hanging.

**How to test locally:**

1. Start the server:
```bash
uvicorn main:app --reload
```

2. Open `http://localhost:8000/docs` in your browser.

3. Click on an endpoint (e.g. `POST /predict/asin`), click **Try it out**, enter your input, click **Execute**.

4. The response appears below with the prediction result.

`--reload` restarts the server automatically every time you save `main.py` — no need to restart manually during development.

Note: typing the URL directly in the browser sends a GET request — endpoints defined as `POST` will return `{"detail": "Method Not Allowed"}`. Always use `/docs` to test POST endpoints.

---

### `app.py`

Streamlit UI — user-facing frontend that talks to the FastAPI backend.

**How Streamlit works:**

Streamlit runs top to bottom like a regular Python script. Every time the user interacts (clicks a button, types in a box), the entire script reruns from the top. You build the UI by calling Streamlit functions in order:

1. **Create the app instance** — `app = FastAPI()` equivalent is just `import streamlit as st`. No explicit app object needed.
2. **Add a title** — `st.title("Amazon Launch Predictor")`
3. **Create input boxes** — `st.text_input("ASIN")` renders a text box and returns whatever the user typed
4. **Add buttons** — `st.button("Predict")` returns `True` only on the run where the user clicked it
5. **Connect to FastAPI** — inside the button block, call `requests.post(API_URL + "/predict/asin", params={"asin": asin})` — same as any HTTP client
6. **Display results** — `st.metric()` for prominent numbers, `st.write()` for text

**How to run:**
```bash
# Terminal 1 — backend
uvicorn main:app --reload

# Terminal 2 — frontend
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

**User flow:**
- Paste an ASIN or Amazon product URL into the input boxes
- Click **Predict** — app calls FastAPI, which calls Keepa, which runs inference
- Result displayed prominently: launch probability and success/failure label
- Product title and ASIN shown below

**Session state:**

Because Streamlit reruns the entire script on every interaction, variables reset to their default values each time. `st.session_state` is a dictionary that persists across reruns — values stored there survive the rerun.

```python
if "clear_count" not in st.session_state:
    st.session_state.clear_count = 0   # set once, persists across reruns
```

**Clear button:**

Streamlit does not allow you to directly modify a widget's value after it renders — this crashes with `StreamlitAPIException`. The workaround is to change the widget's `key`. When a key changes, Streamlit treats it as a brand new widget and renders it empty.

Streamlit tracks widget values in a dictionary using the key as the lookup:

```
# clear_count = 0 → key is "asin_0" → Streamlit finds "B08XYZ123" → shows it in box
session_state = {"asin_0": "B08XYZ123", "clear_count": 0}

# user clicks Clear → clear_count becomes 1 → key is now "asin_1"
session_state = {"asin_0": "B08XYZ123", "asin_1": ???, "clear_count": 1}
#                                         ↑ never seen before → renders empty
```

`clear_count` is just a number that gets embedded into the key name via f-string:
- `clear_count = 0` → key `"asin_0"` → Streamlit finds old value → shows it
- `clear_count = 1` → key `"asin_1"` → Streamlit has never seen this → renders empty

```python
asin_input = st.text_input("ASIN", key=f"asin_{st.session_state.clear_count}")

if clear_clicked:
    st.session_state.clear_count += 1  # changes the key on next rerun
    st.rerun()                         # force rerun immediately so box clears now
```

`st.rerun()` alone would not work — it reruns with the same `clear_count` so the key stays the same and Streamlit restores the old value. The increment must happen first.

---

### Pipeline Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA COLLECTION                         │
│  ingest.py                                                  │
│  Keepa API ──────────────────► product_launch.db            │
│                                  ├── products table         │
│                                  └── asin_list table        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     FEATURE PIPELINE                        │
│                                                             │
│  load.py                                                    │
│  product_launch.db ──────────► product_launch.parquet       │
│                                (data/raw/)                  │
│                          │                                  │
│                          ▼                                  │
│  preprocessing.py                                           │
│  product_launch.parquet ─────► preprocessed.parquet         │
│                                (data/processed/)            │
│                          │                                  │
│                          ▼                                  │
│  feature_engineering.py                                     │
│  preprocessed.parquet ───────► features_train.parquet  ┐    │
│  (title → TF-IDF)              features_test.parquet   ├── data/processed/
│                                y_train.parquet         │    │
│                                y_test.parquet          ┘    │
│                                preprocessor.pkl ─────────── models/
│                                                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                        │
│                                                             │
│  train.py                                                   │
│  features_train.parquet ─────► lgbm_tfidf_model.pkl          │
│  y_train.parquet               (models/)                    │
│                          │                                  │
│                          ▼                                  │
│  evaluate.py                                                │
│  features_test.parquet ──────► metrics {roc_auc, pr_auc, f1}│
│  y_test.parquet                                             │
│  model.pkl                                                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   INFERENCE PIPELINE                        │
│                                                             │
│  inference.py                                               │
│  new product data ────────────► predicted_probability       │
│  preprocessor.pkl               predicted_label             │
│  lgbm_tfidf_model.pkl           (returned, not saved)       │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 4: Deployment (Google Cloud Run)

### `requirements.txt`

Generate from code imports only — cleaner than `pip freeze` which dumps everything in the environment:

```bash
pip install pipreqs
pipreqs . --force
```

After generating, make two manual edits:
1. **Add `uvicorn`** — pipreqs misses it because it's called from the command line, not imported in code. Check version with `pip show uvicorn` and add e.g. `uvicorn==0.46.0`
2. **Remove `pytest`** — dev-only dependency, not needed in production Docker image

### `Dockerfile`

Containerizes both FastAPI and Streamlit so they run the same way everywhere.

```dockerfile
FROM python:3.11-slim          # base image — slim keeps it small

WORKDIR /app                   # all commands run from /app inside container

COPY requirements.txt .        # copy deps first so Docker caches this layer
RUN pip install --no-cache-dir -r requirements.txt

COPY . .                       # copy all project files including src/serving/model/

EXPOSE 8000 8501               # FastAPI on 8000, Streamlit on 8501

# start both servers in one command — & runs FastAPI in background, Streamlit in foreground 0.0.0.0 allows traffic from all route
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]
```

Note: Cloud Run only exposes one port per service — deploy two separate Cloud Run services from the same image with different start commands.

### created google cloud project and install google cli 

### Deploy to Google Cloud Run

**Step 1 — Install Google Cloud CLI:**
```bash
brew install --cask google-cloud-sdk
```
Installs `gcloud` — the command line tool for managing Google Cloud from your terminal.

**Step 2 — Login and configure project:**
```bash
gcloud auth login                         # opens browser to log in to your Google account
gcloud config set project amazon-launch  # tells gcloud which project to use for all commands
```

**Step 3 — Enable required services:**
```bash
gcloud services enable run.googleapis.com containerregistry.googleapis.com cloudbuild.googleapis.com
```
- `run.googleapis.com` — Cloud Run — runs your container and serves it publicly
- `containerregistry.googleapis.com` — Container Registry — stores your Docker images
- `cloudbuild.googleapis.com` — Cloud Build — builds your Docker image from source code

All three are needed:
```
Cloud Build → builds the image → Container Registry stores it → Cloud Run serves it
```

**Step 3b — Grant Cloud Build storage access:**
```bash
gcloud projects add-iam-policy-binding amazon-launch \
  --member="serviceAccount:{PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/storage.admin"
```
Cloud Build needs permission to read/write to Google Cloud Storage to store build artifacts. Replace `{PROJECT_NUMBER}` with your actual project number (find it in `console.cloud.google.com` → project settings, or run `gcloud projects describe amazon-launch`).

**Step 4 — Build and push Docker image:**
```bash
gcloud builds submit --tag gcr.io/amazon-launch/amazon-predictor
```
Sends your project to Google Cloud Build, builds the Docker image there, and stores it in Container Registry. No need to build locally.

The tag format is `gcr.io/{project-id}/{image-name}`:
- `amazon-launch` — your Google Cloud project ID
- `amazon-predictor` — the name you give the image (you choose this, doesn't have to match project ID)

**Step 5 — Deploy FastAPI backend:**
```bash
gcloud run deploy amazon-predictor-api \
  --image gcr.io/amazon-launch/amazon-predictor \
  --platform managed \
  --allow-unauthenticated \
  --port 8000
```

**Step 6 — Deploy Streamlit frontend:**
```bash
gcloud run deploy amazon-predictor-ui \
  --image gcr.io/amazon-launch/amazon-predictor \
  --platform managed \
  --allow-unauthenticated \
  --port 8501
```

Each deploy gives you a public URL like `https://amazon-predictor-api-xyz.run.app`. Update `API_URL` in `app.py` to point to the FastAPI URL before deploying Streamlit.

**Why Cloud Run over EC2:**
- No server to manage — scales to zero when not in use (no idle charges)
- One command to deploy — no SSH, no security groups, no OS updates
- Free tier: 2 million requests/month














