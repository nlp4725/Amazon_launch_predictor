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

Prepares the clean DataFrame for model training. Joins embeddings first (before row filtering) to preserve index alignment, then filters, labels, splits, and encodes.

**Pipeline order matters:**
```
add_embedding()         ← must run first — index must align with full dataset
filter_and_label()      ← filter rows, create label
drop_and_extract_date() ← select columns, extract month/year
split()                 ← time-based train/test split
fit_and_transform()     ← fit OHE on train only, apply to test
```

| Function | In | Out |
|---|---|---|
| `add_embedding(df, emb_path)` | full preprocessed df | df + 384 `emb_*` columns |
| `filter_and_label(df)` | df with `most_recent_review_time`, `most_recent_review` | filtered df + `label` column (1 = >10 reviews, 0 = failure) |
| `drop_and_extract_date(df)` | labeled df | df with selected columns + `month` (int), `year`, `date` |
| `split(df)` | df with `date` and `label` | `X_train`, `y_train`, `X_test`, `y_test` |
| `fit_and_transform(X_train, X_test, models_dir)` | train/test DataFrames | transformed np.arrays + `models/preprocessor.pkl` |
| `run_feature_engineering(df, output_dir)` | clean df from `run_preprocess()` | `features_train.parquet`, `features_test.parquet`, `y_train.parquet`, `y_test.parquet` |

**Key design decisions:**
- `fit_transform()` on train, `.transform()` only on test — prevents data leakage
- `preprocessor.pkl` saved to `models/` so inference can apply the same transformations to new data without refitting

**Testing logic (`tests/test_feature_engineering.py`):**
- `fake_emb_path` fixture generates a random `.npy` file matching the sample df row count — no real embeddings needed
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
preprocessing: in: df out: df with column extracted and a .parquett file in data/processed add extract information from raw_data columns

test: create a fake df with fake raw_data json and buy_box json and modify the df and  make sure all columns were added

feature_enginerrig: add embedding and lable data drop na save feature as .parqueet and return x_train, y_train etc

label()                  ← success/failure definition
filter_review_time()     ← keep >= 180 days
split()                  ← train/test by date
encode_categoricals()    ← OHE for cat, seller (fit on train only)
add_embedding()          ← join 384 emb_ columns
run_feature_engineering() ← chains all, saves features_train.parquet, features_test.parquet, ohe_encoder.pkl


how to save model

```bash
from joblib import dump
dump(preprocessor, Path("models") / "preprocessor.pkl")
```

why differenet Path/out_dir/mkdir?

DATA_PATH=Path() fixed path to pass to functon
out_dir= DATA_PATH (DEFUALT) OR tmp_path for testing
mkdir() Make sure it is created for the first time




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
│  title_embedding_all.npy       features_test.parquet   ├── data/processed/
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
│  features_train.parquet ─────► model.pkl                    │
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
│  model.pkl                      (returned, not saved)       │
└─────────────────────────────────────────────────────────────┘
```














