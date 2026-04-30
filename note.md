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






