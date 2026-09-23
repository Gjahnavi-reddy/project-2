# AI/ML Capstone Project — Zepto Data & AI Platform

This repository implements the three modules required by the supplied capstone brief:

1. `data_pipeline/` — scrape, clean, convert, normalize into SQLite, and query.
2. `analytics/` — Titanic EDA, preprocessing, classification, imbalance handling, tuning, regression, and model persistence.
3. `support_assistant/` — offline-first policy RAG with Sentence Transformers, ChromaDB, LangGraph, Pydantic, FastAPI, and Docker.

## Important submission notes

- The assignment requires the root folders to be named exactly `data_pipeline`, `analytics`, and `support_assistant`.
- The supplied PDF explicitly says the written deliverables belong in Markdown/README/notebook cells; the PDF itself is not a required repository deliverable.
- The required currency baseline is **1 GBP = 105.50 INR**.
- The analytics module loads the Titanic dataset from `seaborn` once, immediately writes `titanic.csv`, and then uses the CSV for later work.
- The support assistant defaults to `MOCK_LLM=1`, so the graded path does not require an API key.
- Do not commit API keys, `.env` files, or generated vector-store data.

## Setup

Use one environment per module or one environment for the whole repository.

```bash
pip install -r data_pipeline/requirements.txt
pip install -r analytics/requirements.txt
pip install -r support_assistant/requirements.txt
```

## Run

### Module 1

```bash
python data_pipeline/scrape_pipeline.py
```

This creates the cleaned CSV, SQLite database, and `sql_results.txt`.

### Module 2

The first run needs internet access for `sns.load_dataset("titanic")`. It immediately creates the offline fallback:

```text
analytics/titanic.csv
```

Then run:

```bash
python analytics/01_eda.py
python analytics/02_modeling.py
python analytics/reload_model.py
```

### Module 3

Build the local vector index:

```bash
python support_assistant/build_index.py
```

Run the API:

```bash
uvicorn support_assistant.main:app --reload
```

Test:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the standard delivery fee?"}'
```

The default mock path is deterministic and requires no LLM provider.

## Git workflow required by the brief

The supplied brief requires at least one feature branch that is committed at least twice and then merged into `main`.

Example:

```bash
git checkout -b feature/capstone-completion
git add .
git commit -m "Add data pipeline"
git commit -m "Add analytics and support assistant"
git checkout main
git merge --no-ff feature/capstone-completion -m "Merge capstone feature"
git log --graph --all --oneline
```

Make sure the final GitHub history actually shows the branch and merge. Do not create fake history.

## Academic integrity

This package is a corrected implementation scaffold. Before submission, run it yourself, inspect the generated outputs, understand the code, and rewrite explanations/results in your own words where your course requires original work.
