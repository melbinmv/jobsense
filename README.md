# JobSense

Real-time job market intelligence powered by a custom RAG pipeline + MLflow.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python pipeline.py index   # build the vector index
python pipeline.py query   # ask questions
```

## MLflow UI

```bash
mlflow ui
# open http://localhost:5000
```
