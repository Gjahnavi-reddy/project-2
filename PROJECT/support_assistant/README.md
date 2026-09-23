# Support Assistant

## Architecture

```text
8 policy docs
   ↓
chunk/document ingestion
   ↓
SentenceTransformer all-MiniLM-L6-v2
   ↓
ChromaDB persistent collection
   ↓
LangGraph classify_intent
   ├── policy_question → retrieve_and_answer → structured response
   └── general_question → direct_answer → structured response
   ↓
FastAPI POST /ask
```

The required mock mode is enabled by default (`MOCK_LLM=1`).

- `build_index.py` handles ingestion, embedding, and ChromaDB storage.
- `main.py` defines the LangGraph state and nodes.
- `retrieve_and_answer` performs real vector retrieval in both modes.
- The mock generation branch uses the top retrieved chunk and deterministic Pydantic output.
- The prompt template includes role/context/task/format/length, a negative constraint, and a few-shot example.

## Run

```bash
python build_index.py
uvicorn main:app --reload
```

Example policy request:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the standard delivery fee?"}'
```

Example general request:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is your favorite color?"}'
```

## Docker

```bash
docker build -t zepto-support .
docker run -p 7860:7860 zepto-support
```

The image builds the vector index during build so the container can serve the endpoint immediately.
