from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, TypedDict

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END

ROOT = Path(__file__).resolve().parent
DB = ROOT / "chroma_db"
COLLECTION = "zepto_policies"

KEYWORDS = [
    "delivery", "return", "refund", "membership", "tracking",
    "cancel", "gift card", "support hours"
]

PROMPT_TEMPLATE = """ROLE:
You are a Zepto policy support assistant.

CONTEXT:
Use only the policy excerpts supplied below.

TASK:
Answer the user's policy question using the retrieved context.

FORMAT:
Return a concise answer and identify the source document IDs.

LENGTH:
Keep the answer short and directly relevant.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.

FEW-SHOT EXAMPLE:
User: What is the delivery fee below INR 149?
Context: Orders below INR 149 incur a flat INR 25 delivery fee.
Answer: Orders below INR 149 incur a flat INR 25 delivery fee.
"""

class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0, le=1)

class State(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    answer: str
    sources: list[str]
    confidence: float

def mock_enabled() -> bool:
    return os.getenv("MOCK_LLM", "1") != "0"

def classify(query: str) -> str:
    q = query.lower()
    return "policy_question" if any(k in q for k in KEYWORDS) else "general_question"

def retrieve(query: str):
    client = chromadb.PersistentClient(path=str(DB))
    collection = client.get_collection(COLLECTION)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode([query], normalize_embeddings=True).tolist()
    return collection.query(
        query_embeddings=embedding,
        n_results=min(3, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

def classify_intent(state: State) -> State:
    # Required graded baseline: deterministic heuristic. Real-LLM extension can
    # be inserted here when MOCK_LLM=0 without changing graph routing.
    state["intent"] = classify(state["query"])
    return state

def retrieve_and_answer(state: State) -> State:
    result = retrieve(state["query"])
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    if not docs:
        state["answer"] = "No matching policy context was retrieved."
        state["sources"] = []
        state["confidence"] = 0.0
        return state

    top = docs[0]
    sources = [m.get("source", "") for m in metas]
    if mock_enabled():
        state["answer"] = f"Based on the retrieved context: {top[:200]}"
        state["sources"] = sources
        state["confidence"] = 1.0
        return state

    # Optional real-LLM extension point. The required project path is MOCK_LLM=1.
    state["answer"] = f"Based on the retrieved context: {top[:200]}"
    state["sources"] = sources
    state["confidence"] = 0.8
    return state

def direct_answer(state: State) -> State:
    state["answer"] = "I can only answer questions about Zepto policies right now."
    state["sources"] = []
    state["confidence"] = 1.0
    return state

def route(state: State) -> str:
    return state["intent"]

graph = StateGraph(State)
graph.add_node("classify_intent", classify_intent)
graph.add_node("retrieve_and_answer", retrieve_and_answer)
graph.add_node("direct_answer", direct_answer)
graph.add_edge(START, "classify_intent")
graph.add_conditional_edges(
    "classify_intent",
    route,
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer",
    },
)
graph.add_edge("retrieve_and_answer", END)
graph.add_edge("direct_answer", END)
app_graph = graph.compile()

app = FastAPI(title="Zepto Policy Support Assistant")

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = app_graph.invoke({
        "query": request.query,
        "intent": "",
        "answer": "",
        "sources": [],
        "confidence": 0.0,
    })
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
    )
