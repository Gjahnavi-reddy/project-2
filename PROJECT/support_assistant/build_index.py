from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
DB = ROOT / "chroma_db"
COLLECTION = "zepto_policies"

def main():
    client = chromadb.PersistentClient(path=str(DB))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    model = SentenceTransformer("all-MiniLM-L6-v2")

    ids, texts, metadatas = [], [], []
    for path in sorted(DOCS.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        ids.append(path.stem)
        texts.append(text)
        metadatas.append({"source": path.name})

    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    print(f"Indexed {len(ids)} documents in {DB}")

if __name__ == "__main__":
    main()
