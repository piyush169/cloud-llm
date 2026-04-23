import pickle
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "rag/indexes/single.index"
META_PATH = "rag/indexes/single_meta.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)
meta = pickle.load(open(META_PATH, "rb"))

def retrieve(query, k=5):
    qv = model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, k)
    out = []
    for score, idx in zip(D[0], I[0]):
        r = meta[idx]
        out.append({
            "score": float(score),
            "source": r.get("source", "unknown"),
            "url": r.get("url", ""),
            "text": (r.get("chunk_text", "")[:220] + "..."),
        })
    return out

if __name__ == "__main__":
    q = input("Query: ").strip()
    res = retrieve(q, 5)
    print("\n=== Single-RAG Top 5 ===")
    for i, r in enumerate(res, 1):
        print(f"{i}. [{r['source']}] score={r['score']:.4f}")
        print(f"   {r['url']}")
        print(f"   {r['text']}\n")
