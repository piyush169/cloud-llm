import os
import glob
import pickle
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = "rag/indexes"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

# load per-source indexes
sources = {}
for meta_path in glob.glob(f"{INDEX_DIR}/*_meta.pkl"):
    if meta_path.endswith("single_meta.pkl"):
        continue
    src = os.path.basename(meta_path).replace("_meta.pkl", "")
    idx_path = f"{INDEX_DIR}/{src}.index"
    if os.path.exists(idx_path):
        sources[src] = {
            "index": faiss.read_index(idx_path),
            "meta": pickle.load(open(meta_path, "rb"))
        }

def route_sources(query, top_n=3):
    qv = model.encode([query], normalize_embeddings=True).astype("float32")
    scores = []
    for src, obj in sources.items():
        D, I = obj["index"].search(qv, 1)
        scores.append((float(D[0][0]), src))
    scores.sort(reverse=True)
    return [s[1] for s in scores[:top_n]]

def retrieve(query, top_sources=3, k_each=3):
    qv = model.encode([query], normalize_embeddings=True).astype("float32")
    selected = route_sources(query, top_sources)

    results = []
    for src in selected:
        obj = sources[src]
        D, I = obj["index"].search(qv, k_each)
        for score, idx in zip(D[0], I[0]):
            r = obj["meta"][idx]
            results.append({
                "score": float(score),
                "source": src,
                "url": r.get("url", ""),
                "text": (r.get("chunk_text", "")[:220] + "..."),
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return selected, results[:5]

if __name__ == "__main__":
    q = input("Query: ").strip()
    routed, res = retrieve(q, top_sources=3, k_each=3)

    print("\nRouted sources:", routed)
    print("\n=== Multi-RAG Top 5 ===")
    for i, r in enumerate(res, 1):
        print(f"{i}. [{r['source']}] score={r['score']:.4f}")
        print(f"   {r['url']}")
        print(f"   {r['text']}\n")
