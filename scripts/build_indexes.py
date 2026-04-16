import os
import glob
import json
import pickle
from collections import defaultdict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------- paths for YOUR folder layout ----------
# run from: ~/rag
INPUT_GLOB = "data/**/docs_chunks.jsonl"
OUT_DIR = "rag/indexes"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384
MIN_CHARS = 80

os.makedirs(OUT_DIR, exist_ok=True)

files = glob.glob(INPUT_GLOB, recursive=True)
print(f"[INFO] chunk files found: {len(files)}")
for f in files[:20]:
    print(" -", f)

if not files:
    raise RuntimeError("No docs_chunks.jsonl files found under data/**")

rows = []
for fp in files:
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue

            txt = (r.get("chunk_text") or "").strip()
            if len(txt) < MIN_CHARS:
                continue

            # ensure source exists
            if "source" not in r or not r["source"]:
                # infer source from path: data/<source>/.../docs_chunks.jsonl
                parts = fp.replace("\\", "/").split("/")
                if len(parts) > 1:
                    r["source"] = parts[1]
                else:
                    r["source"] = "unknown"

            rows.append(r)

print(f"[INFO] usable chunk rows: {len(rows)}")
if not rows:
    raise RuntimeError("No usable rows after filtering. Check chunk_text field in files.")

texts = [r["chunk_text"] for r in rows]

print(f"[INFO] loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

print("[INFO] encoding embeddings...")
emb = model.encode(
    texts,
    batch_size=128,
    show_progress_bar=True,
    normalize_embeddings=True
)
emb = np.asarray(emb, dtype="float32")

if emb.shape[1] != DIM:
    print(f"[WARN] model dim is {emb.shape[1]} not {DIM}. Using model dim.")
    DIM = emb.shape[1]

# ---------- single index ----------
single_idx = faiss.IndexFlatIP(DIM)
single_idx.add(emb)
single_idx_path = os.path.join(OUT_DIR, "single.index")
single_meta_path = os.path.join(OUT_DIR, "single_meta.pkl")

faiss.write_index(single_idx, single_idx_path)
with open(single_meta_path, "wb") as f:
    pickle.dump(rows, f)

print(f"[OK] single index saved: {single_idx_path} (vectors={single_idx.ntotal})")
print(f"[OK] single meta saved : {single_meta_path}")

# ---------- multi index (per source) ----------
grouped = defaultdict(list)
for r, v in zip(rows, emb):
    src = (r.get("source") or "unknown").strip()
    grouped[src].append((r, v))

for src, items in grouped.items():
    safe_src = src.replace("/", "_").replace(" ", "_")
    meta = [x[0] for x in items]
    vecs = np.asarray([x[1] for x in items], dtype="float32")

    idx = faiss.IndexFlatIP(DIM)
    idx.add(vecs)

    idx_path = os.path.join(OUT_DIR, f"{safe_src}.index")
    meta_path = os.path.join(OUT_DIR, f"{safe_src}_meta.pkl")

    faiss.write_index(idx, idx_path)
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"[OK] {safe_src}: vectors={idx.ntotal}")

print("\n[DONE] all indexes created successfully.")
