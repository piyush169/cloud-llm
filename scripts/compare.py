import os, glob, pickle, csv, time, faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = "rag/indexes"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT = "rag/outputs/comparison_report.csv"
os.makedirs("rag/outputs", exist_ok=True)

queries = [q.strip() for q in open("queries.txt", encoding="utf-8") if q.strip()]
model = SentenceTransformer(MODEL)

sidx = faiss.read_index(f"{INDEX_DIR}/single.index")
smeta = pickle.load(open(f"{INDEX_DIR}/single_meta.pkl", "rb"))

multi = {}
for mp in glob.glob(f"{INDEX_DIR}/*_meta.pkl"):
    if mp.endswith("single_meta.pkl"): continue
    src = os.path.basename(mp).replace("_meta.pkl", "")
    ip = f"{INDEX_DIR}/{src}.index"
    if os.path.exists(ip):
        multi[src] = (faiss.read_index(ip), pickle.load(open(mp, "rb")))

def single_top_sources(q):
    qv = model.encode([q], normalize_embeddings=True).astype("float32")
    D,I = sidx.search(qv, 5)
    return [smeta[i].get("source","") for i in I[0]]

def routed_sources(q):
    qv = model.encode([q], normalize_embeddings=True).astype("float32")
    scores=[]
    for src,(idx,_) in multi.items():
        D,I = idx.search(qv,1)
        scores.append((float(D[0][0]),src))
    scores.sort(reverse=True)
    return [x[1] for x in scores[:3]]

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["query","single_top5_sources","multi_routed_sources","single_ms","multi_ms"])
    for q in queries:
        t1=time.time(); s=single_top_sources(q); t2=time.time()
        m=routed_sources(q); t3=time.time()
        w.writerow([q, "|".join(s), "|".join(m), int((t2-t1)*1000), int((t3-t2)*1000)])

print("saved:", OUT)
