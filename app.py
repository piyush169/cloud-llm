import os
import re
import glob
import pickle
import time
from typing import List, Dict, Tuple

import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
import ollama

# ---------------------------
# Config
# ---------------------------
INDEX_DIR = "rag/indexes"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")

TOP_K_SINGLE = 5
TOP_SOURCES_MULTI = 3
TOP_K_EACH_MULTI = 3
MAX_CONTEXT_CHARS = 700

st.set_page_config(page_title="Cloud LLMOps: Single-RAG vs Multi-RAG", layout="wide")
st.title("Cloud LLMOps: Single-RAG vs Multi-RAG (with Answers + Metrics + Progress)")

# ---------------------------
# Domain rules (weak labels)
# ---------------------------
DOMAIN_KEYWORDS = {
    "jenkins": ["jenkins", "pipeline", "declarative pipeline", "groovy"],
    "docker": ["docker", "container", "image", "buildx", "multi-stage"],
    "kubernetes": ["kubernetes", "k8s", "pod", "deployment", "liveness probe"],
    "argocd": ["argocd", "gitops", "sync policy", "autosync"],
    "gitlab_cicd": ["gitlab", "gitlab ci", ".gitlab-ci", "stages", "runner"],
    "terraform": ["terraform", "tfstate", "state", "backend", "plan", "apply"],
    "aws_s3": ["s3", "bucket policy", "object storage"],
    "aws_iam": ["iam", "role", "trust policy", "assume role", "sts"],
    "gcp": ["gcp", "gke", "google cloud", "workload identity", "cloud run"],
}

DOMAIN_CONCEPTS = {
    "jenkins": ["pipeline", "agent", "stages", "stage", "steps", "jenkinsfile"],
    "docker": ["dockerfile", "build", "stage", "image"],
    "kubernetes": ["probe", "liveness", "readiness", "pod", "deployment"],
    "argocd": ["application", "sync", "automated", "selfHeal", "prune"],
    "gitlab_cicd": ["stages", "jobs", ".gitlab-ci.yml", "runner"],
    "terraform": ["backend", "state", "init", "plan", "apply"],
    "aws_s3": ["bucket", "policy", "principal", "action", "resource"],
    "aws_iam": ["role", "trust", "principal", "assume role", "policy"],
    "gcp": ["cluster", "gke", "namespace", "deployment", "service account"],
}

# ---------------------------
# Utility
# ---------------------------
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_\.:-]+", normalize_text(s))

def guess_expected_domains(query: str) -> List[str]:
    q = normalize_text(query)
    hits = []
    for domain, kws in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in q)
        if score > 0:
            hits.append((score, domain))
    hits.sort(reverse=True)
    if not hits:
        return list(DOMAIN_KEYWORDS.keys())  # fallback
    best = hits[0][0]
    return [d for s, d in hits if s == best]

# ---------------------------
# Startup checks
# ---------------------------
def check_required_files() -> Tuple[bool, str]:
    if not os.path.exists(f"{INDEX_DIR}/single.index"):
        return False, f"Missing {INDEX_DIR}/single.index"
    if not os.path.exists(f"{INDEX_DIR}/single_meta.pkl"):
        return False, f"Missing {INDEX_DIR}/single_meta.pkl"
    return True, "ok"

def check_ollama_available() -> Tuple[bool, str]:
    try:
        ollama.list()
        return True, f"Ollama connected ✅ | model: {OLLAMA_MODEL}"
    except Exception as e:
        return False, f"Ollama not reachable ❌ ({e})"

@st.cache_resource
def load_resources():
    emb_model = SentenceTransformer(EMBED_MODEL)

    single_index = faiss.read_index(f"{INDEX_DIR}/single.index")
    with open(f"{INDEX_DIR}/single_meta.pkl", "rb") as f:
        single_meta = pickle.load(f)

    multi = {}
    for meta_path in glob.glob(f"{INDEX_DIR}/*_meta.pkl"):
        if meta_path.endswith("single_meta.pkl"):
            continue
        src = os.path.basename(meta_path).replace("_meta.pkl", "")
        idx_path = f"{INDEX_DIR}/{src}.index"
        if os.path.exists(idx_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            multi[src] = {"index": faiss.read_index(idx_path), "meta": meta}

    return emb_model, single_index, single_meta, multi

# ---------------------------
# Retrieval
# ---------------------------
def embed_query(emb_model, query: str):
    return emb_model.encode([query], normalize_embeddings=True).astype("float32")

def single_retrieve(emb_model, single_index, single_meta, query: str, k: int = TOP_K_SINGLE) -> List[Dict]:
    qv = embed_query(emb_model, query)
    D, I = single_index.search(qv, k)
    out = []
    for score, idx in zip(D[0], I[0]):
        r = single_meta[idx]
        out.append({
            "score": float(score),
            "source": r.get("source", "unknown"),
            "url": r.get("url", ""),
            "text": (r.get("chunk_text", "") or "").strip(),
        })
    return out

def route_sources_multi(emb_model, multi, query: str, top_n: int = TOP_SOURCES_MULTI) -> List[str]:
    if not multi:
        return []
    qv = embed_query(emb_model, query)
    scores = []
    for src, obj in multi.items():
        D, _ = obj["index"].search(qv, 1)
        scores.append((float(D[0][0]), src))
    scores.sort(reverse=True)
    return [x[1] for x in scores[:top_n]]

def multi_retrieve(emb_model, multi, query: str, top_sources: int = TOP_SOURCES_MULTI, k_each: int = TOP_K_EACH_MULTI):
    if not multi:
        return [], []
    qv = embed_query(emb_model, query)
    routed = route_sources_multi(emb_model, multi, query, top_sources)
    out = []
    for src in routed:
        obj = multi[src]
        D, I = obj["index"].search(qv, k_each)
        for score, idx in zip(D[0], I[0]):
            r = obj["meta"][idx]
            out.append({
                "score": float(score),
                "source": src,
                "url": r.get("url", ""),
                "text": (r.get("chunk_text", "") or "").strip(),
            })
    out.sort(key=lambda x: x["score"], reverse=True)
    return routed, out[:TOP_K_SINGLE]

# ---------------------------
# Rerank (light lexical)
# ---------------------------
def lexical_rerank(query: str, results: List[Dict]) -> List[Dict]:
    q_tokens = set(tokenize(query))
    reranked = []
    for r in results:
        t_tokens = set(tokenize(r["text"][:1200]))
        overlap = len(q_tokens & t_tokens) / max(1, len(q_tokens))
        final_score = 0.75 * float(r["score"]) + 0.25 * overlap
        rr = dict(r)
        rr["rerank_score"] = final_score
        reranked.append(rr)
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked

# ---------------------------
# Metrics
# ---------------------------
def retrieval_metrics(results: List[Dict], relevant_domains: List[str], k: int = TOP_K_SINGLE) -> Dict[str, float]:
    top = results[:k]
    if not top:
        return {"precision@k": 0.0, "recall@k": 0.0, "mrr": 0.0}

    rel = set(relevant_domains)
    hits = [1 if r["source"] in rel else 0 for r in top]

    precision = sum(hits) / k
    recall = sum(hits) / k  # weak-label online proxy
    rr = 0.0
    for i, h in enumerate(hits, start=1):
        if h == 1:
            rr = 1.0 / i
            break
    return {"precision@k": precision, "recall@k": recall, "mrr": rr}

def faithfulness_score(answer: str, contexts: List[Dict]) -> float:
    if not answer.strip() or not contexts:
        return 0.0
    ctx = " ".join((c.get("text", "")[:1400] for c in contexts))
    ctx_tok = set(tokenize(ctx))
    sents = [s.strip() for s in re.split(r"[.\n]+", answer) if s.strip()]
    if not sents:
        return 0.0
    supported = 0
    for s in sents:
        stoks = set(tokenize(s))
        if not stoks:
            continue
        ov = len(stoks & ctx_tok) / max(1, len(stoks))
        if ov >= 0.35:
            supported += 1
    return supported / max(1, len(sents))

def proxy_accuracy(answer: str, relevant_domains: List[str]) -> float:
    a = normalize_text(answer)
    if not a:
        return 0.0
    pool = []
    for d in relevant_domains:
        pool.extend(DOMAIN_CONCEPTS.get(d, []))
    if not pool:
        return 0.0
    hits = sum(1 for c in pool if c.lower() in a)
    return min(1.0, hits / max(3, int(len(pool) * 0.4)))

# ---------------------------
# Generation
# ---------------------------
def build_prompt(question: str, contexts: List[Dict]) -> str:
    lines = []
    for c in contexts[:5]:
        txt = c["text"][:MAX_CONTEXT_CHARS].replace("\n", " ")
        lines.append(f"[{c['source']}] {txt}")
    context = "\n\n".join(lines)

    return f"""You are a Cloud/DevOps assistant.
Use ONLY the context below.
If context is insufficient, say: "I don't have enough context from retrieved docs."

Question:
{question}

Context:
{context}

Output:
1) Direct answer
2) Practical steps
3) Sources used (source names only)
"""

def generate_answer_ollama(question: str, contexts: List[Dict], model: str = OLLAMA_MODEL) -> str:
    prompt = build_prompt(question, contexts)
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        return f"Local LLM error: {e}"

# ---------------------------
# Render helpers
# ---------------------------
def render_context_block(title: str, results: List[Dict]):
    st.markdown(f"### {title}")
    if not results:
        st.info("No retrieval results.")
        return
    for i, r in enumerate(results, 1):
        key = "rerank_score" if "rerank_score" in r else "score"
        st.markdown(f"**{i}. {r['source']}** ({key}: {r[key]:.4f})")
        if r["url"]:
            st.markdown(r["url"])
        preview = r["text"][:220].replace("\n", " ")
        st.caption(preview + ("..." if len(r["text"]) > 220 else ""))

def render_metrics_table(title: str, ret: Dict[str, float], ans: Dict[str, float]):
    st.markdown(f"### {title}")
    st.table({
        "Metric": ["Precision@K", "Recall@K", "MRR", "Accuracy (proxy)", "Faithfulness"],
        "Value": [
            f"{ret['precision@k']:.3f}",
            f"{ret['recall@k']:.3f}",
            f"{ret['mrr']:.3f}",
            f"{ans['accuracy']:.3f}",
            f"{ans['faithfulness']:.3f}",
        ],
    })

# ---------------------------
# Main UI
# ---------------------------
ok, msg = check_required_files()
if not ok:
    st.error(msg)
    st.stop()

ollama_ok, ollama_msg = check_ollama_available()
if ollama_ok:
    st.success(ollama_msg)
else:
    st.error(ollama_msg)
    st.info("Start Ollama: `ollama serve` and pull model: `ollama pull phi3:mini`")

use_rerank = st.toggle("Enable reranking impact analysis", value=True)

with st.spinner("Loading embeddings + indexes..."):
    emb_model, single_index, single_meta, multi = load_resources()

query = st.text_input("Enter your DevOps/Cloud question", "How to create Jenkins declarative pipeline?")

# Progress UI placeholders
progress_title = st.empty()
progress_bar = st.progress(0)
progress_log = st.empty()

if st.button("Run"):
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    logs = []

    def step(pct: int, title: str, log_line: str):
        progress_title.info(title)
        progress_bar.progress(pct)
        logs.append(f"• {log_line}")
        progress_log.markdown("\n".join(logs))

    t0 = time.time()
    step(5, "Initializing run...", "Run started")

    relevant_domains = guess_expected_domains(query)
    step(12, "Detecting expected domain...", f"Expected relevant domains: {', '.join(relevant_domains)}")

    # Retrieval
    t1 = time.time()
    step(20, "Running Single-RAG retrieval...", "Searching global FAISS index")
    single_results = single_retrieve(emb_model, single_index, single_meta, query, TOP_K_SINGLE)

    step(32, "Running Multi-RAG routing...", "Routing query across domain indexes")
    routed, multi_results = multi_retrieve(emb_model, multi, query, TOP_SOURCES_MULTI, TOP_K_EACH_MULTI)

    single_before = single_results[:]
    multi_before = multi_results[:]
    t2 = time.time()

    # Rerank
    if use_rerank:
        step(42, "Applying reranking...", "Lexical-semantic reranking on retrieved chunks")
        single_results = lexical_rerank(query, single_results)[:TOP_K_SINGLE]
        multi_results = lexical_rerank(query, multi_results)[:TOP_K_SINGLE]
    else:
        step(42, "Skipping reranking...", "Reranking disabled")

    # Generation
    if ollama_ok:
        step(56, "Generating Single-RAG answer...", f"Calling Ollama model: {OLLAMA_MODEL}")
        single_answer = generate_answer_ollama(query, single_results)

        step(72, "Generating Multi-RAG answer...", f"Calling Ollama model: {OLLAMA_MODEL}")
        multi_answer = generate_answer_ollama(query, multi_results)
    else:
        single_answer = "Ollama unavailable."
        multi_answer = "Ollama unavailable."

    t3 = time.time()

    # Metrics
    step(84, "Computing retrieval metrics...", "Precision@K, Recall@K, MRR")
    s_ret = retrieval_metrics(single_results, relevant_domains, TOP_K_SINGLE)
    m_ret = retrieval_metrics(multi_results, relevant_domains, TOP_K_SINGLE)
    s_before = retrieval_metrics(single_before, relevant_domains, TOP_K_SINGLE)
    m_before = retrieval_metrics(multi_before, relevant_domains, TOP_K_SINGLE)

    step(92, "Computing answer quality metrics...", "Accuracy (proxy), Faithfulness")
    s_ans = {
        "accuracy": proxy_accuracy(single_answer, relevant_domains),
        "faithfulness": faithfulness_score(single_answer, single_results),
    }
    m_ans = {
        "accuracy": proxy_accuracy(multi_answer, relevant_domains),
        "faithfulness": faithfulness_score(multi_answer, multi_results),
    }

    t4 = time.time()
    step(100, "Completed ✅", f"Total latency: {(t4 - t0):.2f}s | Retrieval: {(t2 - t1):.2f}s | Generation: {(t3 - t2):.2f}s")

    # Results layout
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Single-RAG")
        st.markdown("### Answer")
        st.write(single_answer)
        render_context_block("Retrieved Context", single_results)
        render_metrics_table("Metrics (Single-RAG)", s_ret, s_ans)

    with col2:
        st.subheader("Multi-RAG")
        st.markdown("**Routed sources:** " + (", ".join(routed) if routed else "None"))
        st.markdown("### Answer")
        st.write(multi_answer)
        render_context_block("Retrieved Context", multi_results)
        render_metrics_table("Metrics (Multi-RAG)", m_ret, m_ans)

    st.markdown("---")
    st.markdown("## Comparison Summary")
    st.write(f"**Expected relevant domain(s):** {', '.join(relevant_domains)}")
    st.table({
        "Metric": ["Precision@K", "Recall@K", "MRR", "Accuracy (proxy)", "Faithfulness"],
        "Single-RAG": [
            f"{s_ret['precision@k']:.3f}",
            f"{s_ret['recall@k']:.3f}",
            f"{s_ret['mrr']:.3f}",
            f"{s_ans['accuracy']:.3f}",
            f"{s_ans['faithfulness']:.3f}",
        ],
        "Multi-RAG": [
            f"{m_ret['precision@k']:.3f}",
            f"{m_ret['recall@k']:.3f}",
            f"{m_ret['mrr']:.3f}",
            f"{m_ans['accuracy']:.3f}",
            f"{m_ans['faithfulness']:.3f}",
        ],
    })

    if use_rerank:
        st.markdown("## Reranking Impact")
        st.table({
            "Metric": ["Precision@K", "Recall@K", "MRR"],
            "Single Before": [f"{s_before['precision@k']:.3f}", f"{s_before['recall@k']:.3f}", f"{s_before['mrr']:.3f}"],
            "Single After": [f"{s_ret['precision@k']:.3f}", f"{s_ret['recall@k']:.3f}", f"{s_ret['mrr']:.3f}"],
            "Multi Before": [f"{m_before['precision@k']:.3f}", f"{m_before['recall@k']:.3f}", f"{m_before['mrr']:.3f}"],
            "Multi After": [f"{m_ret['precision@k']:.3f}", f"{m_ret['recall@k']:.3f}", f"{m_ret['mrr']:.3f}"],
        })

st.markdown("---")
st.caption("Metrics are online proxy metrics for live queries. For strict benchmarking, use labeled offline datasets.")
