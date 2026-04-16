import os, glob, pickle, faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Cloud LLMOps RAG", layout="wide")
st.title("Cloud LLMOps: Single-RAG vs Multi-RAG")

INDEX_DIR = "rag/indexes"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def load():
    model = SentenceTransformer(MODEL)
    sidx = faiss.read_index(f"{INDEX_DIR}/single.index")
    smeta = pickle.load(open(f"{INDEX_DIR}/single_meta.pkl","rb"))
    multi={}
    for mp in glob.glob(f"{INDEX_DIR}/*_meta.pkl"):
        if mp.endswith("single_meta.pkl"): continue
        src=os.path.basename(mp).replace("_meta.pkl","")
        ip=f"{INDEX_DIR}/{src}.index"
        if os.path.exists(ip):
            multi[src]=(faiss.read_index(ip), pickle.load(open(mp,"rb")))
    return model,sidx,smeta,multi

model,sidx,smeta,multi = load()

def single(q,k=5):
    qv=model.encode([q], normalize_embeddings=True).astype("float32")
    D,I=sidx.search(qv,k)
    out=[]
    for s,i in zip(D[0],I[0]):
        r=smeta[i]
        out.append((float(s), r.get("source",""), r.get("url","")))
    return out

def multi_search(q,top_sources=3,k_each=3):
    qv=model.encode([q], normalize_embeddings=True).astype("float32")
    sc=[]
    for src,(idx,meta) in multi.items():
        D,I=idx.search(qv,1)
        sc.append((float(D[0][0]),src))
    sc.sort(reverse=True)
    routed=[x[1] for x in sc[:top_sources]]

    out=[]
    for src in routed:
        idx,meta=multi[src]
        D,I=idx.search(qv,k_each)
        for s,i in zip(D[0],I[0]):
            r=meta[i]
            out.append((float(s),src,r.get("url","")))
    out.sort(key=lambda x:x[0], reverse=True)
    return routed, out[:5]

q = st.text_input("Enter your DevOps/Cloud question")
if st.button("Run") and q.strip():
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Single-RAG")
        for s,src,url in single(q):
            st.write(f"- **{src}** ({s:.4f})  \n{url}")
    with c2:
        st.subheader("Multi-RAG")
        routed,res = multi_search(q)
        st.write("Routed:", routed)
        for s,src,url in res:
            st.write(f"- **{src}** ({s:.4f})  \n{url}")
