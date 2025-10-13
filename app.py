# app.py
import os
import pathlib
from typing import Optional, List
import datetime as dt
import streamlit as st
from dataclasses import dataclass
from time import time
from typing import Dict

# --- loaders ---
import docx2txt
from bs4 import BeautifulSoup
import markdown as md
from pypdf import PdfReader

# --- vector + llm ---
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient


# ------------------- Config -------------------
API_KEY = os.environ.get("HUGGING_FACE_KEY")
DATA_DIR = pathlib.Path("./data")
CHROMA_DIR = "./chromaDB"
COLLECTION = "real_docs"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"   # fast, permissive
SYSTEM_PROMPT = """You are a research scholar. Your job is to come up with a novel research topic that aligns with the query question.
"""


#"""You are a careful, grounded assistant.
# Use ONLY the provided context to answer the question.
# If the answer is not supported by the context, say you don't know.
# Include brief citations in the form (title → file path)."""


# ------------------- Data model -------------------
@dataclass
class RawDoc:
   doc_id: str
   title: str
   text: str
   path: str
   mtime_iso: str
   filetype: str

# ------------------- Utilities: loaders -------------------
def read_text_txt(path: pathlib.Path) -> str:
   return path.read_text(encoding="utf-8", errors="ignore")

def read_text_pdf(path: pathlib.Path) -> str:
   reader = PdfReader(str(path))
   pages = []
   for p in reader.pages:
       pages.append(p.extract_text() or "")
   return "\n".join(pages)

def read_text_docx(path: pathlib.Path) -> str:
   return docx2txt.process(str(path)) or ""

def html_to_text(html: str) -> str:
   soup = BeautifulSoup(html, "html.parser")
   for bad in soup(["script", "style", "noscript"]):
       bad.extract()
   return soup.get_text(separator=" ", strip=True)

def read_text_html(path: pathlib.Path) -> str:
   return html_to_text(path.read_text(encoding="utf-8", errors="ignore"))

def read_text_md(path: pathlib.Path) -> str:
   html = md.markdown(path.read_text(encoding="utf-8", errors="ignore"))
   return html_to_text(html)

def load_file(path: pathlib.Path) -> Optional[RawDoc]:
   if not path.is_file():
       return None
   ext = path.suffix.lower()
   try:
       if ext == ".pdf":
           text = read_text_pdf(path)
           ftype = "pdf"
       elif ext == ".docx":
           text = read_text_docx(path)
           ftype = "docx"
       elif ext in (".html", ".htm"):
           text = read_text_html(path)
           ftype = "html"
       elif ext in (".md", ".markdown"):
           text = read_text_md(path)
           ftype = "markdown"
       elif ext in (".txt",):
           text = read_text_txt(path)
           ftype = "text"
       else:
           return None  # unsupported
   except Exception as e:
       st.warning(f"[loader] Skipping {path.name} due to error: {e}")
       return None
  
   text = " ".join(text.split())
   if not text.strip():
       return None

   mtime = dt.datetime.fromtimestamp(path.stat().st_mtime)
   title = path.stem.replace("_", " ").strip() or path.name
   return RawDoc(
       doc_id=str(path.resolve()),
       title=title,
       text=text,
       path=str(path.resolve()),
       mtime_iso=mtime.isoformat(),
       filetype=ftype,
   )

def walk_data_dir(data_dir: pathlib.Path) -> List[RawDoc]:
   docs: List[RawDoc] = []
   for p in data_dir.rglob("*"):
       rd = load_file(p)
       if rd:
           docs.append(rd)
   return docs

# -------------------- Chunk: overlapped chunk ---------------------
def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
   chunks = []
   pre_chunks = text.split("\n\n")
   for t in pre_chunks:
       t = " ".join(t.split())
       start = 0
       n = len(t)
       while start < n:
           end = min(n, start + max_chars)
           chunks.append(t[start:end])
           if end == n: break
           start = max(0, end - overlap)
   return chunks

# ------------------- Embedding + Chroma -------------------
@st.cache_resource
def get_embedder():
   return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def get_collection():
   client = chromadb.PersistentClient(
       path=CHROMA_DIR,
       settings=Settings(anonymized_telemetry=False),
   )
   try:
       return client.get_collection(COLLECTION)
   except Exception:
       return client.create_collection(COLLECTION)

def index_docs(docs: List[RawDoc], embedder, max_chars: int, overlap: int):
   coll = get_collection()
   ids, docs_text, metas = [], [], []

   for rd in docs:
       for i, ch in enumerate(chunk_text(rd.text, max_chars=max_chars, overlap=overlap)):
           ids.append(f"{rd.doc_id}::chunk::{i}")
           docs_text.append(ch)
           metas.append({
               "title": rd.title,
               "source_path": rd.path,
               "filetype": rd.filetype,
               "last_modified": rd.mtime_iso,
               "chunk_index": i,
           })

   if not ids:
       st.info("[index] No documents to upsert.")
       return 0

   st.write(f"[index] Embedding {len(ids)} chunks…")
   vecs = embedder.encode(docs_text, normalize_embeddings=True).tolist()
   coll.upsert(ids=ids, documents=docs_text, metadatas=metas, embeddings=vecs)
   return len(ids)

def needs_reindex(docs: List[RawDoc]) -> bool:
   """If any file mtime is newer than last index time marker, or marker missing."""
   marker = pathlib.Path(CHROMA_DIR) / ".last_indexed"
   if not marker.exists():
       return True
   last = dt.datetime.fromtimestamp(marker.stat().st_mtime)
   for rd in docs:
       mtime = dt.datetime.fromisoformat(rd.mtime_iso)
       if mtime > last:
           return True
   return False

def touch_index_marker():
   marker = pathlib.Path(CHROMA_DIR) / ".last_indexed"
   marker.parent.mkdir(parents=True, exist_ok=True)
   marker.write_text(str(time()))


# ------------------- Retrieval + LLM -------------------
def retrieve(question: str, embedder, k: int = 5, where: Optional[Dict]=None) -> List[Dict]:
   coll = get_collection()
   qvec = embedder.encode([question], normalize_embeddings=True).tolist()
   res = coll.query(query_embeddings=qvec, n_results=k, where=where)
   out = []
   if not res or not res.get("ids") or not res["ids"][0]:
       return out
   for i in range(len(res["ids"][0])):
       out.append({
           "id": res["ids"][0][i],
           "text": res["documents"][0][i],
           "metadata": res["metadatas"][0][i],
           "score": float(res["distances"][0][i]) if "distances" in res else None
       })
   return out


def build_prompt(question: str, hits: List[Dict]) -> str:
   ctx = []
   for h in hits:
       m = h["metadata"]
       ctx.append(
           f"### {m.get('title')} ({m.get('source_path')})\n"
           f"[filetype={m.get('filetype')}, last_modified={m.get('last_modified')}, chunk={m.get('chunk_index')}]\n"
           f"{h['text']}"
       )
   context = "\n\n".join(ctx)
   return f"""{SYSTEM_PROMPT}

               # Context
               {context}
              
               # Question
               {question}
              
               # Answering rules
                   1. First, do the literature review. Look for gaps, under-explored research areas.
                   2. Second, identify key themes. Focus on topics that align with the goals and pay attention to recurring themes, particular aspects, methodologies across different studies.
                   3. Thirdly, formulate research questions. Develop specific topics based on the gaps or themes identified. Ensure these topics offer fresh perspectives or new insights. They should be clear, focused, and researchable.
                   4. Make sure the scope is neither too broad nor too narrow. The research topics should connect with existing theories or models which provide foundations for the research.
               """
#                - Rely strictly on the context.
#                - Cite (title → file path) after each factual claim.
def generate(prompt: str, temperature: float = 0.2) -> str:
   try:
       client = InferenceClient(
           provider="nscale",
           api_key=API_KEY,
           )

       resp = client.chat.completions.create(
           model="Qwen/Qwen3-4B-Instruct-2507",  #"Qwen/Qwen3-4B-Thinking-2507"
           messages=[
               {
                   "role": "user",
                   "content": prompt
               }
           ],
           stream = False,
           temperature= temperature
       )
       return resp.choices[0].message.content

       # resp = ollama.chat(
       #     model="llama3.2:3b",
       #     messages=[{"role": "user", "content": prompt}],
       #     stream=False,
       #     options={"temperature": temperature}
       # )
       # return resp["message"]["content"]

   except Exception as e:
       return f"[ERROR calling hugging face llm] {e}"
  
  
# ------------------- UI -------------------
st.set_page_config(page_title="RAG • ChromaDB + QWen 3 (4B)", layout="wide")
st.title("RAG with ChromaDB + QWen 3 (4B)")

# Ensure data dir exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

with st.sidebar:
   st.header("Settings")
   top_k = st.slider("Top-K (retrieval)", min_value=1, max_value=10, value=5, step=1)
   chunk_chars = st.slider("Chunk size (chars)", min_value=300, max_value=3000, value=1200, step=100)
   chunk_overlap = st.slider("Chunk overlap (chars)", min_value=0, max_value=600, value=150, step=10)
   filt = st.selectbox("Filter by file type (optional)", options=["", "pdf", "docx", "html", "markdown", "text"])
   temperature = st.slider("LLM temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

   st.markdown("---")
   reindex_btn = st.button("(Re)Index")



# Upload files
st.subheader("Upload documents")
uploaded = st.file_uploader(
   "Add PDFs, DOCX, HTML, MD, or TXT",
   type=["pdf", "docx", "html", "htm", "md", "markdown", "txt"],
   accept_multiple_files=True,
)
if uploaded:
   for f in uploaded:
       dest = DATA_DIR / f.name
       with open(dest, "wb") as out:
           out.write(f.read())
   st.success(f"Uploaded {len(uploaded)} file(s) to {DATA_DIR.resolve()}")

# List current files
st.subheader("Current corpus")
files = sorted([p for p in DATA_DIR.rglob("*") if p.is_file()])
if files:
   st.write(f"{len(files)} file(s) in {DATA_DIR}")
   for p in files[:50]:
       stat = p.stat()
       st.caption(f"• {p.name} — {p.suffix[1:].lower()} — {dt.datetime.fromtimestamp(stat.st_mtime)}")
else:
   st.info("No files yet. Upload above or place files in ./data")

# Indexing
embedder = get_embedder()
docs = walk_data_dir(DATA_DIR)

# Auto-index if needed or on button
auto_needed = needs_reindex(docs)
if auto_needed:
   st.info("Index appears stale or missing. Click (Re)Index to build.")
if reindex_btn:
   if not docs:
       st.warning("No supported documents found to index.")
   else:
       with st.spinner("Indexing… this can take a minute on first run (downloading embedding model)."):
           n_chunks = index_docs(docs, embedder, max_chars=chunk_chars, overlap=chunk_overlap)
           touch_index_marker()
       st.success(f"Indexed {n_chunks} chunk(s) into Chroma at {CHROMA_DIR}")

st.markdown("---")
st.header("Ask your corpus")
q = st.text_input("Question", placeholder="e.g., What are our warranty terms for frames and forks?")
ask_col1, ask_col2 = st.columns([1, 3])
with ask_col1:
   ask_btn = st.button("Ask")
with ask_col2:
   filt_dict = {"filetype": filt} if filt else None
   st.caption(f"Filter: {filt_dict if filt_dict else 'None'} • Top-K: {top_k} • Temp: {temperature}")

if ask_btn and q.strip():
   with st.spinner("Retrieving relevant chunks…"):
       hits = retrieve(q, embedder, k=top_k, where=filt_dict)
   if not hits:
       st.warning("No relevant context found. Try re-indexing or broadening your query.")
       st.stop()

   st.subheader("Retrieved chunks")
   for i, h in enumerate(hits, start=1):
       m = h["metadata"]
       with st.expander(f"{i}. {m['title']}  |  {m['filetype']}  |  score={h['score']}", expanded=(i == 1)):
           st.caption(f"{m['source_path']}  •  last_modified={m['last_modified']}  •  chunk={m['chunk_index']}")
           st.write(h["text"])

   prompt = build_prompt(q, hits)
   with st.spinner("Generating grounded answer with QWen 3 (4B) …"):
       answer = generate(prompt, temperature=temperature)

   st.subheader("Answer")
   st.write(answer)
  

# if __name__ == "__main__":
#     doc = walk_data_dir(DATA_DIR)
#     print("chunck finished!")
#     tick1 = time()
#     embedder = get_embedder()
#     num_chunks = index_docs(docs = doc, embedder = embedder, max_chars=5000, overlap=100)
#     tick2 = time()
#     print("chromadb indexing finished! takes: ", tick2-tick1)
#     question = "summarize the strategies trading ETF"
#     hits = retrieve(question, embedder)
#     tick3 = time()
#     print("retrieve finished! takes: ", tick3-tick2)
#     prompt = build_prompt(question, hits)
#     answer = generate(prompt)
#     tick4 = time()
#     print("generate answer takes: ", tick4-tick3)
#     print("answer: ", answer)
