# app.py – Bud Chat Bot (Streamlit + Groq **or** OpenAI)
# -----------------------------------------------------------------------------
# Chooses Groq‑hosted Llama‑3 if GROQ_API_KEY is set, otherwise falls back to
# OpenAI ChatGPT (gpt‑3.5‑turbo or gpt‑4o) when OPENAI_API_KEY is present.
# -----------------------------------------------------------------------------

from __future__ import annotations
import os, re, string, logging, uuid, traceback, json
from datetime import datetime
from pathlib import Path
import streamlit as st

import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI  # <‑‑ NEW: OpenAI wrapper
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings.base import Embeddings

# ------------------------------ helpers --------------------------------------

def _rerun():
    (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()

def normalize(text: str) -> str:
    return re.sub(rf"[{re.escape(string.punctuation)}]", "", text.lower().strip())

GREET_RX = re.compile(r"\b(how (are|r) you( doing)?|how'?s it going|what'?s up)\b")

# ------------------------------ env & constants ------------------------------

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BOT_NAME   = os.getenv("BOT_NAME", "Bud")
BOT_TONE   = os.getenv("BOT_TONE", "kids").lower()
GROQ_KEY   = os.getenv("GROQ_API_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")  # <‑‑ NEW
HF_TOKEN   = os.getenv("HF_TOKEN")
UPLOAD_DIR = "data"; os.makedirs(UPLOAD_DIR, exist_ok=True)
SESSIONS_PATH = Path("chat_sessions.json")

TONE_INSTRUCTION = {
    "academic": "You are a scholarly assistant.",
    "kids":     "You are a patient teacher for kids. Use simple words and friendly emojis.",
    "kid":      "You are a patient teacher for kids. Use simple words and friendly emojis."
}.get(BOT_TONE, "You are a cheerful, friendly helper.")

# ------------------------------ prompts --------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_template(
    f"{TONE_INSTRUCTION}\n(Answer in ONE short paragraph. Use ONLY the information in <context>. "
    "If the context is empty or not relevant, say 'I don't know based on the provided documents.')\n"
    "<context>{{context}}</context>\nQuestion: {{input}}")

FREE_PROMPT = ChatPromptTemplate.from_template(
    f"{TONE_INSTRUCTION}\nAnswer the question in ONE kid‑friendly paragraph. "
    "Think step‑by‑step *inside* <scratchpad> then finish with **Answer:**.\n"
    "<scratchpad>{{scratch}}</scratchpad>\nQuestion: {{input}}")

# ------------------------------ LLM factory ----------------------------------

def make_llm():
    """Return a ChatGroq, ChatOpenAI, or None depending on available keys."""
    if GROQ_KEY:
        logging.info("Using Groq Llama‑3 via GROQ_API_KEY")
        return ChatGroq(groq_api_key=GROQ_KEY,
                        model_name="Llama3-8b-8192",
                        temperature=0.3,
                        max_tokens=256)
    if OPENAI_KEY:
        logging.info("Using OpenAI ChatGPT via OPENAI_API_KEY")
        # Choose the best model you have access to; fallback to 3.5 turbo.
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        return ChatOpenAI(openai_api_key=OPENAI_KEY,
                          model_name=model,
                          temperature=0.3,
                          max_tokens=256)
    logging.warning("No GROQ_API_KEY or OPENAI_API_KEY found – LLM disabled")
    return None

llm = make_llm()

# ------------------------------ embeddings -----------------------------------

class STEmbeddings(Embeddings):
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model, cache_folder="models", use_auth_token=HF_TOKEN)
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False)
    def embed_query(self, text):
        return self.model.encode([text])[0]

# ------------------------------ vector stores --------------------------------

@st.cache_resource(show_spinner="📄 Building PDF index…")
def load_pdf_vectors():
    docs = PyPDFDirectoryLoader(UPLOAD_DIR).load()
    if not docs:
        return None
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)
    return FAISS.from_documents(chunks, STEmbeddings())

@st.cache_resource(show_spinner="📖 Indexing Excel Q&A…")
def load_qa_vectors(path=f"{UPLOAD_DIR}/questions_answers.xlsx"):
    if not os.path.exists(path):
        return None, None
    df = pd.read_excel(path)
    questions = df.iloc[:, 0].astype(str).tolist()
    answers   = df.iloc[:, 1].astype(str).tolist()
    metas     = [{"index": i} for i in range(len(questions))]
    vs = FAISS.from_texts(questions, STEmbeddings(), metadatas=metas)
    return vs, answers

PDF_VS = load_pdf_vectors()
QA_VS, QA_ANSWERS = load_qa_vectors()
SIM_THRESHOLD = 0.25  # smaller = stricter match

# ------------------------------ answer logic ---------------------------------

def answer(q: str) -> str:
    try:
        if not q:
            return ""
        q_clean = normalize(q)

        # Small‑talk & identity
        if q_clean in {"hi", "hello", "hey", "good morning", "good evening", "good afternoon"} or GREET_RX.search(q_clean):
            return "😊 I'm just happy computer code, thanks for asking! How can I help you today?"
        if re.search(r"\b(who (are|r) (you|u)|what('?s| is) your name|introduce yourself)\b", q_clean):
            return f"😊 I’m **{BOT_NAME}**, your friendly chatbot buddy!"

        # 1. Excel semantic Q&A
        if QA_VS:
            docs = QA_VS.similarity_search_with_score(q, k=1)
            if docs:
                doc, score = docs[0]
                if score < SIM_THRESHOLD:
                    return QA_ANSWERS[doc.metadata["index"]]

        # 2. PDF RAG
        if PDF_VS and llm:
            chain = create_retrieval_chain(
                retriever=PDF_VS.as_retriever(k=4),
                combine_documents_chain=create_stuff_documents_chain(llm, RAG_PROMPT))
            ans = chain.invoke({"input": q}).get("answer", "").strip()
            if ans and "I don't know" not in ans:
                return ans

        # 3. Free reasoning
        if llm:
            return llm.invoke(FREE_PROMPT.format(input=q, scratch="")).strip()

        return "🤷‍♂️ Sorry, I don’t have an answer right now."

    except Exception:
        logging.error("answer() crashed:\n" + traceback.format_exc())
        return "⚠️ Oops, something went wrong. Please try again."

# ------------------------------ UI (unchanged) -------------------------------

st.set_page_config("Bud Bot", "🤖", layout="centered")
if "sessions" not in st.session_state:
    st.session_state.sessions = (json.loads(SESSIONS_PATH.read_text()) if SESSIONS_PATH.exists() else {"New Chat": []})
if "current" not in st.session_state:
    st.session_state.current = list(st.session_state.sessions.keys())[0]

with st.sidebar:
    st.header("💻 Chats")
    if st.button("➕ New Chat"):
        n, i = "New Chat", 1
        while n in st.session_state.sessions:
            i += 1; n = f"New Chat {i}"
        st.session_state.sessions[n] = []
        st.session_state.current = n
        json.dump(st.session_state.sessions, open(SESSIONS_PATH, "w"))
        _rerun()
    st.markdown("---")
    for n in reversed(list(st.session_state.sessions.keys())):
        if st.button(("🟢 " if n == st.session_state.current else "➡️ ") + n, key=f"btn-{n}"):
            st.session_state.current = n
            _rerun()
    st.markdown("---")
    st.write("📗 **Add PDFs to knowledge base**")
    pdf = st.file_uploader("Upload a PDF", type="pdf")
    if pdf:
        uid = f"{uuid.uuid4()}.pdf"
        open(f"{UPLOAD_DIR}/{uid}", "wb").write(pdf.read())
        load_pdf_vectors.clear()
        st.success("PDF saved! I’ll learn from it after your next question.")
    st.caption("Chats are stored locally in chat_sessions.json")

st.image("https://s.tmimgcdn.com/scr/1200x750/153700/business-analytics-logo-template_153739-original.jpg", width=60)

st.markdown("""
<h1 style='text-align:center;color:#00B7FF;'>🤖 Bud Chat Bot</h1>
<p style='text-align:center;'>Most Welcome to you ask me about Kepler college!</p>
""", unsafe_allow_html=True)

st.divider()
for m in st.session_state.sessions[st.session_state.current]:
    st.chat_message(m["role"]).markdown(m["content"])

q = st.chat_input("Ask me anything…")
if q:
    with st.spinner("Thinking…"):
        st.chat_message("user").markdown(q)
        st.session_state.sessions[st.session_state.current].append({"role": "user", "content": q, "time": datetime.now().isoformat()})
        if st.session_state.current.startswith("New Chat"):
            raw = re.sub("\s+", " ", q.strip()).title()[:40] or "Untitled"
            title, base, i = raw, raw, 1
            while title in st.session_state.sessions:
                i += 1; title = f"{base} ({i})"
            st.session_state.sessions[title] = st.session_state.sessions.pop(st.session_state
