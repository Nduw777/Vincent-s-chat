# app.py – Baho Chat Bot_Ndvt (Streamlit + Groq + PDF + Excel + Free‑Reasoning)
# -----------------------------------------------------------------------------
# This version lets Bud:       1) search Excel semantically      2) use PDF RAG
#                              3) reason freely when docs fail   4) keep kid tone
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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings.base import Embeddings

# ------------------------------ helpers --------------------------------------

def _rerun():
    """Works on all Streamlit versions."""
    (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()

def normalize(t: str) -> str:
    return re.sub(rf"[{re.escape(string.punctuation)}]", "", t.lower().strip())

# ------------------------------ env & constants ------------------------------

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

BOT_NAME = os.getenv("BOT_NAME", "Bud")
BOT_TONE = os.getenv("BOT_TONE", "kids").lower()
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN")
UPLOAD_DIR = "data"; os.makedirs(UPLOAD_DIR, exist_ok=True)
SESSIONS_PATH = Path("chat_sessions.json")

TONE_INSTRUCTION = {
    "academic": "You are a scholarly assistant.",
    "kids":     "You are a patient teacher for kids. Use simple words and friendly emojis.",
    "kid":      "You are a patient teacher for kids. Use simple words and friendly emojis."
}.get(BOT_TONE, "You are a cheerful, friendly helper.")

# ------------------------------ prompts --------------------------------------

# Strict, context‑only prompt used for PDF RAG
RAG_PROMPT = ChatPromptTemplate.from_template(
    f"{TONE_INSTRUCTION}\n(Answer in ONE short paragraph. Use ONLY the information in <context>. "
    "If the context is empty or not relevant, say 'I don't know based on the provided documents.')\n"
    "<context>{{context}}</context>\nQuestion: {{input}}")

# Free‑reasoning prompt for final fallback
FREE_PROMPT = ChatPromptTemplate.from_template(
    f"{TONE_INSTRUCTION}\nAnswer the question in ONE kid‑friendly paragraph. "
    "Think step‑by‑step *inside* <scratchpad> then finish with **Answer:**.\n"
    "<scratchpad>{{scratch}}</scratchpad>\nQuestion: {{input}}")

# ------------------------------ LLM ------------------------------------------

llm = (ChatGroq(groq_api_key=GROQ_KEY,
                model_name="Llama3-8b-8192",
                temperature=0.3,
                max_tokens=256) if GROQ_KEY else None)

# ------------------------------ embeddings class -----------------------------

class STEmbeddings(Embeddings):
    """SentenceTransformer wrapper so LangChain sees an Embeddings object."""
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model, cache_folder="models", use_auth_token=HF_TOKEN)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False)

    def embed_query(self, text):
        return self.model.encode([text])[0]

# ------------------------------ PDF vector store -----------------------------

@st.cache_resource(show_spinner="📄 Building PDF index…")
def load_pdf_vectors():
    docs = PyPDFDirectoryLoader(UPLOAD_DIR).load()
    if not docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, STEmbeddings())

# ------------------------------ Excel Q&A vector store -----------------------

@st.cache_resource(show_spinner="📖 Indexing Excel Q&A…")
def load_qa_vectors(path=f"{UPLOAD_DIR}/questions_answers.xlsx"):
    if not os.path.exists(path):
        return None, None
    df = pd.read_excel(path)
    questions = df.iloc[:, 0].astype(str).tolist()
    answers = df.iloc[:, 1].astype(str).tolist()
    metadatas = [{"index": i} for i in range(len(questions))]
    qa_vs = FAISS.from_texts(questions, STEmbeddings(), metadatas=metadatas)
    return qa_vs, answers

PDF_VS  = load_pdf_vectors()
QA_VS, QA_ANSWERS = load_qa_vectors()

# ------------------------------ answer logic ---------------------------------

def answer(q: str) -> str:
    """Generate a response using layered strategy: Excel → PDF RAG → free."""
    try:
        if not q:
            return ""
        q_clean = normalize(q)

        # ----- small‑talk handling ------------------------------------------------
        if q_clean in {"hi", "hello", "hey", "good morning", "good evening", "good afternoon"}:
            return "👋 Hi there! How can I help you today?"
        if re.search(r"\b(who (are|r) (you|u)|what('?s| is) your name|introduce yourself)\b", q_clean):
            return f"😊 I’m **{BOT_NAME}**, your friendly chatbot buddy!"

        # ----- 1. semantic Excel lookup ------------------------------------------
        if QA_VS:
            docs = QA_VS.similarity_search(q, k=1)
            if docs:
                idx = docs[0].metadata["index"]
                return QA_ANSWERS[idx]

        # ----- 2. PDF Retrieval‑Augmented Generation -----------------------------
        if PDF_VS and llm:
            chain = create_retrieval_chain(
                retriever=PDF_VS.as_retriever(k=4),
                combine_documents_chain=create_stuff_documents_chain(llm, RAG_PROMPT)
            )
            ans = chain.invoke({"input": q}).get("answer", "").strip()
            if ans and "I don't know" not in ans:
                return ans

        # ----- 3. Free reasoning fallback ----------------------------------------
        if llm:
            return llm.invoke(FREE_PROMPT.format(input=q, scratch="")).strip()

        # ----- no LLM available ---------------------------------------------------
        return "🤷‍♂️ Sorry, I don’t have an answer right now."

    except Exception:
        logging.error("answer() crashed:\n" + traceback.format_exc())
        return "⚠️ Oops, something went wrong. Please try again."

# ------------------------------ UI -------------------------------------------

st.set_page_config("Baho Bot_Ndvt", "🤖", layout="centered")

if "sessions" not in st.session_state:
    st.session_state.sessions = (json.loads(SESSIONS_PATH.read_text())
                                 if SESSIONS_PATH.exists() else {"New Chat": []})
if "current" not in st.session_state:
    st.session_state.current = list(st.session_state.sessions.keys())[0]

# ----- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("💻 Chats")

    # New chat button
    if st.button("➕ New Chat"):
        n = "New Chat"; i = 1
        while n in st.session_state.sessions:
            i += 1; n = f"New Chat {i}"
        st.session_state.sessions[n] = []
        st.session_state.current = n
        json.dump(st.session_state.sessions, open(SESSIONS_PATH, "w"))
        _rerun()

    st.markdown("---")

    # Chat list
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
        load_pdf_vectors.clear()  # rebuild on next question
        st.success("PDF saved! I’ll learn from it after your next question.")

    st.caption("Chats are stored locally in chat_sessions.json")

# ----- main chat area --------------------------------------------------------
st.image("https://s.tmimgcdn.com/scr/1200x750/153700/business-analytics-logo-template_153739-original.jpg", width=60)

st.markdown("""
<h1 style='text-align:center;color:#00B7FF;'>🤖 Bud Chat Bot</h1>
<p style='text-align:center;'>Most Welcome to you ask me about Kepler college!</p>
""", unsafe_allow_html=True)

st.divider()

# Display chat history
for m in st.session_state.sessions[st.session_state.current]:
    st.chat_message(m["role"]).markdown(m["content"])

# Chat input
q = st.chat_input("Ask me anything…")
if q:
    with st.spinner("Thinking…"):
        # Add user message
        st.chat_message("user").markdown(q)
        st.session_state.sessions[st.session_state.current].append({
            "role": "user", "content": q, "time": datetime.now().isoformat()})

        # Auto‑rename untitled chat
        if st.session_state.current.startswith("New Chat"):
            raw = re.sub("\s+", " ", q.strip()).title()[:40] or "Untitled"
            title, base, i = raw, raw, 1
            while title in st.session_state.sessions:
                i += 1; title = f"{base} ({i})"
            st.session_state.sessions[title] = st.session_state.sessions.pop(st.session_state.current)
            st.session_state.current = title

        # Get answer
        reply = answer(q)
        st.chat_message("assistant").markdown(reply)
        st.session_state.sessions[st.session_state.current].append({
            "role": "assistant", "content": reply, "time": datetime.now().isoformat()})

        # Persist
        json.dump(st.session_state.sessions, open(SESSIONS_PATH, "w"))
        _rerun()

# Quick suggestions
quick = ["Who are you?", "Tell me a fun fact!", "How do planes fly?"]
link_bar = " | ".join(f"🟢 [{x}](?q={x.replace(' ', '%20')})" for x in quick)
st.markdown(f"**Try one:** {link_bar}")

# Styling tweaks
st.markdown("""
<style>
.stChatMessage{border-radius:18px!important;}
button[kind='secondary'] svg{width:20px;height:20px;}
button[kind='secondary']{background:#00B7FF!important;color:white!important;border-radius:50%!important;padding:14px!important;}
footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# Allow URL pre‑fill
params = st.experimental_get_query_params()
if not q and params.get("q"):
    st.experimental_set_query_params()
    st.session_state.input = params["q"][0]
    _rerun()
