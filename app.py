# app.py — Bud Chat Bot (Streamlit + Groq + PDF + Excel)
# -------------------------------------------------------------------------
# 🧒 Kid‑friendly chatbot with colorful layout and rounded chat bubbles 😊
#   • st.chat_input() → arrow send button (ChatGPT‑style)
#   • First user question renames the chat from “New Chat” automatically
#   • Sidebar lists recent chats as clickable buttons, newest first
#   • Added _rerun() helper so the code works on BOTH old & new Streamlit builds
#   • Same answer pipeline: Excel → PDF → Groq fallback
# -------------------------------------------------------------------------

from __future__ import annotations

import os, re, string, logging, uuid, traceback, json
from difflib import get_close_matches
from datetime import datetime
from pathlib import Path

import streamlit as st

# 🔄 Helper so the app can reload itself no matter which Streamlit version
# you have (old versions used st.experimental_rerun, new ones use st.rerun).

def _rerun():
    """Reload the app regardless of Streamlit version."""
    if hasattr(st, "rerun"):
        st.rerun()                     # Streamlit ≥ 1.27
    else:
        st.experimental_rerun()        # Streamlit 1.21 – 1.26

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

# ── ENV & LOG ─────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

BOT_NAME = os.getenv("BOT_NAME", "Bud")
BOT_TONE = os.getenv("BOT_TONE", "kids").lower()  # default to kid tone
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN")
UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)
SESSIONS_PATH = Path("chat_sessions.json")  # persistent chat logs

# ── PROMPT STYLE ──────────────────────────────────────────────────────────
TONE_INSTRUCTION = {
    "academic": "You are a scholarly assistant. Answer formally and precisely.",
    "kids":     "You are a patient teacher for kids. Use simple words and friendly emojis.",
    "kid":      "You are a patient teacher for kids. Use simple words and friendly emojis.",
}.get(BOT_TONE, "You are a cheerful, friendly helper. Use warm conversational language and emojis.")

PROMPT = ChatPromptTemplate.from_template(
    f"{TONE_INSTRUCTION}\n"
    "(Answer in ONE short paragraph. Use ONLY the information in <context>. "
    "If the context is empty or not relevant, say 'I don't know based on the provided documents.')\n"
    "<context>{{context}}</context>\nQuestion: {{input}}"
)

# ── LLM ────────────────────────────────────────────────────────────────────
llm = None
if GROQ_KEY:
    llm = ChatGroq(
        groq_api_key=GROQ_KEY,
        model_name="Llama3-8b-8192",
        temperature=0.3,
        max_tokens=256,
    )
    logging.info("✅ Groq client ready")

# ── EMBEDDING MODEL ────────────────────────────────────────────────────────
class STEmbeddings(Embeddings):
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model, cache_folder="models", use_auth_token=HF_TOKEN)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False)

    def embed_query(self, text):
        return self.model.encode([text])[0]

# ── LOAD PDF & EXCEL ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔍 Indexing PDFs…")
def load_vectors():
    docs = PyPDFDirectoryLoader(UPLOAD_DIR).load()
    if not docs:
        return None
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)
    return FAISS.from_documents(chunks, STEmbeddings())

@st.cache_resource(show_spinner="📖 Reading Excel Q&A…")
def load_qa(path=os.path.join(UPLOAD_DIR, "questions_answers.xlsx")):
    if not os.path.exists(path):
        return {}
    df = pd.read_excel(path)
    df.iloc[:, 0] = (
        df.iloc[:, 0]
        .astype(str)
        .str.replace(f"[{re.escape(string.punctuation)}]", "", regex=True)
        .str.lower()
    )
    return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

VECTORS = load_vectors()
QA_DATA = load_qa()

# ── CHAT SESSION HELPERS ──────────────────────────────────────────────────

def load_all_sessions() -> dict[str, list[dict[str, str]]]:
    """Return saved sessions from disk or a default starter."""
    if SESSIONS_PATH.exists():
        try:
            return json.loads(SESSIONS_PATH.read_text())
        except Exception:
            logging.warning("⚠️ Could not read chat_sessions.json; starting fresh.")
    return {"New Chat": []}


def save_sessions(sessions: dict[str, list[dict[str, str]]]):
    """Persist sessions to disk."""
    SESSIONS_PATH.write_text(json.dumps(sessions))

# ── CLEANING FUNCTION ──────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return re.sub(rf"[{re.escape(string.punctuation)}]", "", text.lower().strip())

# ── ANSWER LOGIC ───────────────────────────────────────────────────────────

def answer(q: str) -> str:
    try:
        if not q:
            return ""

        q_norm = normalize(q)

        # 1️⃣ Greetings
        greetings = {"hi", "hello", "hey", "good morning", "good evening", "good afternoon"}
        if q_norm in greetings:
            return "👋 Hi there! How can I help you today?"

        # 2️⃣ Identity
        if re.search(r"\b(who (are|r) (you|u)|what('?s| is) your name|introduce yourself)\b", q_norm):
            return f"😊 I’m **{BOT_NAME}**, your friendly chatbot buddy! Ask me anything."

        # 3️⃣ Excel direct match
        if q_norm in QA_DATA:
            return str(QA_DATA[q_norm])

        # 4️⃣ Excel close match
        close = get_close_matches(q_norm, QA_DATA.keys(), n=1, cutoff=0.85)
        if close:
            return str(QA_DATA[close[0]])

        # 5️⃣ PDF vector + Groq
        if VECTORS and llm:
            chain = create_retrieval_chain(
                retriever=VECTORS.as_retriever(k=4),
                combine_documents_chain=create_stuff_documents_chain(llm, PROMPT),
            )
            out = chain.invoke({"input": q})
            ans = out.get("answer", "").strip()
            if ans and "I don't know" not in ans:
                return ans

        # 6️⃣ Groq fallback
        if llm:
            prompt = PROMPT.format(context="", input=q)
            raw = llm.invoke(prompt)
            return raw.strip()

        return "🤷‍♂️ Sorry, I don’t have an answer for that right now."

    except Exception:
        logging.error("answer() crashed:\n" + traceback.format_exc())
        raise  # Let Streamlit show “Oops! I got confused.”

# ── STREAMLIT UI ───────────────────────────────────────────────────────────

st.set_page_config("Bud Bot", "🤖", layout="centered")

# ─── Session‑state defaults ───────────────────────────────────────────────

if "sessions" not in st.session_state:
    st.session_state.sessions = load_all_sessions()

if "current" not in st.session_state:
    st.session_state.current = list(st.session_state.sessions.keys())[0]

# ─── Sidebar: Chat picker ─────────────────────────────────────────────────

with st.sidebar:
    st.header("💬 Chats")

    # ➕ New Chat button
    if st.button("➕ New Chat"):
        base = "New Chat"
        name = base
        i = 1
        while name in st.session_state.sessions:
            i += 1
            name = f"{base} {i}"
        st.session_state.sessions[name] = []
        st.session_state.current = name
        save_sessions(st.session_state.sessions)
        _rerun()

    st.markdown("---")

    # List chats, newest first
    for name in reversed(list(st.session_state.sessions.keys())):
        button_label = f"➡️ {name}" if name != st.session_state.current else f"🟢 {name}"
        if st.button(button_label, key=f"chat-{name}"):
            st.session_state.current = name
            st.experimental_rerun()

    st.markdown("---")
    st.write("📗 **Add PDFs to knowledge base**")
    pdf = st.file_uploader("Upload a PDF", type="pdf")
    if pdf:
        uid = f"{uuid.uuid4()}.pdf"
        open(os.path.join(UPLOAD_DIR, uid), "wb").write(pdf.read())
        load_vectors.clear()  # rebuild on next question
        st.success("PDF saved! I’ll learn from it after your next question.")

    st.markdown("---")
    st.caption("All chats auto‑save locally in *chat_sessions.json*.")

# ─── Main chat area ───────────────────────────────────────────────────────

st.image(
    "https://s.tmimgcdn.com/scr/1200x750/153700/business-analytics-logo-template_153739-original.jpg",
    width=60,
)
st.markdown(
    "<h1 style='text-align:center;color:#00B7FF;'>🤖 Bud Chat Bot</h1><p style='text-align:center;'>Ask me anything in a friendly way!</p>",
    unsafe_allow_html=True,
)
st.divider()

# Show chat history
for msg in st.session_state.sessions[st.session_state.current]:
    st.chat_message(msg["role"]).markdown(msg["content"])

# ─── Question input (now with built‑in arrow) ─────────────────────────────

q = st.chat_input("Ask me anything…")

if q:
    with st.spinner("Thinking…"):
        # Display user message
        st.chat_message("user").markdown(q)
        st.session_state.sessions[st.session_state.current].append(
            {"role": "user", "content": q, "time": datetime.now().isoformat()}
        )

        # Rename chat if it was still “New Chat”
        if st.session_state.current.startswith("New Chat"):
            raw_title = re.sub(r"\s+", " ", q.strip()).title()[:40] or "Untitled"
            title = raw_title
            base = raw_title
            i = 1
            while title in st.session_state.sessions:
                i += 1
                title = f"{base} ({i})"
            st.session_state.sessions[title] = st.session_state.sessions.pop(st.session_state.current)
            st.session_state.current = title

        # Get answer
        try:
            reply = answer(q)
        except Exception:
            reply = "Oops! I got confused. Try again?"

        st.chat_message("assistant").markdown(reply)
        st.session_state.sessions[st.session_state.current].append(
            {"role": "assistant", "content": reply, "time": datetime.now().isoformat()}
        )
        save_sessions(st.session_state.sessions)
        st.experimental_rerun()

# ─── Quick example links ─────────────────────────────────────────────────

quick = ["Who are you?", "Tell me a fun fact!", "How do planes fly?"]
st.markdown("**Try one:** " + " | ".join(f"🟢 [{x}](?q={x.replace(' ','%20')})" for x in quick))

# ─── CSS tweaks for bubbles & arrow ───────────────────────────────────────

st.markdown(
    """
    <style>
    .stChatMessage {border-radius:18px!important;}
    /* style for the arrow send‑button */
    button[kind="secondary"] svg {width:20px;height:20px;}
    button[kind="secondary"] {
        background:#00B7FF!important;
        color:white!important;
        border-radius:50%!important;
        padding:14px!important;
    }
    footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Handle ?q=... in URL (prefill input) ────────────────────────────────

params = st.experimental_get_query_params()
param_q = params.get("q")
if param_q:
    st.experimental_set_query_params()  # clear it so it doesn't loop
    st.session_state._on_chat_input_state_change = None  # hack: avoid key collision
    st.chat_input("Ask me anything…", key="unique_prefill_key", disabled=True)
    st.info("➡️ Click the green example link again if nothing happened ✨")
