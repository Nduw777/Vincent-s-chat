# app.py — Bud Chat Bot (Streamlit + Groq + PDF + Excel)
# -------------------------------------------------------------------------
# Kid‑friendly chatbot with colorful layout and rounded chat bubbles 😊
# Answers from Excel, PDF (vector), or fallback to Groq (Llama3).
# -------------------------------------------------------------------------

import os, re, string, logging, uuid, traceback
from difflib import get_close_matches
from datetime import datetime

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

# ── ENV & LOG ─────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

BOT_NAME = os.getenv("BOT_NAME", "Bud")
BOT_TONE = os.getenv("BOT_TONE", "friendly").lower()
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN")
UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
@st.cache_resource(show_spinner="🔍 Indexing PDFs...")
def load_vectors():
    docs = PyPDFDirectoryLoader(UPLOAD_DIR).load()
    if not docs:
        return None
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)
    return FAISS.from_documents(chunks, STEmbeddings())

@st.cache_resource(show_spinner="📖 Reading Excel Q&A...")
def load_qa(path=os.path.join(UPLOAD_DIR, "questions_answers.xlsx")):
    if not os.path.exists(path):
        return {}
    df = pd.read_excel(path)
    df.iloc[:, 0] = (df.iloc[:, 0].astype(str)
                     .str.replace(f"[{re.escape(string.punctuation)}]", "", regex=True)
                     .str.lower())
    return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

VECTORS = load_vectors()
QA_DATA = load_qa()

# ── CLEANING FUNCTION ──────────────────────────────────────────────────────
def normalize(text: str) -> str:
    return re.sub(rf"[{re.escape(string.punctuation)}]", "", text.lower().strip())

# ── ANSWER LOGIC ───────────────────────────────────────────────────────────
def answer(q: str) -> str:
    try:
        if not q:
            return ""

        q_norm = normalize(q)

        #1️⃣ Greetings
        greetings = {"hi", "hello", "hey", "good morning", "good evening", "good afternoon"}
        if q_norm in greetings:
            return "👋 Hi there! How can I help you today?"

        # 2️⃣ Identity
        if re.search(r"\b(who (are|r) (you|u)|what('?s| is) your name|introduce yourself)\b", q_norm):
            return f"😊 I’m **{BOT_NAME}**, your friendly chatbot buddy! Ask me anything."

        # 3️⃣ Excel direct match
        if q_norm in QA_DATA:
            return QA_DATA[q_norm]

        # 4️⃣ Excel close match
        close = get_close_matches(q_norm, QA_DATA.keys(), n=1, cutoff=0.85)
        if close:
            return QA_DATA[close[0]]

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

st.image("data:image/jpeg;base64,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", width=110)
st.markdown("<h1 style='text-align:center;color:#00B7FF;'>🤖 Bud Chat Bot</h1><p style='text-align:center;'>Ask me anything in a friendly way!</p>", unsafe_allow_html=True)
st.divider()

col_chat, col_up = st.columns([3,1])

with col_chat:
    q = st.text_input("🧏🏼‍♂️ Type your question here:", key="input")
    if st.button("Ask", type="primary") and q:
        with st.spinner("Thinking…"):
            st.chat_message("user").markdown(q)
            try:
                reply = answer(q)
            except Exception:
                reply = "Oops! I got confused. Try again?"
            st.chat_message("assistant").markdown(reply)

with col_up:
    st.write("📗 **Add PDFs**")
    pdf = st.file_uploader(" ", type="pdf", label_visibility="collapsed")
    if pdf:
        uid = f"{uuid.uuid4()}.pdf"
        open(os.path.join(UPLOAD_DIR, uid), "wb").write(pdf.read())
        load_vectors.clear()
        st.success("PDF uploaded! Vector index will rebuild on next question.")

# Quick examples
quick = ["Who are you?", "Tell me a fun fact!", "How do planes fly?"]
st.markdown("**Try one:** " + " | ".join(f"🟢 [{x}](?q={x.replace(' ','%20')})" for x in quick))

# CSS styling
st.markdown("""
<style>
.stChatMessage {border-radius:18px!important;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# Handle ?q=... in URL
params = st.experimental_get_query_params()
param_q = params.get("q")
if q == "" and param_q:
    val = param_q[0] if isinstance(param_q, list) else str(param_q)
    st.session_state.input = val
    st.rerun()
