"""
Streamlit version of **Bud Chat Bot**
Kid‑friendly layout, bright colors, rounded chat bubbles 😊
Works on Streamlit Cloud (port 8501).  PDF upload + Groq‑powered answers.
"""
import os, re, string, logging, uuid
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

# ── ENV & LOG ───────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

BOT_NAME   = os.getenv("BOT_NAME", "Bud")
BOT_TONE   = os.getenv("BOT_TONE", "friendly").lower()
GROQ_KEY   = os.getenv("GROQ_API_KEY", "")
HF_TOKEN   = os.getenv("HF_TOKEN")
UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── PROMPT HELPER ───────────────────────────────────────────────────────────
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

# ── LLM ─────────────────────────────────────────────────────────────────────
llm = None
if GROQ_KEY:
    llm = ChatGroq(
        groq_api_key=GROQ_KEY,
        model_name="Llama3-8b-8192",
        temperature=0.3,
        max_tokens=256,
    )
    logging.info("✅ Groq client ready")

# ── EMBEDDINGS ──────────────────────────────────────────────────────────────
class STEmbeddings(Embeddings):
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model, cache_folder="models", use_auth_token=HF_TOKEN)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False)

    def embed_query(self, text):
        return self.model.encode([text])[0]

# ── CACHED LOADERS ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔍 Building PDF index…")
def load_vectors():
    docs = PyPDFDirectoryLoader(UPLOAD_DIR).load()
    if not docs:
        return None
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)
    return FAISS.from_documents(chunks, STEmbeddings())

@st.cache_resource(show_spinner="📖 Loading Excel Q&A…")
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

# ── ANSWER FUNCTION ─────────────────────────────────────────────────────────
def normalize(text:str)->str:
    return re.sub(rf"[{re.escape(string.punctuation)}]", "", text.lower().strip())

def answer(q:str)->str:
    if not q:
        return ""
    key = normalize(q)
    if key in QA_DATA:
        return QA_DATA[key]
    close = get_close_matches(key, QA_DATA.keys(), n=1, cutoff=0.85)
    if close:
        return QA_DATA[close[0]]

    if VECTORS and llm:
        try:
            chain = create_retrieval_chain(
                retriever=VECTORS.as_retriever(k=4),
                combine_documents_chain=create_stuff_documents_chain(llm, PROMPT),
            )
            out = chain.invoke({"input": q})
            return out.get("answer", "I don't know yet.")
        except Exception as e:
            logging.error(e)
    return "Sorry, I don't know yet."

# ── STREAMLIT PAGE ──────────────────────────────────────────────────────────
st.set_page_config("Bud Bot", "🤖", layout="centered")

# 1. THEME (via config.toml) already handled; header:
st.image("https://i.imgur.com/nb3G9p6.png", width=110)
st.markdown("<h1 style='text-align:center;color:#00B7FF;'>🤖 Bud Chat Bot</h1><p style='text-align:center;'>Ask me anything in a friendly way!</p>", unsafe_allow_html=True)
st.divider()

# 2. Layout: chat left, upload right
col_chat, col_up = st.columns([3,1])

with col_chat:
    q = st.text_input("💬 Type your question here:", key="input")
    if st.button("Ask", type="primary") and q:
        with st.spinner("Thinking…"):
            st.chat_message("user").markdown(q)
            try:
                reply = answer(q)
            except Exception:
                reply = "Oops! I got confused. Try again?"
            st.chat_message("assistant").markdown(reply)

with col_up:
    st.write("📚 **Add PDFs**")
    pdf = st.file_uploader(" ", type="pdf", label_visibility="collapsed")
    if pdf:
        uid = f"{uuid.uuid4()}.pdf"
        open(os.path.join(UPLOAD_DIR, uid), "wb").write(pdf.read())
        load_vectors.clear()
        st.success("PDF uploaded! Vector index will rebuild on next question.")

# 3. Quick‑ask examples
quick = ["Who are you?", "Tell me a fun fact!", "How do planes fly?"]
st.markdown("**Try one:** " + " | ".join(f"🟢 [{x}](?q={x.replace(' ','%20')})" for x in quick))

# 4. CSS polish
st.markdown("""
<style>
.stChatMessage {border-radius:18px!important;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# 5. Handle quick‑ask query param
params = st.experimental_get_query_params()
param_q = params.get("q")               # may be list or single value
if q == "" and param_q:
    # if it's already a list take first item, else cast to str
    val = param_q[0] if isinstance(param_q, list) else str(param_q)
    st.session_state.input = val
    st.rerun()
