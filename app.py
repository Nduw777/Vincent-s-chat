# app.py – Bud Chat Bot (Streamlit)
# -----------------------------------------------------------------------------
# Kid‑friendly chatbot with:
#   • Greeting replies (hi/hello/hey…)
#   • Excel Q&A lookup
#   • PDF vector search (FAISS) + Groq Llama‑3 answers
#   • Plain Groq fallback
#   • Works on Streamlit Cloud (port 8501) – fileWatcherType set to "poll".
# -----------------------------------------------------------------------------

# ── ENV FIX FOR STREAMLIT CLOUD ──────────────────────────────────────────────
# Must be set *before* importing streamlit so the watcher limit isn’t hit.
import os
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "poll")  # safer than inotify

# ── STANDARD LIB ────────────────────────────────────────────────────────────
import re, string, logging, uuid, traceback
from difflib import get_close_matches
from datetime import datetime  # (kept for future logging if needed)

# ── THIRD‑PARTY ─────────────────────────────────────────────────────────────
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

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
else:
    logging.warning("🚨 GROQ_API_KEY missing – fallback answers will be unavailable!")

# ── EMBEDDINGS WRAPPER ──────────────────────────────────────────────────────
class STEmbeddings(Embeddings):
    """Sentence‑Transformers wrapper for LangChain."""
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model, cache_folder="models", use_auth_token=HF_TOKEN)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False)

    def embed_query(self, text):
        return self.model.encode([text])[0]

# ── HELPER: NORMALISE TEXT ──────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, trim, and strip punctuation so lookups are easy."""
    return re.sub(rf"[{re.escape(string.punctuation)}]", "", text.lower().strip())

# ── CACHED LOADERS ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔍 Building PDF index…")
def load_vectors():
    """Load PDFs from UPLOAD_DIR and create a FAISS vector store."""
    docs = PyPDFDirectoryLoader(UPLOAD_DIR).load()
    if not docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, STEmbeddings())

@st.cache_resource(show_spinner="📖 Loading Excel Q&A…")
def load_qa(path=os.path.join(UPLOAD_DIR, "questions_answers.xlsx")):
    """Load an Excel file where column 0 is Q and column 1 is A."""
    if not os.path.exists(path):
        return {}
    df = pd.read_excel(path)
    if df.shape[1] < 2:
        logging.warning("Excel file needs at least two columns (Q & A).")
        return {}
    df.iloc[:, 0] = (
        df.iloc[:, 0].astype(str)
        .str.replace(f"[{re.escape(string.punctuation)}]", "", regex=True)
        .str.lower()
    )
    return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

VECTORS = load_vectors()
QA_DATA = load_qa()

# ── ANSWER FUNCTION ─────────────────────────────────────────────────────────

def answer(q: str) -> str:
    """Route the user question to greetings, Excel, PDF+LLM, or fallback."""
    try:
        if not q:
            return ""

        q_norm = normalize(q)

        #1️⃣ Greetings
        greetings = {"hi", "hello", "hey", "good morning", "good evening", "good afternoon"}
        if q_norm in greetings:
            return "👋 Hi there! How can I help you today?"

        # 2️⃣ Excel exact match
        if q_norm in QA_DATA:
            return str(QA_DATA[q_norm])

        # 3️⃣ Excel close match
        close = get_close_matches(q_norm, QA_DATA.keys(), n=1, cutoff=0.85)
        if close:
            return str(QA_DATA[close[0]])

        # 4️⃣ PDF vector + Groq
        if VECTORS and llm:
            chain = create_retrieval_chain(
                retriever=VECTORS.as_retriever(k=4),
                combine_documents_chain=create_stuff_documents_chain(llm, PROMPT),
            )
            out = chain.invoke({"input": q})
            ans = out.get("answer", "").strip()
            if ans and "I don't know" not in ans:
                return ans

        # 5️⃣ Plain Groq fallback (no PDF context)
        if llm:
            ans = llm.invoke(PROMPT.format(context="", input=q)).strip()
            return ans

        # 6️⃣ Ultimate fallback
        return "🤷‍♂️ Sorry, I don’t have an answer for that right now."

    except Exception:
        logging.error("answer() crashed:\n" + traceback.format_exc())
        raise  # Let outer try/except show the friendly message

# ── STREAMLIT PAGE ──────────────────────────────────────────────────────────
st.set_page_config("Bud Bot", "🤖", layout="centered")

# Header
st.image("https://i.imgur.com/nb3G9p6.png", width=110)
st.markdown(
    "<h1 style='text-align:center;color:#00B7FF;'>🤖 Bud Chat Bot</h1>"
    "<p style='text-align:center;'>Ask me anything in a friendly way!</p>",
    unsafe_allow_html=True,
)
st.divider()

# Layout: chat left, upload right
col_chat, col_up = st.columns([3, 1])

with col_chat:
    q = st.text_input("🧏🏼‍♂️ Type your question here:", key="input")
    if st.button("Ask", type="primary") and q:
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
        with open(os.path.join(UPLOAD_DIR, uid), "wb") as f:
            f.write(pdf.read())
        load_vectors.clear()  # Rebuild cache next time
        st.success("PDF uploaded! Vector index will rebuild on next question.")

# Quick‑ask examples
quick = ["Who are you?", "Tell me a fun fact!", "How do planes fly?"]
links = " | ".join(f"🟢 [{x}](?q={x.replace(' ', '%20')})" for x in quick)
st.markdown(f"**Try one:** {links}")

# CSS tweaks
st.markdown(
    """
<style>
.stChatMessage {border-radius:18px!important;}
footer {visibility:hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# Handle quick‑ask query param (so links auto‑fill input)
params = st.experimental_get_query_params()
if not q and (val := params.get("q")):
    st.session_state.input = val[0] if isinstance(val, list) else str(val)
    st.rerun()
