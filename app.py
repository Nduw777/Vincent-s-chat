"""
Streamlit version of Bud Chat Bot (kid‑friendly)
Runs on port 8501 (default for Streamlit Cloud)
"""
import os, re, string, logging, uuid
from datetime import datetime
from difflib import get_close_matches

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings.base import Embeddings

# ── CONSTANTS & SETUP ───────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

BOT_NAME   = os.getenv("BOT_NAME", "Bud")
BOT_TONE   = os.getenv("BOT_TONE", "friendly").lower()
GROQ_KEY   = os.getenv("GROQ_API_KEY", "")
HF_TOKEN   = os.getenv("HF_TOKEN")
UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── LLM INITIALISATION ──────────────────────────────────────────────────────

def tone_instruction(tone: str) -> str:
    return {
        "academic": "You are a scholarly assistant. Answer formally and precisely.",
        "kids":     "You are a patient teacher for kids. Use simple words and friendly emojis.",
        "kid":      "You are a patient teacher for kids. Use simple words and friendly emojis.",
    }.get(tone, "You are a cheerful, friendly helper. Use warm conversational language and emojis.")

TONE_INSTRUCTION = tone_instruction(BOT_TONE)

llm = None
if GROQ_KEY:
    llm = ChatGroq(
        groq_api_key=GROQ_KEY,
        model_name="Llama3-8b-8192",
        temperature=0.3,
        max_tokens=256,
    )
    logging.info("✅ Groq Llama‑3 client ready.")
else:
    st.warning("GROQ_API_KEY not set – answers will be limited.")

prompt_template_str = (
    f"{TONE_INSTRUCTION}\n"
    "(Answer in ONE short paragraph. Use ONLY the information in <context>. "
    "If the context is empty or not relevant, say 'I don't know based on the provided documents.')\n"
    "<context>{{context}}</context>\nQuestion: {{input}}"
)
PROMPT = ChatPromptTemplate.from_template(prompt_template_str)

# ── EMBEDDINGS ──────────────────────────────────────────────────────────────
class STEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(
            model_name,
            cache_folder="models",
            use_auth_token=HF_TOKEN,
        )

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False)

    def embed_query(self, text):
        return self.model.encode([text])[0]

# ── DATA LOADERS (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔍 Building PDF vector index…")
def load_vectors():
    docs = PyPDFDirectoryLoader(UPLOAD_DIR).load()
    if not docs:
        return None
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)
    return FAISS.from_documents(chunks, STEmbeddings())

@st.cache_resource(show_spinner="📖 Loading Excel Q&A…")
def load_qa(path: str = os.path.join(UPLOAD_DIR, "questions_answers.xlsx")):
    if not os.path.exists(path):
        return {}
    df = pd.read_excel(path)
    df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.replace(f"[{re.escape(string.punctuation)}]", "", regex=True).str.lower()
    return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

VECTORS = load_vectors()
QA_DATA = load_qa()

# ── ANSWER FUNCTION ─────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return re.sub(rf"[{re.escape(string.punctuation)}]", "", text.lower().strip())


def answer(query: str) -> str:
    if not query:
        return ""

    key = normalize(query)
    if key in QA_DATA:
        return QA_DATA[key]

    # fuzzy match
    close = get_close_matches(key, QA_DATA.keys(), n=1, cutoff=0.85)
    if close:
        return QA_DATA[close[0]]

    if VECTORS and llm:
        chain = create_retrieval_chain(
            retriever=VECTORS.as_retriever(k=4),
            combine_documents_chain=create_stuff_documents_chain(llm, PROMPT),
        )
        try:
            result = chain.invoke({"input": query})
            return result.get("answer") or "I don't know yet."
        except Exception as e:
            logging.error("Retrieval chain failed: %s", e)

    return "Sorry, I don't know yet."  # fallback

# ── STREAMLIT UI ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bud Chat Bot", page_icon="🧑‍🚀", layout="centered")
st.title("🎒 Bud Chat Bot")

# File uploader (optional, for PDF handbooks)
ud_pdf = st.file_uploader("Upload a PDF to add to knowledge base", type="pdf")
if ud_pdf:
    uid = str(uuid.uuid4()) + ".pdf"
    with open(os.path.join(UPLOAD_DIR, uid), "wb") as f:
        f.write(ud_pdf.read())
    st.success("PDF uploaded. Rebuilding vectors…")
    VECTORS = load_vectors(clear_cache=True)  # rebuild index
    st.rerun()

user_q = st.text_input("Ask me anything:")
if user_q:
    with st.spinner("Thinking…"):
        response = answer(user_q)
    st.markdown(response)

st.markdown("---")
st.caption("Powered by LangChain, Groq, and Hugging Face ✨")
