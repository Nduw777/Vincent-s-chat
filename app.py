"""
Bud Chat Bot – Streamlit + Groq / OpenAI
-------------------------------------------------
A simple, kid‑friendly chatbot that can:
1.  Answer questions with plain reasoning (Llama‑3 or GPT‑3.5/4o).
2.  Retrieve answers from uploaded PDFs (basic RAG).
3.  Pull Q&A pairs from an Excel file for instant replies.

Extra goodies:
* Multiple chat sessions saved to disk (JSON).
* "➕ New Chat" button works.
* Clean chat UI with st.chat_message + st.chat_input.
* Auto‑save after each user turn.

Place any PDFs in the **data/** folder and (optionally) an
Excel file called **questions_answers.xlsx** with two columns:
| Question | Answer |

-------------------------------------------------
Run with:
    streamlit run app.py
"""
from __future__ import annotations

# ------------------------------ stdlib ---------------------------------------
import json, logging, os, re, string, traceback, uuid
from datetime import datetime
from pathlib import Path

# ------------------------------ 3rd‑party ------------------------------------
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
# OpenAI wrapper (provider‑split or legacy)
try:
    from langchain_openai import ChatOpenAI  # LangChain ≥ 0.2 provider package
except ModuleNotFoundError:
    from langchain.chat_models import ChatOpenAI  # fallback for LangChain 0.1.x
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings.base import Embeddings

# ------------------------------ helpers --------------------------------------

def _rerun():
    """Safely rerun Streamlit app across versions."""
    (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()

def normalize(text: str) -> str:
    return re.sub(rf"[{re.escape(string.punctuation)}]", "", text.lower().strip())

GREET_RX = re.compile(r"\b(how (are|r) you( doing)?|how'?s it going|what'?s up)\b")

# ------------------------------ env & constants ------------------------------

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

BOT_NAME   = os.getenv("BOT_NAME", "Bud")
BOT_TONE   = os.getenv("BOT_TONE", "kids").lower()
GROQ_KEY   = os.getenv("GROQ_API_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
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
    """Return an LLM client based on available API keys."""
    if GROQ_KEY:
        logging.info("Using Groq Llama‑3 via GROQ_API_KEY")
        return ChatGroq(groq_api_key=GROQ_KEY,
                        model_name="Llama3-8b-8192",
                        temperature=0.3,
                        max_tokens=256)
    if OPENAI_KEY:
        logging.info("Using OpenAI ChatGPT via OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        return ChatOpenAI(openai_api_key=OPENAI_KEY,
                          model_name=model,
                          temperature=0.3,
                          max_tokens=256)
    logging.warning("No LLM API key found – reasoning fallback disabled")
    return None

llm = make_llm()

# ------------------------------ embeddings -----------------------------------

class STEmbeddings(Embeddings):
    """SentenceTransformer wrapper to match LangChain Embeddings interface."""
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
SIM_THRESHOLD = 0.25  # lower = stricter match

# ------------------------------ answer logic ---------------------------------

def answer(q: str) -> str:
    try:
        if not q:
            return ""
        q_clean = normalize(q)

        # Small‑talk & identity
        if q_clean in {"hi", "hello", "hey", "good morning", "good evening", "good afternoon"} or GREET_RX.search(q_clean):
            return "😊 I'm Baho bot_Ndvt, thanks for asking! How can I help you today?"
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

# ------------------------------ Streamlit UI ---------------------------------

st.set_page_config("Bud Bot", "🤖", layout="centered")

# ---------- Session storage (load or init) -----------------------------------
loaded_sessions = (
    json.loads(SESSIONS_PATH.read_text())
    if SESSIONS_PATH.exists() else {"New Chat": []}
)
if "sessions" not in st.session_state:
    st.session_state.sessions = loaded_sessions
if "current" not in st.session_state:
    st.session_state.current = list(st.session_state.sessions.keys())[0]

# ------------------------------ SIDEBAR --------------------------------------
with st.sidebar:
    st.header("💻 Chats")

    # "New Chat" button
    if st.button("➕ New Chat"):
        n, i = "New Chat", 1
        while n in st.session_state.sessions:
            i += 1
            n = f"New Chat {i}"
        st.session_state.sessions[n] = []  # fresh empty list
        st.session_state.current = n       # switch to it
        _rerun()

    # List existing sessions
    if st.session_state.sessions:
        chat_names = list(st.session_state.sessions.keys())
        choice = st.radio("Choose a chat:", chat_names, index=chat_names.index(st.session_state.current))
        if choice != st.session_state.current:
            st.session_state.current = choice
            _rerun()

    st.markdown("---")
    st.caption("PDFs + Excel Q&A live inside the *data/* folder.")

# ------------------------------ MAIN CHAT AREA -------------------------------
st.header("💬 Chat")

# 1️⃣ Display past messages
for msg in st.session_state.sessions[st.session_state.current]:
    st.chat_message(msg["role"]).write(msg["content"])

# 2️⃣ Chat input box
if prompt := st.chat_input("Ask me anything…"):
    # Store user message
    st.session_state.sessions[st.session_state.current].append({
        "role": "user",
        "content": prompt,
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })

    # Get assistant reply
    reply = answer(prompt)
    st.session_state.sessions[st.session_state.current].append({
        "role": "assistant",
        "content": reply,
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })

    # Persist to disk
    try:
        SESSIONS_PATH.write_text(json.dumps(st.session_state.sessions, indent=2))
    except Exception as e:
        logging.warning(f"Failed to save sessions: {e}")

    _rerun()
