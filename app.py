# app.py – Bud Chat Bot (Streamlit + Groq + PDF + Excel)
# Shortened comments to fit.

from __future__ import annotations
import os, re, string, logging, uuid, traceback, json
from difflib import get_close_matches
from datetime import datetime
from pathlib import Path
import streamlit as st

# rerun helper (works on all Streamlit versions)

def _rerun():
    (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()

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

# env & constants
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
BOT_NAME = os.getenv("BOT_NAME", "Bud")
BOT_TONE = os.getenv("BOT_TONE", "kids").lower()
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN")
UPLOAD_DIR = "data"; os.makedirs(UPLOAD_DIR, exist_ok=True)
SESSIONS_PATH = Path("chat_sessions.json")

TONE_INSTRUCTION = {
    "academic": "You are a scholarly assistant.",
    "kids": "You are a patient teacher for kids. Use simple words and friendly emojis.",
    "kid": "You are a patient teacher for kids. Use simple words and friendly emojis."
}.get(BOT_TONE, "You are a cheerful, friendly helper.")

PROMPT = ChatPromptTemplate.from_template(
    f"{TONE_INSTRUCTION}\n(Answer in ONE short paragraph. Use ONLY the information in <context>. "
    "If the context is empty or not relevant, say 'I don't know based on the provided documents.')\n"
    "<context>{{context}}</context>\nQuestion: {{input}}")

# llm
llm = ChatGroq(groq_api_key=GROQ_KEY, model_name="Llama3-8b-8192", temperature=0.3, max_tokens=256) if GROQ_KEY else None

# embeddings
class STEmbeddings(Embeddings):
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model, cache_folder="models", use_auth_token=HF_TOKEN)
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False)
    def embed_query(self, text):
        return self.model.encode([text])[0]

@st.cache_resource(show_spinner="📄 Building PDF index…")
def load_vectors():
    docs = PyPDFDirectoryLoader(UPLOAD_DIR).load();
    if not docs: return None
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)
    return FAISS.from_documents(chunks, STEmbeddings())

@st.cache_resource(show_spinner="📖 Loading Excel Q&A…")
def load_qa(path=f"{UPLOAD_DIR}/questions_answers.xlsx"):
    if not os.path.exists(path): return {}
    df = pd.read_excel(path)
    df.iloc[:,0] = df.iloc[:,0].astype(str).str.replace(f"[{re.escape(string.punctuation)}]", "", regex=True).str.lower()
    return dict(zip(df.iloc[:,0], df.iloc[:,1]))

VECTORS, QA_DATA = load_vectors(), load_qa()

# session helpers

def load_all_sessions():
    if SESSIONS_PATH.exists():
        try: return json.loads(SESSIONS_PATH.read_text())
        except Exception: logging.warning("⚠️ Bad chat_sessions.json, starting fresh")
    return {"New Chat": []}

def save_sessions(s): SESSIONS_PATH.write_text(json.dumps(s))

def normalize(t): return re.sub(rf"[{re.escape(string.punctuation)}]", "", t.lower().strip())

# answer

def answer(q):
    try:
        if not q: return ""
        qn = normalize(q)
        if qn in {"hi","hello","hey","good morning","good evening","good afternoon"}:
            return "👋 Hi there! How can I help you today?"
        if re.search(r"\b(who (are|r) (you|u)|what('?s| is) your name|introduce yourself)\b", qn):
            return f"😊 I’m **{BOT_NAME}**, your friendly chatbot buddy!"
        if qn in QA_DATA: return str(QA_DATA[qn])
        cm = get_close_matches(qn, QA_DATA.keys(), n=1, cutoff=0.85)
        if cm: return str(QA_DATA[cm[0]])
        if VECTORS and llm:
            chain = create_retrieval_chain(retriever=VECTORS.as_retriever(k=4), combine_documents_chain=create_stuff_documents_chain(llm, PROMPT))
            ans = chain.invoke({"input": q}).get("answer", "").strip()
            if ans and "I don't know" not in ans: return ans
        if llm: return llm.invoke(PROMPT.format(context="", input=q)).strip()
        return "🤷‍♂️ Sorry, I don’t have an answer for that right now."
    except Exception:
        logging.error("answer() crashed:\n"+traceback.format_exc()); raise

# UI
st.set_page_config("Bud Bot","🤖",layout="centered")
if "sessions" not in st.session_state: st.session_state.sessions = load_all_sessions()
if "current" not in st.session_state: st.session_state.current = list(st.session_state.sessions.keys())[0]

# sidebar
with st.sidebar:
    st.header("💬 Chats")
    if st.button("➕ New Chat"):
        n="New Chat"; i=1
        while n in st.session_state.sessions: i+=1; n=f"New Chat {i}"
        st.session_state.sessions[n]=[]; st.session_state.current=n; save_sessions(st.session_state.sessions); _rerun()
    st.markdown("---")
    for n in reversed(list(st.session_state.sessions.keys())):
        if st.button(("🟢 " if n==st.session_state.current else "➡️ ")+n, key=f"btn-{n}"):
            st.session_state.current=n; _rerun()
    st.markdown("---")
    st.write("📗 **Add PDFs to knowledge base**")
    pdf=st.file_uploader("Upload a PDF",type="pdf")
    if pdf:
        uid=f"{uuid.uuid4()}.pdf"; open(f"{UPLOAD_DIR}/{uid}","wb").write(pdf.read()); load_vectors.clear(); st.success("PDF saved! I’ll learn from it after your next question.")
    st.caption("Chats are stored locally in chat_sessions.json")

# main area
st.image("https://s.tmimgcdn.com/scr/1200x750/153700/business-analytics-logo-template_153739-original.jpg",width=60)
st.markdown("<h1 style='text-align:center;color:#00B7FF;'>🤖 Bud Chat Bot</h1><p style='text-align:center;'>Ask me anything!</p>",unsafe_allow_html=True)
st.divider()
for m in st.session_state.sessions[st.session_state.current]:
    st.chat_message(m["role"]).markdown(m["content"])

q=st.chat_input("Ask me anything…")
if q:
    with st.spinner("Thinking…"):
        st.chat_message("user").markdown(q)
        st.session_state.sessions[st.session_state.current].append({"role":"user","content":q,"time":datetime.now().isoformat()})
        if st.session_state.current.startswith("New Chat"):
            raw=re.sub("\s+"," ",q.strip()).title()[:40] or "Untitled"; title=raw; base=raw; i=1
            while title in st.session_state.sessions: i+=1; title=f"{base} ({i})"
            st.session_state.sessions[title]=st.session_state.sessions.pop(st.session_state.current); st.session_state.current=title
        try:
            reply=answer(q)
        except Exception:
            reply="Oops! I got confused. Try again?"
        st.chat_message("assistant").markdown(reply)
        st.session_state.sessions[st.session_state.current].append({"role":"assistant","content":reply,"time":datetime.now().isoformat()})
        save_sessions(st.session_state.sessions); _rerun()

quick=["Who are you?","Tell me a fun fact!","How do planes fly?"]
st.markdown("**Try one:** "+" | ".join(f"🟢 [{x}](?q={x.replace(' ','%20')})" for x in quick))

st.markdown("""<style>.stChatMessage{border-radius:18px!important;}button[kind='secondary'] svg{width:20px;height:20px;}button[kind='secondary']{background:#00B7FF!important;color:white!important;border-radius:50%!important;padding:14px!important;}footer{visibility:hidden;}</style>""",unsafe_allow_html=True)

params=st.experimental_get_query_params();
if not q and params.get("q"):
    st.experimental_set_query_params(); st.session_state.input=params["q"][0]; _rerun()
