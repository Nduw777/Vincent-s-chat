# app.py – Bud Chat Board (2025‑07‑04 • stable)
# -----------------------------------------------------------------------------
# Flask + Socket.IO chatbot that can read PDF handbooks, remember rooms, and
# answer in a kid‑friendly tone.  This version fixes:
#   • create_stuff_documents_chain verbose bug
#   • mismatched parentheses that froze the server
#   • detailed logging to trace retrieval vs. LLM fallback
# -----------------------------------------------------------------------------

import os, re, logging, string, uuid, random
from datetime import datetime
from difflib import get_close_matches

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit, join_room
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_core.prompts import ChatPromptTemplate

# ── BASIC SETUP ───────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

BOT_NAME   = os.getenv("BOT_NAME", "Bud")
BOT_TONE   = os.getenv("BOT_TONE", "friendly").lower()
GROQ_KEY   = os.getenv("GROQ_API_KEY", "")
UPLOAD_DIR = "data"           # PDFs live here
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── LLM INITIALISATION ────────────────────────────────────────────────────────

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
    logging.warning("❗ GROQ_API_KEY not set – bot falls back to dumb answers.")

prompt_template_str = (
    f"{TONE_INSTRUCTION}\n"
    "(Answer in ONE short paragraph. Use ONLY the information in <context>. "
    "If the context is empty or not relevant, say \"I don't know based on the provided documents.\")\n"
    "<context>{{context}}</context>\nQuestion: {{input}}"
)
prompt = ChatPromptTemplate.from_template(prompt_template_str)

# ── EMBEDDINGS IMPLEMENTATION ────────────────────────────────────────────────

class STEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, cache_folder="models")
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False)
    def embed_query(self, text):
        return self.model.encode([text])[0]

# ── FLASK + SOCKET.IO APP ────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "change-this-secret")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── IN‑MEMORY CHAT ROOM STORE ────────────────────────────────────────────────
chats: dict[str, dict] = {}   # {room_id: {title: str, messages: list}}

# ── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return re.sub(rf"[{re.escape(string.punctuation)}]", "", text.lower().strip())

# Load Excel Q&A (optional)

def load_qa(path: str = os.path.join(UPLOAD_DIR, "questions_answers.xlsx")) -> dict:
    if not os.path.exists(path):
        return {}
    df = pd.read_excel(path)
    df.iloc[:, 0] = df.iloc[:, 0].astype(str).map(normalize)
    logging.info("Loaded %d Excel Q&A rows.", len(df))
    return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

qa_data = load_qa()

# Build / rebuild FAISS index from PDFs

def build_vectors():
    app.config.pop("vectors", None)
    docs = PyPDFDirectoryLoader(UPLOAD_DIR).load() if os.path.isdir(UPLOAD_DIR) else []
    logging.info("Loaded %d PDF pages.", len(docs))
    if not docs:
        logging.warning("No PDFs to index.")
        return
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)
    app.config["vectors"] = FAISS.from_documents(chunks, STEmbeddings())
    logging.info("✅ Vector index built with %d chunks.", len(chunks))

build_vectors()

# Shorten long answers in a kid‑friendly way

def shorten(raw: str, limit: int = 80) -> str:
    if len(raw.split()) <= limit or not llm:
        return raw
    try:
        short = llm.invoke(
            f"Please rewrite the following answer in no more than {limit} words, kid‑friendly:\n\n{raw}"
        )
        return short.content if hasattr(short, "content") else short
    except Exception:
        return raw

# Turn plain text into HTML if user asked for lists or tables

def postprocess(question: str, raw: str) -> str:
    raw = shorten(raw)

    if re.search(r"\b(table|tabulate|in a table)\b", question, re.I):
        rows = [r.strip() for r in re.split(r"\n|;", raw) if r.strip()]
        data = []
        for r in rows:
            if ":" in r:
                k, v = map(str.strip, r.split(":", 1))
                data.append({"Item": k, "Value": v})
            else:
                data.append({"Item": r})
        return pd.DataFrame(data).to_html(index=False, border=0, classes="table")

    if re.search(r"\b(list|bullet points|bullets)\b", question, re.I):
        items = [i.strip("•- ").strip() for i in re.split(r"\n|;", raw) if i.strip()]
        return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"

    return raw.replace("\n", "<br>")

# ── CORE ANSWER LOGIC ────────────────────────────────────────────────────────

def answer(user_q: str) -> str:
    q_clean = user_q.lower().strip()

    if re.search(r"\b(who (are|r) (you|u)|what('?s| is) your name|introduce yourself)\b", q_clean):
        return postprocess(user_q, f"Hi! I’m <b>{BOT_NAME}</b>, Vincent’s friendly chatbot assistant. 😊")
    if q_clean in {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}:
        return postprocess(user_q, "🙋🏾‍♂️ Hello Beloved! How can I help you today?")

    key = normalize(user_q)
    if key in qa_data:
        logging.info("📄 Excel exact hit for: %s", key)
        return postprocess(user_q, qa_data[key])
    close = get_close_matches(key, qa_data.keys(), n=1, cutoff=0.85)
    if close:
        logging.info("📄 Excel fuzzy hit for: %s → %s", key, close[0])
        return postprocess(user_q, qa_data[close[0]])

    if "vectors" in app.config and app.config["vectors"] and llm:
        try:
            logging.info("🔍 Trying vector search for: %s", user_q)
            chain = create_retrieval_chain(
                retriever=app.config["vectors"].as_retriever(search_type="similarity", k=4),
                combine_documents_chain=create_stuff_documents_chain(llm, prompt)
            )
            result = chain.invoke({"input": user_q})
            logging.info("🔍 Retrieval keys: %s", list(result.keys()))
            answer_txt = result.get("answer") or result.get("result") or result.get("output_text")
            if answer_txt:
                logging.info("✅ Served from PDF vectors.")
                return postprocess(user_q, answer_txt)
            logging.warning("⚠️ Vector search returned no answer text.")
        except Exception as e:
            logging.error("Retrieval chain failed: %s", e)

    if llm:
        try:
            logging.info("🤔 Falling back to LLM for: %s", user_q)
            raw = llm.invoke(user_q)
            raw = raw.content if hasattr(raw, "content") else raw
            return postprocess(user_q, raw)
        except Exception as e:
            logging.error("LLM fallback failed: %s", e)

    return postprocess(user_q, "Sorry, I don't know yet.")

# ── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    room = request.args.get("chat", "default")
    chats.setdefault(room, {"title": room, "messages": []})
    return render_template(
        "index.html",
        titles={cid: c["title"] for cid, c in chats.items()},
        current_chat=room,
        history=chats[room]["messages"],
    )

@app.route("/new")
def new_chat():
    cid = datetime.utcnow().strftime("%Y%m%d%H%M%S") + "-" + str(random.randint(100,999))
    chats[cid] = {"title": "Untitled chat", "messages": []}
    return redirect(url_for("home", chat=cid))

@app.route("/upload", methods=["POST"])
def upload_pdf():
    pdf = request.files.get("pdf")
    if not pdf or not pdf.filename.lower().endswith(".pdf"):
        return "Bad file", 400
    fname = f"{uuid.uuid4()}.pdf"
    pdf.save(os.path.join(UPLOAD_DIR, fname))
    build_vectors()
    return "OK", 200

# ── SOCKET.IO EVENTS ─────────────────────────────────────────────────────────

@socketio.on("join")
def on_join(data):
    room = data.get("room", "default")
    join_room(room)
    emit("message",
         {"role": "bot", "html": f"✅ Joined room <b>{room}</b>. we are to help you!"},
         room=room)

@socketio.on("send")
def on_send(data):
    room = data.get("room", "default")
    text = data.get("text", "")

    if chats[room]["title"] in {"Untitled chat", room}:
        chats[room]["title"] = (text[:30] + ("…" if len(text) > 30 else ""))

    emit("message", {"role": "user", "html": text}, room=room)
    bot_html = answer(text)
    emit("message", {"role": "bot", "html": bot_html}, room=room)
    chats[room]["messages"].extend([
        {"role": "user", "text": text},
        {"role": "bot",  "text": bot_html},
    ])

# ── LAUNCH SERVER ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running on http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000)
