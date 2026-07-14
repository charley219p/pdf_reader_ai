import os
import re
import tempfile
import shutil
import hashlib
import time

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="PDF Chat • RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6C63FF, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
    }
    .app-subtitle {
        color: #9aa0a6;
        font-size: 1rem;
        margin-top: 0rem;
        margin-bottom: 1.5rem;
    }
    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-ready {
        background-color: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
        border: 1px solid rgba(46, 204, 113, 0.4);
    }
    .status-empty {
        background-color: rgba(231, 76, 60, 0.15);
        color: #e74c3c;
        border: 1px solid rgba(231, 76, 60, 0.4);
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    .stChatMessage {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Persistent storage layout
# --------------------------------------------------------------------------
# All collections live under chroma_db/<slug>_<hash8>/, next to this script,
# so they survive app restarts and each uploaded file keeps its own history.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_BASE_DIR = os.path.join(APP_DIR, "chroma_db")
os.makedirs(CHROMA_BASE_DIR, exist_ok=True)


def _slugify(name: str) -> str:
    stem, _ = os.path.splitext(name)
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_").lower()
    return slug or "document"


def get_collection_dir(file_bytes: bytes, filename: str) -> str:
    """A stable, unique folder for this exact file's content."""
    content_hash = hashlib.sha256(file_bytes).hexdigest()[:8]
    folder_name = f"{_slugify(filename)}_{content_hash}"
    return os.path.join(CHROMA_BASE_DIR, folder_name)


def list_known_documents():
    """Previously processed documents available under CHROMA_BASE_DIR."""
    if not os.path.isdir(CHROMA_BASE_DIR):
        return []
    return sorted(
        d for d in os.listdir(CHROMA_BASE_DIR)
        if os.path.isdir(os.path.join(CHROMA_BASE_DIR, d))
    )


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "chroma_dir" not in st.session_state:
    st.session_state.chroma_dir = None

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.
            Use only the provided context to answer the question.
            If the answer is not present in the context,
            say: "I could not find the answer in the document."
            """,
        ),
        (
            "human",
            """Context:
            {context}

            Question:
            {question}
            """,
        ),
    ]
)


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model_name="mistral-small-latest")


# Free-tier Gemini embedding quota is 100 requests/minute. Keep batches small
# and pace them so we stay comfortably under that limit.
EMBED_BATCH_SIZE = 10
MAX_RETRIES = 6


def _add_with_retry(vectorstore, batch):
    """Add a batch of documents to the vectorstore, retrying with backoff on 429s."""
    delay = 5
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            vectorstore.add_documents(batch)
            return
        except Exception as e:
            msg = str(e)
            is_rate_limit = "RESOURCE_EXHAUSTED" in msg or "429" in msg
            if not is_rate_limit or attempt == MAX_RETRIES:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)


def load_existing_collection(collection_dir: str):
    """Load a previously-persisted Chroma collection without re-embedding."""
    embedding_model = get_embedding_model()
    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=collection_dir,
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    st.session_state.vectorstore = vectorstore
    st.session_state.retriever = retriever
    st.session_state.chroma_dir = collection_dir
    st.session_state.processed_file = os.path.basename(collection_dir)
    st.session_state.messages = []
    return vectorstore._collection.count()


def process_pdf(uploaded_file, progress_callback=None):
    """Save the uploaded PDF, chunk it, embed it in small batches, and persist
    it under a unique folder for this file's content. If this exact file was
    already processed before, reuse the existing embeddings instead of
    re-embedding (saves quota and time)."""
    file_bytes = uploaded_file.getbuffer()
    collection_dir = get_collection_dir(bytes(file_bytes), uploaded_file.name)

    already_exists = os.path.isdir(collection_dir) and os.listdir(collection_dir)
    if already_exists:
        return load_existing_collection(collection_dir), True

    os.makedirs(collection_dir, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="pdfchat_")
    pdf_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(file_bytes)

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embedding_model = get_embedding_model()

    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=collection_dir,
    )

    total = len(chunks)
    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        _add_with_retry(vectorstore, batch)

        if progress_callback:
            done = min(i + EMBED_BATCH_SIZE, total)
            progress_callback(done, total)

        # Small pause between batches to stay under the per-minute quota.
        if i + EMBED_BATCH_SIZE < total:
            time.sleep(2)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )

    st.session_state.vectorstore = vectorstore
    st.session_state.retriever = retriever
    st.session_state.chroma_dir = collection_dir
    st.session_state.processed_file = uploaded_file.name
    st.session_state.messages = []

    return total, False




def answer_question(query: str) -> str:
    retriever = st.session_state.retriever
    llm = get_llm()

    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = PROMPT.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)

    content = response.content
    if isinstance(content, list):
        # Some providers can return a list of content blocks (str or dict)
        # instead of a plain string; flatten it into text.
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        content = "".join(parts)

    return content


# --------------------------------------------------------------------------
# Sidebar — upload & controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📁 Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        already_loaded_this_session = st.session_state.processed_file == uploaded_file.name
        if not already_loaded_this_session:
            if st.button("🚀 Process PDF", use_container_width=True, type="primary"):
                progress_bar = st.progress(0.0, text="Reading and chunking document...")

                def _update_progress(done, total):
                    progress_bar.progress(
                        done / total, text=f"Embedding chunks... {done}/{total}"
                    )

                try:
                    n_chunks, from_cache = process_pdf(
                        uploaded_file, progress_callback=_update_progress
                    )
                    progress_bar.empty()
                    if from_cache:
                        st.success(f"Loaded existing embeddings ({n_chunks} chunks) ⚡")
                    else:
                        st.success(f"Processed into {n_chunks} chunks ✅")
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Something went wrong: {e}")
        else:
            st.info(f"'{uploaded_file.name}' is already loaded.")

    # ----------------------------------------------------------------------
    # Previously processed documents (persisted on disk under chroma_db/)
    # ----------------------------------------------------------------------
    known_docs = list_known_documents()
    if known_docs:
        st.markdown("---")
        st.markdown("### 🕘 Previously processed")
        choice = st.selectbox(
            "Load a document without re-uploading",
            options=["— select —"] + known_docs,
            label_visibility="collapsed",
        )
        if choice != "— select —":
            if st.button("📂 Load selected", use_container_width=True):
                collection_dir = os.path.join(CHROMA_BASE_DIR, choice)
                try:
                    count = load_existing_collection(collection_dir)
                    st.success(f"Loaded '{choice}' ({count} chunks) ⚡")
                except Exception as e:
                    st.error(f"Could not load: {e}")

    st.markdown("---")

    if st.session_state.retriever is not None:
        st.markdown(
            f'<span class="status-pill status-ready">● Ready — {st.session_state.processed_file}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pill status-empty">● No document loaded</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    with st.expander("⚙️ Retrieval settings info"):
        st.write("Search type: **MMR**")
        st.write("Top-k results: **4**")
        st.write("Fetch-k candidates: **10**")
        st.write("Lambda (diversity): **0.5**")

# --------------------------------------------------------------------------
# Main area — chat
# --------------------------------------------------------------------------
st.markdown('<div class="app-title">📄 PDF Chat Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Upload a PDF and ask questions grounded in its content.</div>',
    unsafe_allow_html=True,
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.retriever is None:
    st.info("👈 Upload and process a PDF from the sidebar to start chatting.")
else:
    query = st.chat_input("Ask something about your document...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = answer_question(query)
                except Exception as e:
                    answer = f"⚠️ Error while generating answer: {e}"
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})