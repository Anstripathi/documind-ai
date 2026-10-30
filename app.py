"""DocuMind AI - PDF question answering with source-grounded answers (Streamlit)."""
from __future__ import annotations

import hashlib
import os

import streamlit as st

from src.chunking import chunk_document
from src.config import load_settings
from src.embeddings import SentenceTransformerEmbedder
from src.llm_provider import build_provider
from src.pdf_loader import PDFValidationError, load_pdf
from src.rag_pipeline import RAGPipeline
from src.retrieval import Retriever

SETTINGS = load_settings()
st.set_page_config(page_title="DocuMind AI", page_icon="ðŸ“„", layout="wide")

STATUS_LABEL = {
    "answered": "âœ… Answered (citations verified)",
    "unverified": "âš ï¸ Unverified (no valid citation)",
    "not_found": "â„¹ï¸ Not found in document",
    "extractive": "ðŸ“Ž Extractive mode (no LLM)",
    "error": "âŒ Error",
}


@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def get_embedder(model_name: str):
    return SentenceTransformerEmbedder(model_name)


def get_api_key() -> str | None:
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:  # no secrets file configured
        key = None
    return key or os.getenv("GEMINI_API_KEY")


def reset_all() -> None:
    for k in ("doc", "retriever", "doc_key", "messages"):
        st.session_state.pop(k, None)
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1


st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("messages", [])

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Passages to retrieve (top-k)", 1, 10, SETTINGS.top_k)
    chunk_size = st.slider("Chunk size (characters)", 300, 2000, SETTINGS.chunk_size, step=50)
    overlap = st.slider("Chunk overlap (characters)", 0, 500, SETTINGS.chunk_overlap, step=10)
    if overlap >= chunk_size:
        st.error("Overlap must be smaller than chunk size.")
        st.stop()
    st.caption("Changing chunk settings re-processes the document.")
    st.divider()
    api_key = get_api_key()
    provider_override = os.getenv("DOCUMIND_LLM_PROVIDER")

    if provider_override:
        provider_name = provider_override.strip().lower()
        model_name = SETTINGS.llm_model
    elif api_key:
        provider_name = "gemini"
        model_name = os.getenv("DOCUMIND_LLM_MODEL", "gemini-3.8-flash")
    else:
        provider_name = "none"
        model_name = SETTINGS.llm_model

    llm_provider = build_provider(
        api_key,
        model_name,
        provider=provider_name,
        base_url=SETTINGS.ollama_url,
        timeout=SETTINGS.ollama_timeout,
    )
    st.write(
        "LLM:",
        f"?? {llm_provider.name}"
        if llm_provider
        else "?? not configured (extractive mode)",
    )
    if st.button("ðŸ—‘ï¸ Clear document & conversation", use_container_width=True):
        reset_all()
        st.rerun()

# ---------------- Header ----------------
st.title("ðŸ“„ DocuMind AI")
st.write(
    "Upload a PDF and ask questions. Answers are generated **only from retrieved passages** "
    "and cite page numbers, which are checked against the evidence."
)

uploaded = st.file_uploader(
    f"Upload a PDF (max {SETTINGS.max_file_mb} MB, {SETTINGS.max_pages} pages)",
    type=["pdf"],
    key=f"uploader_{st.session_state['uploader_key']}",
)

if uploaded is not None:
    data = uploaded.getvalue()
    key = hashlib.sha256(data).hexdigest() + f"|{chunk_size}|{overlap}"
    if st.session_state.get("doc_key") != key:
        st.session_state["messages"] = []  # new document -> fresh conversation
        try:
            with st.status("Processing document...", expanded=True) as status:
                st.write("Validating and extracting text...")
                doc = load_pdf(data, uploaded.name, SETTINGS.max_file_bytes, SETTINGS.max_pages)
                st.write("Chunking...")
                chunks = chunk_document(doc, chunk_size, overlap)
                if not chunks:
                    raise PDFValidationError("No usable text chunks were produced.")
                st.write(f"Embedding {len(chunks)} chunks and building index...")
                retriever = Retriever(get_embedder(SETTINGS.embedding_model))
                retriever.build(chunks)
                st.session_state.update(doc=doc, retriever=retriever, doc_key=key)
                status.update(label="Document ready", state="complete", expanded=False)
        except PDFValidationError as exc:
            for k in ("doc", "retriever", "doc_key"):
                st.session_state.pop(k, None)
            st.error(str(exc))
        except Exception:
            for k in ("doc", "retriever", "doc_key"):
                st.session_state.pop(k, None)
            st.error("Unexpected error while processing the document.")

doc = st.session_state.get("doc")
retriever = st.session_state.get("retriever")

if doc and retriever:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Document", doc.name)
    c2.metric("Pages", doc.total_pages)
    c3.metric("Chunks", len(retriever.chunks))
    c4.metric("Characters", f"{doc.char_count:,}")
    if doc.empty_pages:
        st.warning(f"No extractable text on page(s): {doc.empty_pages[:20]}"
                   f"{'...' if len(doc.empty_pages) > 20 else ''} (possibly scanned images).")

    for msg in st.session_state["messages"]:
        with st.chat_message("user"):
            st.write(msg.question)
        with st.chat_message("assistant"):
            st.caption(STATUS_LABEL.get(msg.status, msg.status))
            st.markdown(msg.text)
            if msg.warning:
                st.warning(msg.warning)
            if msg.evidence:
                with st.expander(f"Evidence ({len(msg.evidence)} passages)"):
                    for r in msg.evidence:
                        mark = "âœ… cited" if r.chunk.page in msg.cited_pages else ""
                        st.markdown(f"**#{r.rank} Â· {r.chunk.doc_name} Â· Page {r.chunk.page}** "
                                    f"Â· similarity {r.score:.3f} {mark}")
                        st.text(r.chunk.text)  # plain text: never render document content as markdown

    question = st.chat_input("Ask a question about the document...")
    if question:
        pipeline = RAGPipeline(
            retriever,
            llm_provider,
            top_k=top_k,
            min_score=SETTINGS.min_score,
            max_question_chars=SETTINGS.max_question_chars,
        )
        with st.spinner("Searching document and generating answer..."):
            st.session_state["messages"].append(pipeline.answer(question))
        st.rerun()
else:
    st.info("Upload a PDF to begin.")

with st.expander("Methodology & limitations"):
    st.markdown(
        """
**Pipeline:** PDF â†’ validation â†’ page-wise text extraction â†’ chunking (never crossing pages)
â†’ embeddings (`all-MiniLM-L6-v2`) â†’ vector index â†’ top-k retrieval â†’ LLM restricted to the
retrieved passages â†’ citation validation.

**Limitations**
- Similarity scores are cosine similarities, **not probabilities** of being correct.
- Scanned PDFs (images) are not supported; there is no OCR.
- Tables and multi-column layouts may extract poorly.
- Each question is retrieved independently, so follow-ups that rely on earlier turns
  (e.g. "and what about the second one?") may fail.
- Citation validation checks that cited pages were retrieved. It does **not** prove the
  claim is true. Always inspect the evidence.
- The document is processed in memory and discarded when you clear it or close the session.
        """
    )

