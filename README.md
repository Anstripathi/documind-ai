# DocuMind AI — PDF Question Answering with Source Citations

A Retrieval-Augmented Generation (RAG) app: upload a PDF, ask questions, get answers grounded in
retrieved passages with **page citations that are validated against the evidence**.

## Status honesty
Core logic (chunking, retrieval, citation validation, pipeline, metrics) has automated tests.
PDF loading tests, the Streamlit UI, the real embedding model and the real LLM call have **not**
been run by the author of this scaffold; run the checklist below on your machine first.

## Run locally (Windows PowerShell, Python 3.11)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -v
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml   # then edit: add your key
streamlit run app.py
```
Without an API key the app still works in **extractive mode** (shows top passages, clearly labelled).
Get a Gemini key from Google AI Studio (check its current free-tier limits).

## Architecture
`PDF → validation → page-wise extraction → chunking → embeddings → vector index → top-k →
LLM (evidence only) → citation validation → answer + evidence`. See `docs/architecture.md`.

## Project layout
```
app.py                 Streamlit UI
src/pdf_loader.py      validation + PyMuPDF extraction
src/chunking.py        page-aware chunking
src/embeddings.py      Sentence-Transformers embedder
src/retrieval.py       FAISS/NumPy index + Retriever
src/llm_provider.py    provider interface + Gemini
src/rag_pipeline.py    prompt, gating, answer assembly
src/citations.py       citation extraction/validation
src/evaluation.py      Recall@k, MRR
evaluation/            evaluation script + dataset template
tests/                 pytest suite
docs/                  architecture, security, evaluation
```

## Publish to GitHub
```powershell
git init
git add .
git status                      # confirm no .pdf, .env, secrets.toml, .venv
git commit -m "Initial commit: DocuMind AI"
git branch -M main
git remote add origin https://github.com/<your-username>/documind-ai.git
git push -u origin main
```

## Deploy on Streamlit Community Cloud
1. share.streamlit.io → New app → pick your repo, branch `main`, main file `app.py`.
2. Advanced settings → Python 3.11; Secrets: `GEMINI_API_KEY = "..."`.
3. If the build runs out of memory/time because of PyTorch, see `docs/architecture.md` (deployment notes).

## Known limitations
No OCR; tables/multi-column text may extract poorly; follow-up questions are not rewritten into
standalone queries; citation validation proves a cited page was retrieved, not that the claim is true.
