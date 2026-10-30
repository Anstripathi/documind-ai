# Architecture and design decisions (viva notes)

| Stage | Choice | Why / trade-off |
|---|---|---|
| Extraction | PyMuPDF, page by page | Fast, in-memory (`fitz.open(stream=...)`), keeps page numbers. No OCR. |
| Chunking | Characters, 800 size / 150 overlap, never crossing pages | Guarantees chunk→page mapping. Overlap reduces boundary loss. Splits content that spans pages. |
| Embeddings | `all-MiniLM-L6-v2` (384-d, ~90 MB) | Small enough for free hosting. Weaker than large models; English-focused. |
| Index | FAISS `IndexFlatIP` on normalised vectors (= cosine); NumPy fallback | Exact search is fine for hundreds of chunks. Approximate indexes add complexity for no gain here. |
| Threshold | `min_score` 0.20 (configurable) | Refuses to call the LLM when nothing is similar. **0.20 is an untuned starting value** — tune it with your evaluation set. Scores are not probabilities. |
| LLM | `LLMProvider` interface; Gemini implementation | Retrieval is vendor-independent. Errors mapped to Auth / RateLimit / Unavailable. |
| Fallback | Extractive mode when no key | Clearly labelled; never pretends to be generated. |
| Citations | Model must write `[Page N]`; validator removes pages not in retrieved set | Prevents invented page references. Does not prove the claim is true. |
| Isolation | One `Retriever` per Streamlit session (`st.session_state`) | No global index, so no cross-user retrieval. |

## Known weaknesses
- Each question is retrieved independently (no query rewriting for follow-ups).
- Character chunking ignores sentence/section structure.
- Regex citation parsing can miss unusual formats (e.g. page ranges) — such cites are simply not credited.

## Deployment notes
- `sentence-transformers` pulls in PyTorch; the default wheel can be large. If the Cloud build fails
  or the app exceeds memory, options: (a) pin a CPU-only torch via an extra index URL,
  (b) switch the embedder to a hosted embeddings API behind the same `Embedder` interface.
- The embedding model downloads on first run; `st.cache_resource` keeps it for the process lifetime.
- Pin exact versions after your first successful install.
