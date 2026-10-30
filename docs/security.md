# Security and privacy

| Risk | Control | Verified how |
|---|---|---|
| Malicious/oversized upload | Extension, `%PDF-` header, size cap, page cap, password/corrupt handling | `tests/test_pdf_loader.py` (needs PyMuPDF; run locally) |
| Path tricks in filename | `os.path.basename` | test |
| Execution of document content | Text extraction only; no links followed, no code run, no tools given to LLM | by design |
| Prompt injection | Passages delimited by `<passage>` tags, delimiter tags escaped, system rule 2, extra rule: NOT_FOUND | `test_injection_text_is_delimited_and_escaped_in_prompt` checks prompt construction only. **Real-model resistance is untested** — try it manually. |
| Remote image beacons from output | `sanitize_markdown` removes images; evidence shown with `st.text` | test |
| Secrets | `st.secrets` / env var; `.gitignore` excludes `.env`, `secrets.toml` | review before every push |
| Session isolation | Per-session `Retriever` | `test_sessions_are_isolated` |
| Retention | No disk writes; data in session memory; Clear button + new-upload reset | manual |
| Third-party disclosure | Retrieved passages and the question are sent to the LLM provider | **Tell users; don't upload confidential documents** |
| Dependency CVEs | Not scanned. Run `pip install pip-audit; pip-audit` | todo |
