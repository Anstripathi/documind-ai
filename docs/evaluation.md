# Evaluation plan

**No results are reported here.** Scores must come from running the script on your own labelled data.

1. Choose 1–2 PDFs (e.g. a lecture note, a research paper). Don't commit private PDFs.
2. Copy `evaluation/dataset_template.json`; write 20–30 questions. Label `relevant_pages` by reading the PDF.
   Include ~5 `not_found` questions.
3. Retrieval: `python evaluation/evaluate_retrieval.py --pdf my.pdf --dataset evaluation/my_dataset.json --k 5`
   → Recall@k (hit-rate: any relevant page in top-k) and MRR. Try k = 1, 3, 5 and chunk sizes 500/800/1200.
4. Answer quality (manual, with LLM configured). For each question record:

| # | Question | Expected | Behaviour OK? (answer/not_found) | Faithful to evidence? (Y/N) | Citations correct? (Y/N) | Notes |
|---|---|---|---|---|---|---|

5. Tune `min_score` by checking the scores of your `not_found` questions vs answerable ones.
6. Document failure cases (tables, follow-ups, numeric reasoning) in your report.
