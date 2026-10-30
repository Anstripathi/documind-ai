"""Compute Recall@k and MRR on YOUR labelled dataset. Run from the project root:

    python evaluation/evaluate_retrieval.py --pdf my.pdf --dataset evaluation/my_dataset.json --k 5

Dataset format: see evaluation/dataset_template.json. Only questions with
"expected_behavior": "answer" and non-empty "relevant_pages" are scored.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import chunk_document  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.embeddings import SentenceTransformerEmbedder  # noqa: E402
from src.evaluation import mean, recall_at_k, reciprocal_rank  # noqa: E402
from src.pdf_loader import load_pdf  # noqa: E402
from src.retrieval import Retriever  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    s = load_settings()
    data = Path(args.pdf).read_bytes()
    doc = load_pdf(data, Path(args.pdf).name, s.max_file_bytes, s.max_pages)
    retriever = Retriever(SentenceTransformerEmbedder(s.embedding_model))
    retriever.build(chunk_document(doc, s.chunk_size, s.chunk_overlap))

    items = [q for q in json.loads(Path(args.dataset).read_text(encoding="utf-8"))["questions"]
             if q.get("expected_behavior") == "answer" and q.get("relevant_pages")]
    if not items:
        sys.exit("No scorable questions found (need expected_behavior=answer and relevant_pages).")

    recalls, rrs = [], []
    for q in items:
        pages = [r.chunk.page for r in retriever.retrieve(q["question"], args.k)]
        rel = set(q["relevant_pages"])
        recalls.append(recall_at_k(pages, rel, args.k))
        rrs.append(reciprocal_rank(pages, rel))
        print(f"{q['question'][:60]!r:64} retrieved={pages} relevant={sorted(rel)}")
    print(f"\nN={len(items)}  Recall@{args.k}={mean(recalls):.3f}  MRR={mean(rrs):.3f}")


if __name__ == "__main__":
    main()
