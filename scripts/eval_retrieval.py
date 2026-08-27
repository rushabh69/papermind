"""Retrieval eval on the sample PDF: P@5, R@5, MRR for dense vs hybrid, with and
without re-ranking. Swap GOLD for your own labels to test another document."""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import rag, retrieval  # noqa: E402

# question -> set of pages that contain the answer (for data/sample.pdf)
GOLD = [
    ("How many ratings are in the dataset?", {2}),
    ("What rating scale is used for the ratings?", {2}),
    ("How many users and movies are there?", {2}),
    ("How is similarity between users computed?", {2}),
    ("How many latent factors does the matrix factorization use?", {2}),
    ("What RMSE did the SVD model achieve?", {3}),
    ("Which model had the best precision at ten?", {3}),
    ("What was the popularity baseline precision at ten?", {3}),
    ("How long did the matrix factorization take to train?", {3}),
    ("What is the main contribution of the paper?", {1}),
    ("What does collaborative filtering do?", {1}),
    ("What is the future work?", {4}),
]

CONFIGS = [
    ("dense, no rerank",  dict(mode="dense", use_rerank=False)),
    ("hybrid, no rerank", dict(mode="hybrid", use_rerank=False)),
    ("dense + rerank",    dict(mode="dense", use_rerank=True)),
    ("hybrid + rerank",   dict(mode="hybrid", use_rerank=True)),
]
K = 5


def evaluate(doc_id: int, cfg: dict):
    P = R = MRR = 0.0
    for q, gold in GOLD:
        hits = retrieval.retrieve(doc_id, q, top_k=K, **cfg)
        pages = [h["page_number"] for h in hits]
        flags = [p in gold for p in pages]
        P += sum(flags) / K
        R += len({p for p in pages if p in gold}) / len(gold)
        rr = next((1.0 / (i + 1) for i, f in enumerate(flags) if f), 0.0)
        MRR += rr
    n = len(GOLD)
    return P / n, R / n, MRR / n


def main():
    rag.init()
    sample = Path("data/sample.pdf")
    if not sample.exists():
        print("data/sample.pdf not found - add a sample PDF to evaluate.")
        return
    doc_id = rag.ingest_pdf(sample.read_bytes(), "sample_eval.pdf")
    try:
        print(f"Retrieval evaluation on {len(GOLD)} questions (k={K})\n")
        print(f"{'config':22} {'P@5':>7} {'R@5':>7} {'MRR':>7}")
        print("-" * 46)
        for name, cfg in CONFIGS:
            p, r, m = evaluate(doc_id, cfg)
            print(f"{name:22} {p:7.3f} {r:7.3f} {m:7.3f}")
        print("\nHigher is better. Expect hybrid + rerank to be strongest.")
    finally:
        rag.delete_document(doc_id)   # keep the dev DB clean


if __name__ == "__main__":
    main()