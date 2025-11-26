from __future__ import annotations

from pathlib import Path

from src.summarizers.extractive import centroid_extractive_summary
from src.utils.evaluation import compute_rouge


def main() -> None:
    base = Path(__file__).resolve().parents[1] / "examples"
    article_path = base / "sample_article.txt"
    ref_path = base / "sample_summary.txt"

    article = article_path.read_text(encoding="utf-8")
    reference = ref_path.read_text(encoding="utf-8")

    result = centroid_extractive_summary(article, num_sentences=2)
    candidate = result.summary

    scores = compute_rouge([reference], [candidate])
    print("Reference summary:\n", reference)
    print("\nModel summary:\n", candidate)
    print("\nROUGE scores:")
    for metric, vals in scores.items():
        print(
            f"{metric}: P={vals['precision']:.3f} "
            f"R={vals['recall']:.3f} F1={vals['fmeasure']:.3f}"
        )


if __name__ == "__main__":
    main()


