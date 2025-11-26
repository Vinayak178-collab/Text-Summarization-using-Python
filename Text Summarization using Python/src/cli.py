from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.preprocessing.pdf_loader import extract_text_from_pdf_bytes
from src.summarizers.abstractive import abstractive_summary
from src.summarizers.extractive import (
    centroid_extractive_summary,
    textrank_extractive_summary,
)


def _read_input_text(path: str | None) -> str:
    if path:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"File not found: {p}")
        if p.suffix.lower() == ".pdf":
            data = p.read_bytes()
            return extract_text_from_pdf_bytes(data)
        return p.read_text(encoding="utf-8")
    # Read from stdin
    return sys.stdin.read()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Text Summarizer CLI")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to input file (.txt or .pdf). If omitted, read from stdin.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="extractive",
        choices=["extractive", "abstractive"],
        help="Summarization mode.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="centroid",
        choices=["centroid", "textrank"],
        help="Extractive method (when mode=extractive).",
    )
    parser.add_argument(
        "-n",
        "--num-sentences",
        type=int,
        default=3,
        help="Number of sentences for extractive summary.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=30,
        help="Min length for abstractive summary.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=130,
        help="Max length for abstractive summary.",
    )

    args = parser.parse_args(argv)

    text = _read_input_text(args.input)
    if not text.strip():
        raise SystemExit("No input text provided.")

    if args.mode == "extractive":
        if args.method == "centroid":
            result = centroid_extractive_summary(text, num_sentences=args.num_sentences)
        else:
            result = textrank_extractive_summary(text, num_sentences=args.num_sentences)
        print(result.summary)
    else:
        result = abstractive_summary(
            text,
            max_length=args.max_length,
            min_length=args.min_length,
        )
        print(result.summary)


if __name__ == "__main__":
    main()


