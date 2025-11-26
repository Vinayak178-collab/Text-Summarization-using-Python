from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from transformers import pipeline

from src.preprocessing.text_cleaning import basic_clean


_abstractive_pipeline = None


def get_abstractive_pipeline(
    model_name: str = "facebook/bart-large-cnn",
    device: int = -1,
):
    """
    Lazy-load a HuggingFace summarization pipeline.
    """
    global _abstractive_pipeline
    if _abstractive_pipeline is None:
        _abstractive_pipeline = pipeline(
            "summarization",
            model=model_name,
            device=device,
        )
    return _abstractive_pipeline


@dataclass
class AbstractiveSummaryResult:
    summary: str
    chunks: List[str]
    chunk_summaries: List[str]


def _chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    """
    Simple character-based chunking with overlap.
    This avoids token-level dependencies but works reasonably for long documents.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def abstractive_summary(
    text: str,
    max_length: int = 130,
    min_length: int = 30,
    model_name: str = "facebook/bart-large-cnn",
    device: int = -1,
) -> AbstractiveSummaryResult:
    """
    Abstractive summarization with optional chunking for long texts.
    """
    cleaned = basic_clean(text)
    summarizer = get_abstractive_pipeline(model_name=model_name, device=device)

    chunks = _chunk_text(cleaned)
    chunk_summaries: List[str] = []

    for chunk in chunks:
        result = summarizer(
            chunk,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )
        chunk_summaries.append(result[0]["summary_text"])

    if len(chunk_summaries) == 1:
        final_summary = chunk_summaries[0]
    else:
        combined = " ".join(chunk_summaries)
        result = summarizer(
            combined,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )
        final_summary = result[0]["summary_text"]

    return AbstractiveSummaryResult(
        summary=final_summary,
        chunks=chunks,
        chunk_summaries=chunk_summaries,
    )


