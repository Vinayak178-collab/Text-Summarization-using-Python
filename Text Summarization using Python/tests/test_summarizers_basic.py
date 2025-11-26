from __future__ import annotations

from src.summarizers.extractive import centroid_extractive_summary


def test_extractive_returns_non_empty_summary():
    text = (
        "Text summarization is the process of shortening a text while preserving its main ideas. "
        "It is useful for quickly understanding large documents. "
        "This project implements both extractive and abstractive summarization methods."
    )
    result = centroid_extractive_summary(text, num_sentences=2)
    assert result.summary
    assert len(result.selected_sentences) <= 2


