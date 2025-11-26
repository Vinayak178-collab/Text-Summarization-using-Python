from __future__ import annotations

import re
from typing import List

import nltk


def ensure_nltk_punkt() -> None:
    """
    Ensure that the NLTK punkt tokenizer models are available.
    This is safe to call multiple times.
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def basic_clean(text: str) -> str:
    """
    Basic text cleaning: normalize whitespace and remove obvious control chars.
    Keep punctuation and casing to preserve information for summarization.
    """
    # Remove non-printable control characters (except common whitespace)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = normalize_whitespace(text)
    return text


def sentence_split(text: str) -> List[str]:
    """
    Split text into sentences using NLTK's punkt tokenizer.
    """
    ensure_nltk_punkt()
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    # Filter out very short / empty sentences
    return [s.strip() for s in sentences if s and not s.isspace()]


