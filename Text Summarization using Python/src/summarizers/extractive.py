from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer, util

from src.preprocessing.text_cleaning import basic_clean, sentence_split


_sentence_model: Optional[SentenceTransformer] = None


def get_sentence_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer(model_name)
    return _sentence_model


@dataclass
class ExtractiveSummaryResult:
    summary: str
    selected_sentences: List[str]
    indices: List[int]
    scores: List[float]


def _build_similarity_matrix(sentences: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Build a cosine similarity matrix between sentence embeddings.
    """
    embeddings = model.encode(sentences, convert_to_tensor=True).cpu().numpy()
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normed = embeddings / norm
    sim_matrix = np.dot(normed, normed.T)
    np.fill_diagonal(sim_matrix, 0.0)
    return sim_matrix


def centroid_extractive_summary(
    text: str,
    num_sentences: int = 3,
    model_name: str = "all-MiniLM-L6-v2",
) -> ExtractiveSummaryResult:
    """
    Extractive summarization based on sentence embeddings and document centroid.
    """
    cleaned = basic_clean(text)
    sentences = sentence_split(cleaned)
    if not sentences:
        return ExtractiveSummaryResult(summary="", selected_sentences=[], indices=[], scores=[])

    model = get_sentence_model(model_name)
    embeddings = model.encode(sentences, convert_to_tensor=True)
    doc_centroid = embeddings.mean(dim=0)
    scores_tensor = util.cos_sim(embeddings, doc_centroid).cpu().numpy().squeeze()

    if scores_tensor.ndim == 0:
        scores_tensor = np.array([float(scores_tensor)])

    ranked_idx = np.argsort(-scores_tensor)
    k = min(num_sentences, len(sentences))
    selected_idx = sorted(ranked_idx[:k].tolist())

    selected_sentences = [sentences[i] for i in selected_idx]
    selected_scores = [float(scores_tensor[i]) for i in selected_idx]
    summary = " ".join(selected_sentences)

    return ExtractiveSummaryResult(
        summary=summary,
        selected_sentences=selected_sentences,
        indices=selected_idx,
        scores=selected_scores,
    )


def textrank_extractive_summary(
    text: str,
    num_sentences: int = 3,
    model_name: str = "all-MiniLM-L6-v2",
) -> ExtractiveSummaryResult:
    """
    TextRank-style extractive summarization using sentence embeddings
    as similarity features for a PageRank graph.
    """
    cleaned = basic_clean(text)
    sentences = sentence_split(cleaned)
    if not sentences:
        return ExtractiveSummaryResult(summary="", selected_sentences=[], indices=[], scores=[])

    model = get_sentence_model(model_name)
    sim_matrix = _build_similarity_matrix(sentences, model)

    # Build graph
    graph = nx.from_numpy_array(sim_matrix)
    scores_dict = nx.pagerank(graph, max_iter=100, tol=1.0e-6)

    # Convert to ordered arrays
    indices = list(scores_dict.keys())
    scores = np.array([scores_dict[i] for i in indices])
    ranked_idx = np.argsort(-scores)
    k = min(num_sentences, len(sentences))
    selected_idx = sorted(ranked_idx[:k].tolist())

    selected_sentences = [sentences[i] for i in selected_idx]
    selected_scores = [float(scores[i]) for i in selected_idx]
    summary = " ".join(selected_sentences)

    return ExtractiveSummaryResult(
        summary=summary,
        selected_sentences=selected_sentences,
        indices=selected_idx,
        scores=selected_scores,
    )



