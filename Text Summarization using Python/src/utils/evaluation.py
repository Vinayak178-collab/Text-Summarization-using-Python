from __future__ import annotations

from typing import Dict, List

from rouge_score import rouge_scorer


def compute_rouge(
    references: List[str],
    candidates: List[str],
    use_stemmer: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compute average ROUGE scores over a list of reference and candidate summaries.

    Returns a dict of metric -> {precision, recall, fmeasure}.
    """
    if len(references) != len(candidates):
        raise ValueError("references and candidates must have the same length.")

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=use_stemmer,
    )

    agg_scores = {
        "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
        "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
        "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
    }

    n = len(references)
    for ref, cand in zip(references, candidates):
        scores = scorer.score(ref, cand)
        for key in agg_scores.keys():
            agg_scores[key]["precision"] += scores[key].precision
            agg_scores[key]["recall"] += scores[key].recall
            agg_scores[key]["fmeasure"] += scores[key].fmeasure

    for key in agg_scores.keys():
        agg_scores[key]["precision"] /= n
        agg_scores[key]["recall"] /= n
        agg_scores[key]["fmeasure"] /= n

    return agg_scores


