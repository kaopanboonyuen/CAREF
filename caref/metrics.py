"""
caref/metrics.py
================
Evaluation metrics used in the CAREF paper:

    • Task accuracy  (exact match of model answer vs. gold label)
    • nBERT          (BERTScore-normalized explanation alignment)

Reference
---------
Panboonyuen, T. (2026). CAREF: Calibration-Aware Regularization for
Explanation Faithfulness Without Rationale Supervision. EMNLP 2026.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task Accuracy
# ---------------------------------------------------------------------------

def compute_accuracy(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Exact-match accuracy (case-insensitive, stripped).

    Parameters
    ----------
    predictions : list[str]
        Model-generated answers.
    references : list[str]
        Gold answers.

    Returns
    -------
    float in [0, 1].
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length.")

    correct = sum(
        p.strip().lower() == r.strip().lower()
        for p, r in zip(predictions, references)
    )
    return correct / len(predictions)


# ---------------------------------------------------------------------------
# nBERT — BERTScore-Normalized Explanation Alignment
# ---------------------------------------------------------------------------

def compute_nbert(
    explanations: Sequence[str],
    references: Sequence[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 32,
    device: str | None = None,
    rescale_with_baseline: bool = True,
) -> dict[str, float]:
    """Compute BERTScore F1 (nBERT) between generated and gold explanations.

    nBERT is the primary explanation-quality metric in Table 1 of the paper.
    It uses DeBERTa-XL-MNLI as the scoring model (consistent with the FEB
    protocol) and returns the mean F1 scaled to [0, 100].

    Parameters
    ----------
    explanations : list[str]
        Generated explanations (one per example).
    references : list[str]
        Gold explanations.
    model_type : str
        HuggingFace model for BERTScore (paper uses DeBERTa-XL-MNLI).
    batch_size : int
        BERTScore batch size.
    device : str, optional
        Inference device (defaults to 'cuda' if available, else 'cpu').
    rescale_with_baseline : bool
        Whether to apply baseline rescaling (True, following FEB).

    Returns
    -------
    dict with keys: 'precision', 'recall', 'f1' (all scaled ×100).
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError as e:
        raise ImportError(
            "bert-score is required for nBERT computation. "
            "Install with: pip install bert-score"
        ) from e

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    P, R, F = bert_score_fn(
        cands=list(explanations),
        refs=list(references),
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        rescale_with_baseline=rescale_with_baseline,
        verbose=False,
    )

    return {
        "precision": float(P.mean()) * 100,
        "recall": float(R.mean()) * 100,
        "f1": float(F.mean()) * 100,
    }


# ---------------------------------------------------------------------------
# Aggregate over 60 FEB splits
# ---------------------------------------------------------------------------

def aggregate_splits(
    per_split_results: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Aggregate per-split results into mean ± std (as in Table 1).

    Parameters
    ----------
    per_split_results : list of dicts, each with metric_name → value.

    Returns
    -------
    dict mapping metric_name → {'mean': float, 'std': float}.

    Example output (matching Table 1 format)
    ----------------------------------------
    {
        'accuracy': {'mean': 84.59, 'std': 1.54},
        'nbert_f1': {'mean': 74.42, 'std': 1.37},
    }
    """
    if not per_split_results:
        return {}

    metrics = per_split_results[0].keys()
    aggregated = {}
    for metric in metrics:
        values = np.array([r[metric] for r in per_split_results])
        aggregated[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return aggregated


def format_results_table(
    aggregated: dict[str, dict[str, float]],
) -> str:
    """Pretty-print results in the style of Table 1 of the paper."""
    lines = ["=" * 50, f"{'Metric':<20}{'Mean':>10}{'±Std':>10}", "=" * 50]
    for metric, stats in aggregated.items():
        lines.append(
            f"{metric:<20}{stats['mean']:>10.2f}{stats['std']:>10.2f}"
        )
    lines.append("=" * 50)
    return "\n".join(lines)
