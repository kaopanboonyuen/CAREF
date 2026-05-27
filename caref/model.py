"""
caref/model.py
==============
CAREF: Model wrapper with PEFT integration and the CAREF-AQ variant.

CAREF-AQ updates *only* the decoder attention query projections
(6.43 % of Flan-T5 parameters), matching or exceeding full fine-tuning
on both accuracy and nBERT (see Table 1 of the paper).

Reference
---------
Panboonyuen, T. (2026). CAREF: Calibration-Aware Regularization for
Explanation Faithfulness Without Rationale Supervision. EMNLP 2026 Submission.
https://kaopanboonyuen.github.io/CAREF
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supported CAREF variants
# ---------------------------------------------------------------------------

CAREF_VARIANTS = {
    # variant_name: (parameter_pattern, trainable_pct_approx)
    "CAREF-BASE":  (None,          "100.00%"),   # full fine-tuning
    "CAREF-DEC":   ("decoder",     " 52.23%"),   # all decoder params
    "CAREF-AQKV":  ("q|k|v",       " 19.28%"),   # query + key + value projections
    "CAREF-LAQ":   ("q",           "  6.44%"),   # all query projections
    "CAREF-AQ":    ("decoder.*q",  "  6.43%"),   # decoder query only  ← RECOMMENDED
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CAREFModelConfig:
    """Model and PEFT configuration.

    Attributes
    ----------
    base_model : str
        HuggingFace model ID (default: "google/flan-t5-large").
    variant : str
        One of the CAREF_VARIANTS keys.
    max_input_length : int
        Maximum tokenized input length.
    max_target_length : int
        Maximum tokenized target length.
    """
    base_model: str = "google/flan-t5-large"
    variant: str = "CAREF-AQ"
    max_input_length: int = 256
    max_target_length: int = 64
    additional_trainable_patterns: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parameter selection helpers
# ---------------------------------------------------------------------------

def _freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def _unfreeze_by_pattern(model: nn.Module, pattern: str) -> int:
    """Unfreeze parameters whose name matches *pattern* (regex).
    Returns the number of newly unfrozen parameters.
    """
    import re
    count = 0
    for name, param in model.named_parameters():
        if re.search(pattern, name):
            param.requires_grad_(True)
            count += param.numel()
    return count


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ---------------------------------------------------------------------------
# CAREF Model
# ---------------------------------------------------------------------------

class CAREFModel(nn.Module):
    """Thin wrapper around Flan-T5 that enforces CAREF-AQ (or another
    CAREF variant) parameter selection and exposes a clean forward pass.

    Usage
    -----
    >>> model, tokenizer = CAREFModel.from_pretrained(CAREFModelConfig())
    >>> outputs = model(input_ids=..., attention_mask=..., labels=...)
    >>> logits = outputs.logits  # (B, T, V) — plug straight into CAREFLoss
    """

    def __init__(self, model_config: CAREFModelConfig):
        super().__init__()
        self.cfg = model_config

        logger.info("Loading backbone: %s", model_config.base_model)
        self.backbone: T5ForConditionalGeneration = (
            T5ForConditionalGeneration.from_pretrained(model_config.base_model)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.base_model)

        self._apply_peft()

        trainable, total = count_parameters(self.backbone)
        logger.info(
            "Variant: %s | Trainable: %s / %s (%.2f%%)",
            model_config.variant,
            f"{trainable:,}",
            f"{total:,}",
            100 * trainable / total,
        )

    # ------------------------------------------------------------------
    # PEFT setup
    # ------------------------------------------------------------------

    def _apply_peft(self) -> None:
        variant = self.cfg.variant
        if variant not in CAREF_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from: {list(CAREF_VARIANTS)}"
            )

        pattern, _ = CAREF_VARIANTS[variant]

        if pattern is None:
            # CAREF-BASE: all params trainable
            return

        _freeze_all(self.backbone)
        unfrozen = _unfreeze_by_pattern(self.backbone, pattern)

        # Optional extra patterns (e.g., task-head LM head)
        for extra in self.cfg.additional_trainable_patterns:
            unfrozen += _unfreeze_by_pattern(self.backbone, extra)

        if unfrozen == 0:
            raise RuntimeError(
                f"Pattern '{pattern}' matched no parameters. Check variant config."
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        """Transparent forward to backbone. Returns a ModelOutput with
        `.logits` of shape (B, T, V)."""
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs):
        return self.backbone.generate(input_ids, **kwargs)

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls, config: CAREFModelConfig
    ) -> tuple["CAREFModel", AutoTokenizer]:
        model = cls(config)
        return model, model.tokenizer

    def save_pretrained(self, path: str) -> None:
        self.backbone.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("Saved CAREF model to %s", path)
