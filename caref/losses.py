"""
caref/losses.py
===============
CAREF: Calibration-Aware Regularization for Explanation Faithfulness
EMNLP 2026 Submission — Official Reproduction Code

Implements the three-term training objective:

    L_CAREF = L_CE + λ_SCED · L_SCED + λ_KL · L_KL

where L_SCED is the novel Sparsity-Calibrated Entropic Divergence:

    L_SCED = Σ_t Σ_v |P_{t,v} log(P_{t,v}/U_v)|^α · (1 - P_{t,v})^β

Reference
---------
Panboonyuen, T. (2026). CAREF: Calibration-Aware Regularization for
Explanation Faithfulness Without Rationale Supervision. EMNLP 2026 Submission.
https://kaopanboonyuen.github.io/CAREF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Hyperparameter container
# ---------------------------------------------------------------------------

@dataclass
class CAREFConfig:
    """All CAREF regularization hyperparameters in one place.

    Attributes
    ----------
    alpha : float
        Entropic curvature exponent (α ≥ 1).
        α=1 → standard KL contribution; α>1 → super-linear penalty on
        large deviations from uniform.
    beta : float
        Adaptive sparsity exponent (β ≥ 0).
        β=0 → all tokens equally penalized; β>0 → penalty
        concentrated on low-probability (tail) tokens.
    lambda_sced : float
        Weight of the L_SCED term in the total loss.
    lambda_kl : float
        Weight of the global KL regularization term.
    eps : float
        Numerical stability epsilon for log computations.
    """
    alpha: float = 2.0
    beta: float = 1.0
    lambda_sced: float = 0.1
    lambda_kl: float = 0.1
    eps: float = 1e-9


# ---------------------------------------------------------------------------
# L_SCED — Sparsity-Calibrated Entropic Divergence
# ---------------------------------------------------------------------------

class SCEDLoss(nn.Module):
    """Sparsity-Calibrated Entropic Divergence  (Eq. 2 in the paper).

    Parameters
    ----------
    alpha, beta, eps : see CAREFConfig.

    Special cases (recovered by parameter choice)
    ----------------------------------------------
    α=1, β=0  →  per-token KL divergence from uniform
    α>1, β=0  →  power-law entropic penalty
    α=1, β>0  →  sparsity-weighted KL
    α>1, β>0  →  full CAREF regime  ← recommended
    """

    def __init__(self, alpha: float = 2.0, beta: float = 1.0, eps: float = 1e-9):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : Tensor, shape (B, T, V)
            Raw decoder logits before softmax.
        attention_mask : Tensor, shape (B, T), optional
            1 for valid positions, 0 for padding. When provided, padded
            positions are excluded from the loss sum.

        Returns
        -------
        Scalar tensor — mean L_SCED value over the batch.
        """
        # P_{t,v}  —  shape (B, T, V)
        P = F.softmax(logits, dim=-1)

        V = logits.size(-1)
        # U_v = 1/|V|  (uniform prior)
        log_ratio = torch.log(P + self.eps) - torch.log(
            torch.tensor(1.0 / V, device=logits.device, dtype=logits.dtype)
        )

        # |P log(P/U)|^α  ·  (1-P)^β
        entropic_term = torch.abs(P * log_ratio) ** self.alpha          # (B, T, V)
        sparsity_weight = (1.0 - P) ** self.beta                        # (B, T, V)
        sced = (entropic_term * sparsity_weight).sum(dim=-1)            # (B, T)

        if attention_mask is not None:
            # mask out padding tokens
            mask = attention_mask.float()
            sced = (sced * mask).sum() / (mask.sum() + self.eps)
        else:
            sced = sced.mean()

        return sced


# ---------------------------------------------------------------------------
# Global KL Regularization Term
# ---------------------------------------------------------------------------

class GlobalKLLoss(nn.Module):
    """Global KL divergence from uniform prior (Eq. 1, L_KL term).

        L_KL = Σ_t Σ_v P_{t,v} log(P_{t,v} / U_v)

    This provides broad calibration pressure across the full decoding
    horizon, complementing the adaptive token-level focus of L_SCED.
    """

    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        P = F.softmax(logits, dim=-1)
        V = logits.size(-1)
        log_uniform = torch.log(torch.tensor(1.0 / V, device=logits.device, dtype=logits.dtype))
        kl = (P * (torch.log(P + self.eps) - log_uniform)).sum(dim=-1)  # (B, T)

        if attention_mask is not None:
            mask = attention_mask.float()
            kl = (kl * mask).sum() / (mask.sum() + self.eps)
        else:
            kl = kl.mean()

        return kl


# ---------------------------------------------------------------------------
# Unified CAREF Loss  (Eq. 1 in the paper)
# ---------------------------------------------------------------------------

class CAREFLoss(nn.Module):
    """Complete CAREF training objective.

        L_CAREF = L_CE + λ_SCED · L_SCED + λ_KL · L_KL

    This module wraps all three terms and returns both the combined
    scalar and a diagnostics dict for logging.

    Parameters
    ----------
    config : CAREFConfig
        Hyperparameter bundle.
    ignore_index : int
        Token ID to ignore in cross-entropy (typically the pad token).
    """

    def __init__(self, config: CAREFConfig, ignore_index: int = -100):
        super().__init__()
        self.config = config
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.sced = SCEDLoss(alpha=config.alpha, beta=config.beta, eps=config.eps)
        self.kl = GlobalKLLoss(eps=config.eps)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        logits : Tensor, shape (B, T, V)
        labels : Tensor, shape (B, T)
            Ground-truth token IDs; positions with ignore_index are skipped.
        decoder_attention_mask : Tensor, shape (B, T), optional

        Returns
        -------
        total_loss : scalar Tensor
        diagnostics : dict[str, float]
            Individual loss components for logging / W&B.
        """
        # --- Cross-entropy (reshape for nn.CrossEntropyLoss) ---
        B, T, V = logits.shape
        loss_ce = self.ce(logits.view(B * T, V), labels.view(B * T))

        # --- SCED and global KL ---
        loss_sced = self.sced(logits, decoder_attention_mask)
        loss_kl = self.kl(logits, decoder_attention_mask)

        total = (
            loss_ce
            + self.config.lambda_sced * loss_sced
            + self.config.lambda_kl * loss_kl
        )

        diagnostics = {
            "loss/ce": loss_ce.item(),
            "loss/sced": loss_sced.item(),
            "loss/kl": loss_kl.item(),
            "loss/total": total.item(),
        }

        return total, diagnostics
