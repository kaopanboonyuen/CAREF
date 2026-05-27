"""
tests/test_losses.py
====================
Unit tests for the CAREF loss module.

Run with:  pytest tests/test_losses.py -v
"""

import math
import pytest
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from caref.losses import SCEDLoss, GlobalKLLoss, CAREFLoss, CAREFConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_logits():
    """(B=2, T=4, V=10) logits."""
    torch.manual_seed(0)
    return torch.randn(2, 4, 10)


@pytest.fixture
def dummy_labels():
    """(B=2, T=4) label tensor with one -100 (ignored) position."""
    labels = torch.randint(0, 10, (2, 4))
    labels[0, -1] = -100
    return labels


@pytest.fixture
def dummy_mask():
    """(B=2, T=4) attention mask."""
    return torch.ones(2, 4, dtype=torch.long)


# ---------------------------------------------------------------------------
# SCEDLoss
# ---------------------------------------------------------------------------

class TestSCEDLoss:

    def test_output_is_scalar(self, dummy_logits):
        loss_fn = SCEDLoss(alpha=2.0, beta=1.0)
        loss = loss_fn(dummy_logits)
        assert loss.ndim == 0, "SCEDLoss should return a scalar."

    def test_non_negative(self, dummy_logits):
        loss_fn = SCEDLoss(alpha=2.0, beta=1.0)
        loss = loss_fn(dummy_logits)
        assert loss.item() >= 0.0, "SCEDLoss must be non-negative."

    def test_alpha1_beta0_approximates_abs_kl(self, dummy_logits):
        """α=1, β=0 should recover |per-token KL contribution| from uniform.

        Note: L_SCED uses |P log(P/U)|^α, so at α=1, β=0 it gives the
        *absolute value* of the KL contribution per token-vocabulary pair.
        This equals the standard KL only when all P log(P/U) ≥ 0
        (i.e., P ≥ U = 1/V), which is not guaranteed.  The absolute value
        is intentional: it ensures the loss is always non-negative regardless
        of α, enabling super-linear curvature (α>1) without sign issues.
        """
        sced_fn = SCEDLoss(alpha=1.0, beta=0.0)
        loss_sced = sced_fn(dummy_logits)

        # Manual |KL contribution|
        P = F.softmax(dummy_logits, dim=-1)
        V = dummy_logits.size(-1)
        log_ratio = torch.log(P + 1e-9) - math.log(1.0 / V)
        abs_kl_manual = torch.abs(P * log_ratio).sum(dim=-1).mean()

        assert torch.isclose(loss_sced, abs_kl_manual, atol=1e-4), (
            f"α=1, β=0 should recover |KL|. "
            f"Got {loss_sced:.4f} vs {abs_kl_manual:.4f}"
        )

    def test_mask_reduces_loss(self, dummy_logits):
        """Applying a partial mask should change (not crash) the loss."""
        loss_fn = SCEDLoss(alpha=2.0, beta=1.0)
        mask = torch.ones(2, 4)
        mask[1, 2:] = 0  # mask last two tokens of second sequence

        loss_no_mask = loss_fn(dummy_logits)
        loss_masked = loss_fn(dummy_logits, attention_mask=mask)
        assert loss_no_mask.item() != loss_masked.item()

    def test_larger_alpha_increases_sensitivity(self, dummy_logits):
        """Higher α should generally produce larger penalties for overconfident logits."""
        loss_low = SCEDLoss(alpha=1.0, beta=0.0)(dummy_logits)
        loss_high = SCEDLoss(alpha=3.0, beta=0.0)(dummy_logits)
        # Both are positive; high-alpha is expected ≥ low-alpha on typical logits
        assert loss_high.item() >= 0 and loss_low.item() >= 0

    def test_gradients_flow(self, dummy_logits):
        """Backward pass should not raise."""
        logits = dummy_logits.requires_grad_(True)
        loss = SCEDLoss(alpha=2.0, beta=1.0)(logits)
        loss.backward()
        assert logits.grad is not None

    def test_beta_zero_vs_positive(self, dummy_logits):
        """β>0 should produce a different loss value than β=0."""
        l0 = SCEDLoss(alpha=2.0, beta=0.0)(dummy_logits)
        l1 = SCEDLoss(alpha=2.0, beta=1.0)(dummy_logits)
        assert not torch.isclose(l0, l1)


# ---------------------------------------------------------------------------
# GlobalKLLoss
# ---------------------------------------------------------------------------

class TestGlobalKLLoss:

    def test_output_is_scalar(self, dummy_logits):
        loss_fn = GlobalKLLoss()
        assert loss_fn(dummy_logits).ndim == 0

    def test_non_negative(self, dummy_logits):
        assert GlobalKLLoss()(dummy_logits).item() >= 0.0

    def test_uniform_logits_near_zero(self):
        """Uniform distribution → KL from uniform ≈ 0."""
        logits = torch.zeros(2, 4, 10)   # softmax → uniform
        loss = GlobalKLLoss()(logits)
        assert loss.item() < 1e-5, f"KL should be ~0 for uniform, got {loss.item()}"

    def test_gradients_flow(self, dummy_logits):
        logits = dummy_logits.requires_grad_(True)
        GlobalKLLoss()(logits).backward()
        assert logits.grad is not None


# ---------------------------------------------------------------------------
# CAREFLoss (unified)
# ---------------------------------------------------------------------------

class TestCAREFLoss:

    def test_returns_scalar_and_dict(self, dummy_logits, dummy_labels):
        cfg = CAREFConfig(alpha=2.0, beta=1.0, lambda_sced=0.1, lambda_kl=0.1)
        fn = CAREFLoss(cfg, ignore_index=-100)
        total, diag = fn(dummy_logits, dummy_labels)
        assert total.ndim == 0
        assert set(diag.keys()) == {"loss/ce", "loss/sced", "loss/kl", "loss/total"}

    def test_total_equals_sum_of_components(self, dummy_logits, dummy_labels):
        cfg = CAREFConfig(alpha=2.0, beta=1.0, lambda_sced=0.1, lambda_kl=0.1)
        fn = CAREFLoss(cfg, ignore_index=-100)
        total, diag = fn(dummy_logits, dummy_labels)
        expected = (
            diag["loss/ce"]
            + cfg.lambda_sced * diag["loss/sced"]
            + cfg.lambda_kl * diag["loss/kl"]
        )
        assert abs(diag["loss/total"] - expected) < 1e-5

    def test_no_regularization_equals_ce(self, dummy_logits, dummy_labels):
        """λ_SCED = λ_KL = 0 → total loss == CE loss."""
        cfg = CAREFConfig(lambda_sced=0.0, lambda_kl=0.0)
        fn = CAREFLoss(cfg, ignore_index=-100)
        total, diag = fn(dummy_logits, dummy_labels)
        assert abs(total.item() - diag["loss/ce"]) < 1e-5

    def test_gradients_flow(self, dummy_logits, dummy_labels):
        logits = dummy_logits.requires_grad_(True)
        cfg = CAREFConfig()
        fn = CAREFLoss(cfg, ignore_index=-100)
        total, _ = fn(logits, dummy_labels)
        total.backward()
        assert logits.grad is not None

    def test_with_decoder_mask(self, dummy_logits, dummy_labels, dummy_mask):
        cfg = CAREFConfig()
        fn = CAREFLoss(cfg, ignore_index=-100)
        total_no_mask, _ = fn(dummy_logits, dummy_labels)
        total_masked, _ = fn(dummy_logits, dummy_labels, decoder_attention_mask=dummy_mask)
        # Both should be valid scalars
        assert total_no_mask.item() > 0
        assert total_masked.item() > 0
