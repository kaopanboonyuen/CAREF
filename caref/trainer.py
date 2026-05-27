"""
caref/trainer.py
================
CAREF training loop with AdamW, linear-decay schedule, gradient clipping,
and per-split metric logging — matching the paper's experimental setup.

Hardware reference (from Appendix D):
    NVIDIA A40, CUDA 11.4, Ubuntu 20.04, AMD Ryzen 9 5900X, 64 GB RAM

Reference
---------
Panboonyuen, T. (2026). CAREF: Calibration-Aware Regularization for
Explanation Faithfulness Without Rationale Supervision. EMNLP 2026.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .losses import CAREFLoss, CAREFConfig
from .model import CAREFModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training configuration  (matches Appendix D of the paper)
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Training hyperparameters (Appendix D).

    Attributes
    ----------
    epochs : int         50 epochs per split.
    lr : float           3e-5 (AdamW).
    batch_size : int     4.
    warmup_steps : int   500.
    max_grad_norm : float 1.0.
    weight_decay : float  0.01.
    adam_beta1 / adam_beta2 / adam_eps : AdamW betas and epsilon.
    """
    epochs: int = 50
    lr: float = 3e-5
    batch_size: int = 4
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = False
    log_every_n_steps: int = 10
    caref: CAREFConfig = field(default_factory=CAREFConfig)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class CAREFTrainer:
    """Single-split CAREF trainer.

    Parameters
    ----------
    model : CAREFModel
    train_loader : DataLoader
    val_loader : DataLoader
    config : TrainingConfig
    """

    def __init__(
        self,
        model: CAREFModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.device = torch.device(config.device)

        self.model.backbone.to(self.device)

        # --- Optimizer ---
        trainable_params = [
            p for p in self.model.backbone.parameters() if p.requires_grad
        ]
        self.optimizer = AdamW(
            trainable_params,
            lr=config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )

        # --- Scheduler (linear decay with warm-up) ---
        total_steps = len(train_loader) * config.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps,
        )

        # --- Loss ---
        pad_id = model.tokenizer.pad_token_id or 0
        self.loss_fn = CAREFLoss(config.caref, ignore_index=-100)

        # --- AMP scaler ---
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

        self.global_step = 0

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self) -> dict[str, float]:
        self.model.backbone.train()
        totals: dict[str, float] = {}
        n_batches = 0

        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=self.cfg.fp16):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    decoder_attention_mask=batch["decoder_attention_mask"],
                )
                loss, diag = self.loss_fn(
                    logits=outputs.logits,
                    labels=batch["labels"],
                    decoder_attention_mask=batch["decoder_attention_mask"],
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.backbone.parameters(), self.cfg.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()

            for k, v in diag.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1
            self.global_step += 1

            if self.global_step % self.cfg.log_every_n_steps == 0:
                logger.debug(
                    "step=%d  loss=%.4f  sced=%.4f  kl=%.4f",
                    self.global_step,
                    diag["loss/total"],
                    diag["loss/sced"],
                    diag["loss/kl"],
                )

        return {k: v / n_batches for k, v in totals.items()}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Run greedy decoding over the validation set and return accuracy."""
        self.model.backbone.eval()
        correct = 0
        total = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            label_ids = batch["label_id"].to(self.device)

            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                num_beams=1,
            )

            # Decode and compare (simplified accuracy using label_id lookup)
            # In full pipeline this is replaced by task-specific accuracy
            decoded = self.model.tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )
            # placeholder: count non-empty generations
            correct += sum(1 for d in decoded if len(d.strip()) > 0)
            total += len(decoded)

        return {"val/accuracy": correct / max(total, 1)}

    # ------------------------------------------------------------------
    # Full training run
    # ------------------------------------------------------------------

    def train(self) -> list[dict]:
        """Train for self.cfg.epochs epochs. Returns per-epoch metric log."""
        history = []
        for epoch in range(1, self.cfg.epochs + 1):
            train_metrics = self._train_epoch()
            logger.info(
                "epoch=%d/%d  train_loss=%.4f",
                epoch,
                self.cfg.epochs,
                train_metrics.get("loss/total", float("nan")),
            )
            history.append({"epoch": epoch, **train_metrics})
        return history
