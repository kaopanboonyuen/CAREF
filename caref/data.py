"""
caref/data.py
=============
Dataset loading following the FEB protocol (Marasović et al., 2022):

    • 60 random splits of 48 train / 350 validation examples each
    • Class-balanced sampling within each split
    • Supported datasets: COS-E, ECQA, ComVE, e-SNLI

Reference
---------
Marasović et al. (2022). Few-Shot Self-Rationalization with Natural
Language Prompts. ACL Findings.

Panboonyuen, T. (2026). CAREF: Calibration-Aware Regularization for
Explanation Faithfulness Without Rationale Supervision. EMNLP 2026 Submission.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_DATASETS = ("cos_e", "ecqa", "comve", "esnli")

FEB_TRAIN_SIZE = 48
FEB_VAL_SIZE = 350
FEB_N_SPLITS = 60


# ---------------------------------------------------------------------------
# NLE record
# ---------------------------------------------------------------------------

@dataclass
class NLERecord:
    """One input–answer–explanation triplet."""
    input_text: str      # question / premise
    answer: str          # gold label
    explanation: str     # natural language explanation
    label_id: int        # integer class index


# ---------------------------------------------------------------------------
# Prompt templates (task-specific)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES = {
    "cos_e": (
        "explain commonsense reasoning: {input_text} "
        "answer: {answer} explanation:"
    ),
    "ecqa": (
        "explain answer: {input_text} "
        "answer: {answer} explanation:"
    ),
    "comve": (
        "explain why this statement is against commonsense: {input_text} "
        "answer: {answer} explanation:"
    ),
    "esnli": (
        "explain natural language inference: premise: {input_text} "
        "answer: {answer} explanation:"
    ),
}


# ---------------------------------------------------------------------------
# Torch Dataset
# ---------------------------------------------------------------------------

class NLEDataset(Dataset):
    """PyTorch dataset for one NLE split."""

    def __init__(
        self,
        records: list[NLERecord],
        tokenizer: PreTrainedTokenizer,
        dataset_name: str,
        max_input_length: int = 256,
        max_target_length: int = 64,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name.lower()
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        if self.dataset_name not in PROMPT_TEMPLATES:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Supported: {list(PROMPT_TEMPLATES)}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        template = PROMPT_TEMPLATES[self.dataset_name]
        source = template.format(
            input_text=rec.input_text, answer=rec.answer
        )
        target = rec.explanation

        model_inputs = self.tokenizer(
            source,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer():
            target_enc = self.tokenizer(
                target,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        labels = target_enc["input_ids"].squeeze()
        # Replace padding token id with -100 so CE ignores it
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "decoder_attention_mask": target_enc["attention_mask"].squeeze(),
            "labels": labels,
            "label_id": torch.tensor(rec.label_id, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# FEB Split Generator
# ---------------------------------------------------------------------------

class FEBProtocol:
    """Generates 60 class-balanced train/val splits following the FEB protocol.

    Parameters
    ----------
    records : list[NLERecord]
        Full dataset.
    n_splits : int
        Number of random splits (default 60).
    train_size : int
        Training examples per split (default 48).
    val_size : int
        Validation examples per split (default 350).
    seed : int
        Base random seed.
    """

    def __init__(
        self,
        records: list[NLERecord],
        n_splits: int = FEB_N_SPLITS,
        train_size: int = FEB_TRAIN_SIZE,
        val_size: int = FEB_VAL_SIZE,
        seed: int = 42,
    ):
        self.records = records
        self.n_splits = n_splits
        self.train_size = train_size
        self.val_size = val_size
        self.seed = seed

        # Group by label for balanced sampling
        self._by_label: dict[int, list[NLERecord]] = defaultdict(list)
        for rec in records:
            self._by_label[rec.label_id].append(rec)

    def _balanced_sample(
        self, rng: random.Random, size: int, exclude: set[int]
    ) -> list[NLERecord]:
        """Sample *size* records balanced across labels, excluding indices."""
        labels = sorted(self._by_label.keys())
        n_classes = len(labels)
        per_class = size // n_classes
        remainder = size % n_classes

        sampled: list[NLERecord] = []
        for i, label in enumerate(labels):
            pool = [
                r for j, r in enumerate(self._by_label[label])
                if id(r) not in exclude
            ]
            k = per_class + (1 if i < remainder else 0)
            chosen = rng.sample(pool, min(k, len(pool)))
            sampled.extend(chosen)
            exclude.update(id(r) for r in chosen)

        rng.shuffle(sampled)
        return sampled

    def __iter__(self):
        for split_idx in range(self.n_splits):
            rng = random.Random(self.seed + split_idx)
            used: set[int] = set()

            train_records = self._balanced_sample(rng, self.train_size, used)
            val_records = self._balanced_sample(rng, self.val_size, used)

            yield split_idx, train_records, val_records


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    train_records: list[NLERecord],
    val_records: list[NLERecord],
    tokenizer: PreTrainedTokenizer,
    dataset_name: str,
    max_input_length: int = 256,
    max_target_length: int = 64,
    train_batch_size: int = 4,
    val_batch_size: int = 8,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders for one FEB split."""

    kwargs = dict(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
    )

    train_ds = NLEDataset(train_records, **kwargs)
    val_ds = NLEDataset(val_records, **kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
