"""
tests/test_data.py
==================
Tests for the FEB protocol, NLEDataset, and DataLoader factory.

Run with:  pytest tests/test_data.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from caref.data import (
    NLERecord,
    FEBProtocol,
    FEB_TRAIN_SIZE,
    FEB_VAL_SIZE,
)


def make_records(n=1000):
    labels = [0, 1, 2]
    return [
        NLERecord(
            input_text=f"Question {i}",
            answer=f"Answer {i % len(labels)}",
            explanation=f"Because {i}.",
            label_id=i % len(labels),
        )
        for i in range(n)
    ]


class TestFEBProtocol:

    def test_correct_split_sizes(self):
        records = make_records(1000)
        feb = FEBProtocol(records, n_splits=5)
        for _, train, val in feb:
            assert len(train) == FEB_TRAIN_SIZE
            assert len(val) == FEB_VAL_SIZE

    def test_n_splits(self):
        records = make_records(1000)
        feb = FEBProtocol(records, n_splits=7)
        splits = list(feb)
        assert len(splits) == 7

    def test_different_seeds_give_different_splits(self):
        records = make_records(500)
        feb1 = FEBProtocol(records, n_splits=2, seed=0)
        feb2 = FEBProtocol(records, n_splits=2, seed=99)
        splits1 = [(t, v) for _, t, v in feb1]
        splits2 = [(t, v) for _, t, v in feb2]
        # At least one split should differ
        first_train_1 = {r.input_text for r in splits1[0][0]}
        first_train_2 = {r.input_text for r in splits2[0][0]}
        assert first_train_1 != first_train_2

    def test_train_val_disjoint(self):
        records = make_records(1000)
        feb = FEBProtocol(records, n_splits=3)
        for _, train, val in feb:
            train_texts = {r.input_text for r in train}
            val_texts = {r.input_text for r in val}
            assert train_texts.isdisjoint(val_texts), "Train and val must be disjoint."

    def test_class_balance(self):
        """Train split should have ~equal representation across labels."""
        records = make_records(900)   # 300 per class
        feb = FEBProtocol(records, n_splits=5)
        for _, train, _ in feb:
            counts = {}
            for r in train:
                counts[r.label_id] = counts.get(r.label_id, 0) + 1
            # Allow ±1 due to remainder handling
            label_counts = list(counts.values())
            assert max(label_counts) - min(label_counts) <= 1
