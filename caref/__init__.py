"""
CAREF: Calibration-Aware Regularization for Explanation Faithfulness
Without Rationale Supervision.

EMNLP 2026  |  https://kaopanboonyuen.github.io/CAREF

Quick start
-----------
>>> from caref import CAREFLoss, CAREFConfig, CAREFModel, CAREFModelConfig
>>> config = CAREFConfig(alpha=2.0, beta=1.0, lambda_sced=0.1, lambda_kl=0.1)
>>> loss_fn = CAREFLoss(config)
"""

from .losses import CAREFConfig, CAREFLoss, SCEDLoss, GlobalKLLoss
from .model import CAREFModel, CAREFModelConfig, CAREF_VARIANTS, count_parameters
from .data import (
    NLERecord,
    NLEDataset,
    FEBProtocol,
    make_dataloaders,
    SUPPORTED_DATASETS,
    FEB_N_SPLITS,
    FEB_TRAIN_SIZE,
    FEB_VAL_SIZE,
)
from .metrics import compute_accuracy, compute_nbert, aggregate_splits, format_results_table
from .trainer import CAREFTrainer, TrainingConfig

__version__ = "1.0.0"
__author__ = "Teerapong Panboonyuen"
__paper__ = (
    "CAREF: Calibration-Aware Regularization for Explanation Faithfulness "
    "Without Rationale Supervision. EMNLP 2026."
)
__url__ = "https://kaopanboonyuen.github.io/CAREF"

__all__ = [
    # losses
    "CAREFConfig",
    "CAREFLoss",
    "SCEDLoss",
    "GlobalKLLoss",
    # model
    "CAREFModel",
    "CAREFModelConfig",
    "CAREF_VARIANTS",
    "count_parameters",
    # data
    "NLERecord",
    "NLEDataset",
    "FEBProtocol",
    "make_dataloaders",
    "SUPPORTED_DATASETS",
    "FEB_N_SPLITS",
    "FEB_TRAIN_SIZE",
    "FEB_VAL_SIZE",
    # metrics
    "compute_accuracy",
    "compute_nbert",
    "aggregate_splits",
    "format_results_table",
    # training
    "CAREFTrainer",
    "TrainingConfig",
]
