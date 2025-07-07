"""Training module for Axon IA."""

from axon_ia.training.trainer import Trainer
from axon_ia.training.lr_schedulers import create_scheduler

__all__ = [
    "Trainer",
    "create_scheduler"
]