"""Inference module for Axon IA."""

from axon_ia.inference.predictor import Predictor
from axon_ia.inference.sliding_window import SlidingWindowInference
from axon_ia.inference.postprocessing import (
    apply_threshold,
    remove_small_objects,
    fill_holes,
    largest_connected_component,
    apply_postprocessing
)

__all__ = [
    "Predictor",
    "SlidingWindowInference",
    "apply_threshold",
    "remove_small_objects",
    "fill_holes",
    "largest_connected_component",
    "apply_postprocessing"
]