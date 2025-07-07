"""Evaluation module for Axon IA."""

from axon_ia.evaluation.metrics import (
    compute_metrics,
    dice_score,
    iou_score,
    hausdorff_distance,
    precision_score,
    recall_score,
    specificity_score,
    f1_score,
    surface_dice
)
from axon_ia.evaluation.visualization import (
    plot_segmentation_overlay,
    plot_metrics_per_patient,
    plot_volume_correlation,
    plot_multiple_slices
)
from axon_ia.evaluation.report_generator import (
    generate_evaluation_report,
    generate_patient_report
)

__all__ = [
    "compute_metrics",
    "dice_score",
    "iou_score",
    "hausdorff_distance",
    "precision_score",
    "recall_score",
    "specificity_score",
    "f1_score",
    "surface_dice",
    "plot_segmentation_overlay",
    "plot_metrics_per_patient",
    "plot_volume_correlation",
    "plot_multiple_slices",
    "generate_evaluation_report",
    "generate_patient_report"
]