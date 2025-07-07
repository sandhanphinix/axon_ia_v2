"""
Report generation for medical image segmentation evaluation.

This module provides functions for creating comprehensive evaluation reports,
including visualizations, metrics tables, and patient-specific reports.
"""

import os
import time
from pathlib import Path
import json
from typing import Dict, List, Optional, Union, Tuple, Any
import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Template

from axon_ia.evaluation.visualization import (
    plot_segmentation_overlay,
    plot_metrics_per_patient,
    plot_volume_correlation,
    plot_multiple_slices
)
from axon_ia.utils.logger import get_logger

logger = get_logger()


def generate_patient_report(
    patient_id: str,
    image: np.ndarray,
    target: Optional[np.ndarray] = None,
    prediction: Optional[np.ndarray] = None,
    metrics: Optional[Dict[str, float]] = None,
    output_dir: Union[str, Path] = "reports",
    title: Optional[str] = None,
    include_3d_views: bool = True,
    spacing: Optional[Tuple[float, ...]] = None
) -> Path:
    """
    Generate a detailed report for a single patient.
    
    Args:
        patient_id: Patient identifier
        image: Patient image volume
        target: Ground truth segmentation
        prediction: Predicted segmentation
        metrics: Dictionary of evaluation metrics
        output_dir: Directory to save the report
        title: Report title
        include_3d_views: Whether to include 3D visualizations
        spacing: Voxel spacing in mm
        
    Returns:
        Path to generated report
    """
    output_dir = Path(output_dir)
    patient_dir = output_dir / f"patient_{patient_id}"
    patient_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    visualizations = {}
    
    # Generate axial slices
    axial_fig = plot_multiple_slices(
        image=image,
        mask=target,
        prediction=prediction,
        axis=2,
        num_slices=3,
        figsize=(15, 5),
        title=f"Axial Slices - Patient {patient_id}",
        save_path=patient_dir / f"{patient_id}_axial.png"
    )
    visualizations["axial"] = f"{patient_id}_axial.png"
    plt.close(axial_fig)
    
    # Generate coronal slices
    coronal_fig = plot_multiple_slices(
        image=image,
        mask=target,
        prediction=prediction,
        axis=1,
        num_slices=3,
        figsize=(15, 5),
        title=f"Coronal Slices - Patient {patient_id}",
        save_path=patient_dir / f"{patient_id}_coronal.png"
    )
    visualizations["coronal"] = f"{patient_id}_coronal.png"
    plt.close(coronal_fig)
    
    # Generate sagittal slices
    sagittal_fig = plot_multiple_slices(
        image=image,
        mask=target,
        prediction=prediction,
        axis=0,
        num_slices=3,
        figsize=(15, 5),
        title=f"Sagittal Slices - Patient {patient_id}",
        save_path=patient_dir / f"{patient_id}_sagittal.png"
    )
    visualizations["sagittal"] = f"{patient_id}_sagittal.png"
    plt.close(sagittal_fig)
    
    # 3D visualization
    if include_3d_views and target is not None and prediction is not None:
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            # Generate 3D visualization
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get surface points
            target_points = np.argwhere(target > 0)
            pred_points = np.argwhere(prediction > 0)
            
            # Sample points to keep visualization manageable
            if len(target_points) > 1000:
                target_idx = np.random.choice(len(target_points), 1000, replace=False)
                target_points = target_points[target_idx]
            if len(pred_points) > 1000:
                pred_idx = np.random.choice(len(pred_points), 1000, replace=False)
                pred_points = pred_points[pred_idx]
            
            # Plot points
            ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], 
                      color='red', alpha=0.3, label='Ground Truth')
            ax.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                      color='blue', alpha=0.3, label='Prediction')
            
            ax.set_title(f"3D Visualization - Patient {patient_id}")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            
            plt.savefig(patient_dir / f"{patient_id}_3d.png")
            visualizations["3d"] = f"{patient_id}_3d.png"
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to generate 3D visualization: {e}")
    
    # Generate metrics table
    metrics_table = ""
    if metrics is not None:
        metrics_df = pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})
        metrics_table = metrics_df.to_html(index=False)
    
    # Generate report HTML
    report_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Patient {{ patient_id }} - Segmentation Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                max-width: 1200px;
                margin: auto;
            }
            h1, h2 {
                color: #333;
            }
            .container {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            .image-section {
                margin-top: 20px;
            }
            .image-container {
                max-width: 100%;
            }
            img {
                max-width: 100%;
                height: auto;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-top: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            .footer {
                margin-top: 30px;
                font-size: 12px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <h1>{{ title }}</h1>
        <p>Patient ID: {{ patient_id }}</p>
        <p>Generated: {{ timestamp }}</p>
        
        <h2>Metrics</h2>
        {{ metrics_table }}
        
        <div class="container">
            <div class="image-section">
                <h2>Axial View</h2>
                <div class="image-container">
                    <img src="{{ visualizations.axial }}" alt="Axial Slices">
                </div>
            </div>
            
            <div class="image-section">
                <h2>Coronal View</h2>
                <div class="image-container">
                    <img src="{{ visualizations.coronal }}" alt="Coronal Slices">
                </div>
            </div>
            
            <div class="image-section">
                <h2>Sagittal View</h2>
                <div class="image-container">
                    <img src="{{ visualizations.sagittal }}" alt="Sagittal Slices">
                </div>
            </div>
            
            {% if visualizations.get('3d') %}
            <div class="image-section">
                <h2>3D Visualization</h2>
                <div class="image-container">
                    <img src="{{ visualizations['3d'] }}" alt="3D Visualization">
                </div>
            </div>
            {% endif %}
        </div>
        
        <div class="footer">
            <p>Generated by Axon IA v2.0</p>
        </div>
    </body>
    </html>
    """
    
    # Create Jinja2 template
    template = Template(report_template)
    
    # Set up template variables
    if title is None:
        title = f"Segmentation Report - Patient {patient_id}"
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Render template
    html_content = template.render(
        title=title,
        patient_id=patient_id,
        timestamp=timestamp,
        metrics_table=metrics_table,
        visualizations=visualizations
    )
    
    # Write HTML file
    report_path = patient_dir / f"{patient_id}_report.html"
    with open(report_path, "w") as f:
        f.write(html_content)
    
    logger.info(f"Generated patient report at {report_path}")
    
    return report_path


def generate_evaluation_report(
    patient_metrics: Dict[str, Dict[str, float]],
    output_dir: Union[str, Path] = "reports",
    patient_images: Optional[Dict[str, np.ndarray]] = None,
    patient_targets: Optional[Dict[str, np.ndarray]] = None,
    patient_predictions: Optional[Dict[str, np.ndarray]] = None,
    title: str = "Segmentation Evaluation Report",
    model_name: Optional[str] = None,
    metrics_to_include: List[str] = ["dice", "iou", "hausdorff", "precision", "recall"]
) -> Path:
    """
    Generate a comprehensive evaluation report for multiple patients.
    
    Args:
        patient_metrics: Dictionary of metrics for each patient
        output_dir: Directory to save the report
        patient_images: Dictionary of images for each patient
        patient_targets: Dictionary of ground truth segmentations
        patient_predictions: Dictionary of predicted segmentations
        title: Report title
        model_name: Name of the evaluated model
        metrics_to_include: List of metrics to include in report
        
    Returns:
        Path to generated report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figures directory
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    visualizations = {}
    
    # For each metric, generate a per-patient bar chart
    for metric in metrics_to_include:
        if all(metric in metrics for metrics in patient_metrics.values()):
            fig = plot_metrics_per_patient(
                patient_metrics=patient_metrics,
                metric=metric,
                sort_by=metric,
                figsize=(10, 6),
                title=f"{metric.capitalize()} per Patient",
                save_path=figures_dir / f"{metric}_per_patient.png"
            )
            visualizations[f"{metric}_per_patient"] = f"figures/{metric}_per_patient.png"
            plt.close(fig)
    
    # Generate volume correlation plot if volume data is available
    if all("volume" in metrics and "pred_volume" in metrics for metrics in patient_metrics.values()):
        true_volumes = {pid: metrics["volume"] for pid, metrics in patient_metrics.items()}
        pred_volumes = {pid: metrics["pred_volume"] for pid, metrics in patient_metrics.items()}
        
        fig = plot_volume_correlation(
            true_volumes=true_volumes,
            pred_volumes=pred_volumes,
            figsize=(8, 8),
            title="Volume Correlation",
            save_path=figures_dir / "volume_correlation.png"
        )
        visualizations["volume_correlation"] = "figures/volume_correlation.png"
        plt.close(fig)
    
    # Generate example patient visualizations
    if patient_images is not None and patient_targets is not None and patient_predictions is not None:
        # Select up to 3 patients for example visualization
        example_patients = list(patient_metrics.keys())
        if len(example_patients) > 3:
            # Select patients with low, medium, and high Dice scores if available
            if "dice" in metrics_to_include:
                dice_scores = [(pid, metrics.get("dice", 0)) for pid, metrics in patient_metrics.items()]
                dice_scores.sort(key=lambda x: x[1])
                
                # Get patients from different score ranges
                if len(dice_scores) >= 3:
                    low_idx = 0
                    med_idx = len(dice_scores) // 2
                    high_idx = len(dice_scores) - 1
                    example_patients = [dice_scores[low_idx][0], dice_scores[med_idx][0], dice_scores[high_idx][0]]
        
        # Generate visualizations for example patients
        for i, patient_id in enumerate(example_patients[:3]):
            if (patient_id in patient_images and patient_id in patient_targets and 
                patient_id in patient_predictions):
                
                fig = plot_multiple_slices(
                    image=patient_images[patient_id],
                    mask=patient_targets[patient_id],
                    prediction=patient_predictions[patient_id],
                    axis=2,  # Axial view
                    num_slices=3,
                    figsize=(15, 5),
                    title=f"Patient {patient_id} - Axial View",
                    save_path=figures_dir / f"example_{i+1}_patient_{patient_id}.png"
                )
                visualizations[f"example_{i+1}"] = f"figures/example_{i+1}_patient_{patient_id}.png"
                plt.close(fig)
                
                # Generate individual patient reports
                generate_patient_report(
                    patient_id=patient_id,
                    image=patient_images[patient_id],
                    target=patient_targets[patient_id],
                    prediction=patient_predictions[patient_id],
                    metrics=patient_metrics[patient_id],
                    output_dir=output_dir / "patients",
                    title=f"Patient {patient_id} - Detailed Report"
                )
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame(patient_metrics).T
    
    # Calculate summary statistics
    summary_stats = {}
    for metric in metrics_to_include:
        if metric in metrics_df.columns:
            summary_stats[metric] = {
                "mean": metrics_df[metric].mean(),
                "std": metrics_df[metric].std(),
                "median": metrics_df[metric].median(),
                "min": metrics_df[metric].min(),
                "max": metrics_df[metric].max()
            }
    
    # Convert to HTML
    metrics_html = metrics_df.to_html(float_format="%.4f")
    
    # Create summary stats HTML
    summary_html = ""
    for metric, stats in summary_stats.items():
        summary_html += f"<h3>{metric.capitalize()}</h3>\n"
        summary_html += "<table>\n"
        summary_html += "  <tr>\n"
        for stat in ["mean", "std", "median", "min", "max"]:
            summary_html += f"    <th>{stat.capitalize()}</th>\n"
        summary_html += "  </tr>\n"
        
        summary_html += "  <tr>\n"
        for stat, value in stats.items():
            summary_html += f"    <td>{value:.4f}</td>\n"
        summary_html += "  </tr>\n"
        summary_html += "</table>\n"
    
    # Generate report HTML
    report_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{{ title }}</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                max-width: 1200px;
                margin: auto;
            }
            h1, h2, h3 {
                color: #333;
            }
            .container {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            .image-section {
                margin-top: 20px;
            }
            .image-container {
                max-width: 100%;
            }
            img {
                max-width: 100%;
                height: auto;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-top: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            .metrics-table {
                overflow-x: auto;
            }
            .footer {
                margin-top: 30px;
                font-size: 12px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <h1>{{ title }}</h1>
        <p>Generated: {{ timestamp }}</p>
        {% if model_name %}
        <p>Model: {{ model_name }}</p>
        {% endif %}
        <p>Number of patients: {{ num_patients }}</p>
        
        <h2>Summary Statistics</h2>
        <div class="metrics-table">
            {{ summary_html|safe }}
        </div>
        
        <h2>Visualizations</h2>
        <div class="container">
            {% for name, path in visualizations.items() if 'per_patient' in name %}
            <div class="image-section">
                <h3>{{ name.split('_per_patient')[0].capitalize() }} per Patient</h3>
                <div class="image-container">
                    <img src="{{ path }}" alt="{{ name }}">
                </div>
            </div>
            {% endfor %}
            
            {% if visualizations.get('volume_correlation') %}
            <div class="image-section">
                <h3>Volume Correlation</h3>
                <div class="image-container">
                    <img src="{{ visualizations['volume_correlation'] }}" alt="Volume Correlation">
                </div>
            </div>
            {% endif %}
            
            {% for i in range(1, 4) %}
                {% if visualizations.get('example_' + i|string) %}
                <div class="image-section">
                    <h3>Example Patient {{ i }}</h3>
                    <div class="image-container">
                        <img src="{{ visualizations['example_' + i|string] }}" alt="Example Patient {{ i }}">
                    </div>
                </div>
                {% endif %}
            {% endfor %}
        </div>
        
        <h2>Detailed Metrics</h2>
        <div class="metrics-table">
            {{ metrics_html|safe }}
        </div>
        
        <div class="footer">
            <p>Generated by Axon IA v2.0</p>
            <p>Report generated on {{ timestamp }}</p>
        </div>
    </body>
    </html>
    """
    
    # Create Jinja2 template
    template = Template(report_template)
    
    # Set up template variables
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Render template
    html_content = template.render(
        title=title,
        timestamp=timestamp,
        model_name=model_name,
        num_patients=len(patient_metrics),
        metrics_html=metrics_html,
        summary_html=summary_html,
        visualizations=visualizations
    )
    
    # Write HTML file
    report_path = output_dir / "evaluation_report.html"
    with open(report_path, "w") as f:
        f.write(html_content)
    
    # Save metrics as CSV
    metrics_df.to_csv(output_dir / "metrics.csv")
    
    # Save summary as JSON
    with open(output_dir / "summary_stats.json", "w") as f:
        json_data = {metric: {k: float(v) for k, v in stats.items()} 
                    for metric, stats in summary_stats.items()}
        json.dump(json_data, f, indent=4)
    
    logger.info(f"Generated evaluation report at {report_path}")
    
    return report_path