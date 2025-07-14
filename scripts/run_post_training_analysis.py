#!/usr/bin/env python
"""
Complete post-training workflow script.

This script runs the complete analysis pipeline after model training,
including evaluation, visualization, and recommendations generation.
Now uses YAML configuration instead of CLI arguments.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import json
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add parent directory to path to make imports work
sys.path.append(str(Path(__file__).resolve().parent.parent))

from axon_ia.config import ConfigParser

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run complete post-training analysis")
    
    parser.add_argument("--config", type=str, 
                        default="configs/analysis/post_training_analysis_config.yaml",
                        help="Path to analysis config YAML file")
    
    parser.add_argument("--override", type=str, nargs="*",
                        help="Override config values (e.g., training.checkpoint_path=/path/to/model.pth)")
    
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be executed without running")
    
    return parser.parse_args()


def load_config(config_path: str, overrides: List[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to YAML config file
        overrides: List of override strings in format "key.subkey=value"
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    if overrides:
        for override in overrides:
            if '=' not in override:
                continue
            key_path, value = override.split('=', 1)
            keys = key_path.split('.')
            
            # Navigate to the correct nested dict
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the value (try to convert to appropriate type)
            try:
                # Try int first
                if value.isdigit():
                    current[keys[-1]] = int(value)
                # Try float
                elif '.' in value and value.replace('.', '').isdigit():
                    current[keys[-1]] = float(value)
                # Try boolean
                elif value.lower() in ['true', 'false']:
                    current[keys[-1]] = value.lower() == 'true'
                # Keep as string
                else:
                    current[keys[-1]] = value
            except:
                current[keys[-1]] = value
    
    return config


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration and return list of errors.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Check required paths
    training_config = config.get('training', {})
    if not training_config.get('config_path'):
        errors.append("training.config_path is required")
    elif not Path(training_config['config_path']).exists():
        errors.append(f"Training config not found: {training_config['config_path']}")
    
    if not training_config.get('checkpoint_path'):
        errors.append("training.checkpoint_path is required")
    elif not Path(training_config['checkpoint_path']).exists():
        errors.append(f"Checkpoint not found: {training_config['checkpoint_path']}")
    
    # Check data directory
    data_config = config.get('data', {})
    if data_config.get('data_dir') and not Path(data_config['data_dir']).exists():
        errors.append(f"Data directory not found: {data_config['data_dir']}")
    
    # Check splits
    valid_splits = ['train', 'val', 'test']
    splits = data_config.get('splits', ['val'])
    for split in splits:
        if split not in valid_splits:
            errors.append(f"Invalid split: {split}. Must be one of {valid_splits}")
    
    return errors


def run_command(cmd: List[str], description: str, dry_run: bool = False) -> bool:
    """
    Run a command and return success status.
    
    Args:
        cmd: Command to run as list
        description: Description for logging
        dry_run: If True, only show what would be executed
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    if dry_run:
        print("🔍 DRY RUN - Command would be executed")
        return True
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("Error:", e)
        if e.stdout:
            print("STDOUT:", e.stdout[-500:])
        if e.stderr:
            print("STDERR:", e.stderr[-500:])
        return False

def get_python_executable() -> str:
    """
    Get the correct Python executable to use.
    Prioritizes current Python executable to maintain environment.
    
    Returns:
        Path to Python executable
    """
    import sys
    
    # Use the current Python executable (maintains virtual environment)
    python_exe = sys.executable
    
    # Verify it has torch installed
    try:
        subprocess.run([python_exe, "-c", "import torch"], 
                      check=True, capture_output=True)
        return python_exe
    except subprocess.CalledProcessError:
        # Current Python doesn't have torch, try alternatives
        alternatives = ["python", "python3", "py"]
        
        for alt in alternatives:
            try:
                subprocess.run([alt, "-c", "import torch"], 
                              check=True, capture_output=True)
                print(f"⚠️  Using {alt} (current Python lacks torch)")
                return alt
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        # No Python with torch found
        raise RuntimeError(
            "No Python executable with torch found. "
            "Please activate your virtual environment or install torch."
        )


def analyze_metrics(metrics_file: Path, config: Dict[str, Any]) -> Dict:
    """
    Analyze metrics and generate recommendations.
    
    Args:
        metrics_file: Path to metrics JSON file
        config: Analysis configuration
        
    Returns:
        Analysis results dictionary
    """
    if not metrics_file.exists():
        return {"error": "Metrics file not found"}
    
    with open(metrics_file, 'r') as f:
        metrics_data = json.load(f)
    
    overall = metrics_data.get('overall', {})
    per_patient = metrics_data.get('per_patient', {})
    
    # Get thresholds from config
    thresholds = config.get('steps', {}).get('analysis', {}).get('performance_thresholds', {})
    excellent_dice = thresholds.get('excellent_dice', 0.8)
    good_dice = thresholds.get('good_dice', 0.7)
    fair_dice = thresholds.get('fair_dice', 0.5)
    min_precision = thresholds.get('min_precision', 0.7)
    min_recall = thresholds.get('min_recall', 0.7)
    
    analysis = {
        "overall_metrics": overall,
        "num_patients": len(per_patient),
        "recommendations": [],
        "thresholds_used": thresholds
    }
    
    # Generate recommendations based on performance
    if 'dice' in overall:
        dice_score = overall['dice']
        analysis["dice_score"] = dice_score
        
        if dice_score < fair_dice:
            analysis["recommendations"].extend([
                "🔴 CRITICAL: Very low Dice score. Consider major changes:",
                "   - Review data quality and preprocessing pipeline",
                "   - Increase model capacity (feature_size: 128+)",
                "   - Try different architecture (UNETR, SegResNet)",
                "   - Extend training duration significantly (100+ epochs)",
                "   - Check for data leakage or preprocessing bugs"
            ])
        elif dice_score < good_dice:
            analysis["recommendations"].extend([
                "🟡 MODERATE: Dice score needs improvement:",
                "   - Enhance data augmentation strategy",
                "   - Adjust loss function weights",
                "   - Implement ensemble methods",
                "   - Add boundary loss component",
                "   - Increase training epochs to 100+"
            ])
        elif dice_score < excellent_dice:
            analysis["recommendations"].extend([
                "🟢 GOOD: Solid performance with room for optimization:",
                "   - Fine-tune hyperparameters",
                "   - Implement test-time augmentation",
                "   - Try ensemble of multiple models",
                "   - Optimize for specific failure cases"
            ])
        else:
            analysis["recommendations"].extend([
                "🌟 EXCELLENT: Outstanding performance!",
                "   - Focus on deployment optimization",
                "   - Implement uncertainty quantification",
                "   - Consider clinical validation studies",
                "   - Optimize inference speed"
            ])
    
    # Precision/Recall analysis
    if 'precision' in overall and 'recall' in overall:
        precision = overall['precision']
        recall = overall['recall']
        
        if precision < min_precision:
            analysis["recommendations"].append(
                "🔸 LOW PRECISION: Many false positives detected"
            )
            analysis["recommendations"].append(
                "   - Increase focal loss weight or add FP penalty"
            )
        
        if recall < min_recall:
            analysis["recommendations"].append(
                "🔸 LOW RECALL: Missing lesions detected"
            )
            analysis["recommendations"].append(
                "   - Increase Dice loss weight or enhance augmentation"
            )
    
    return analysis


def generate_report(output_dir: Path, analysis: Dict, config: Dict[str, Any]):
    """
    Generate comprehensive analysis report.
    
    Args:
        output_dir: Output directory
        analysis: Analysis results
        config: Full configuration used
    """
    reporting_config = config.get('steps', {}).get('reporting', {})
    formats = reporting_config.get('formats', ['markdown'])
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Markdown report
    if 'markdown' in formats:
        report_path = output_dir / "analysis_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Post-Training Analysis Report\n\n")
            f.write(f"**Generated on:** {timestamp}\n")
            f.write(f"**Training Config:** {config['training']['config_path']}\n")
            f.write(f"**Checkpoint:** {config['training']['checkpoint_path']}\n")
            f.write(f"**Analysis Config:** Generated from YAML configuration\n\n")
            
            # Configuration summary
            f.write("## Configuration Summary\n\n")
            f.write(f"- **Splits Evaluated:** {', '.join(config['data']['splits'])}\n")
            f.write(f"- **Visualization Samples:** {config['steps']['visualization']['num_samples']}\n")
            f.write(f"- **Performance Thresholds:**\n")
            thresholds = analysis.get('thresholds_used', {})
            for key, value in thresholds.items():
                f.write(f"  - {key}: {value}\n")
            f.write("\n")
            
            # Overall metrics
            f.write("## Overall Performance\n\n")
            if "overall_metrics" in analysis:
                for metric, value in analysis["overall_metrics"].items():
                    f.write(f"- **{metric.upper()}:** {value:.4f}\n")
            f.write(f"\n**Evaluated on {analysis.get('num_patients', 'N/A')} patients**\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            for rec in analysis.get("recommendations", []):
                f.write(f"{rec}\n")
            f.write("\n")
            
            # Next steps
            f.write("## Next Steps\n\n")
            f.write("1. **Review Visualizations:** Check generated visualizations for failure patterns\n")
            f.write("2. **Implement Improvements:** Apply recommended changes to configuration\n")
            f.write("3. **Retrain Model:** Use enhanced configuration for next iteration\n")
            f.write("4. **Clinical Validation:** If performance is good, consider clinical studies\n\n")
            
            # Files generated
            f.write("## Generated Files\n\n")
            f.write("- `evaluation_*/`: Detailed evaluation metrics and predictions\n")
            f.write("- `visualizations_*/`: Sample visualizations and error analysis\n")
            f.write("- `analysis_report.md`: This summary report\n")
            f.write("- `recommendations.json`: Machine-readable recommendations\n")
            f.write("- `full_config.yaml`: Complete configuration used for this analysis\n\n")
            
            f.write("## Configuration for Next Training\n\n")
            f.write("Consider using the enhanced configuration:\n")
            f.write("```\nconfigs/training/swinunetr_enhanced_config.yaml\n```\n")
            f.write("This includes improvements based on current analysis and best practices.\n")
        
        print(f"📊 Markdown report saved to: {report_path}")
    
    # JSON report
    if 'json' in formats:
        json_report_path = output_dir / "analysis_report.json"
        json_report = {
            "timestamp": timestamp,
            "config": config,
            "analysis": analysis,
            "summary": {
                "dice_score": analysis.get("dice_score"),
                "num_patients": analysis.get("num_patients"),
                "num_recommendations": len(analysis.get("recommendations", []))
            }
        }
        
        with open(json_report_path, 'w') as f:
            json.dump(json_report, f, indent=4)
        
        print(f"📊 JSON report saved to: {json_report_path}")
    
    # Save full config for reproducibility
    config_path = output_dir / "full_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"⚙️  Full configuration saved to: {config_path}")


def main():
    """Main workflow function."""
    args = parse_args()
    
    # Load and validate configuration
    try:
        config = load_config(args.config, args.override)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return 1
    
    # Validate configuration
    errors = validate_config(config)
    if errors:
        print("❌ Configuration validation failed:")
        for error in errors:
            print(f"   - {error}")
        return 1
    
    # Get correct Python executable
    try:
        python_exe = get_python_executable()
        print(f"🐍 Using Python executable: {python_exe}")
    except RuntimeError as e:
        print(f"❌ {e}")
        return 1
    
    # Setup paths from config
    training_config = config['training']
    data_config = config['data']
    output_config = config['output']
    steps_config = config['steps']
    
    # Create output directory
    base_output_dir = Path(output_config['base_dir'])
    if output_config.get('create_timestamp_folder', True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_output_dir / f"analysis_{timestamp}"
    else:
        output_dir = base_output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration summary
    print(f"🚀 Starting post-training analysis")
    print(f"📁 Output directory: {output_dir}")
    print(f"⚙️  Training config: {training_config['config_path']}")
    print(f"🎯 Checkpoint: {training_config['checkpoint_path']}")
    print(f"📊 Splits: {', '.join(data_config['splits'])}")
    print(f"🔧 Analysis config: {args.config}")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No actual execution")
    
    success_count = 0
    total_steps = 0
    
    # Step 1: Run evaluation for each split
    if steps_config.get('evaluation', {}).get('enabled', True):
        eval_config = steps_config['evaluation']
        
        for split in data_config['splits']:
            total_steps += 1
            eval_dir = output_dir / f"evaluation_{split}"
            
            # Skip if exists and configured to do so
            if eval_config.get('skip_if_exists', False) and (eval_dir / "metrics.json").exists():
                print(f"⏭️  Skipping evaluation for {split} (already exists)")
                success_count += 1
                continue
            
            cmd = [
                python_exe, "scripts/evaluate.py",
                "--config", training_config['config_path'],
                "--checkpoint", training_config['checkpoint_path'],
                "--split", split,
                "--output-dir", str(eval_dir),
                "--batch-size", str(eval_config.get('batch_size', 1)),
                "--metrics"] + eval_config.get('metrics', ['dice', 'iou'])
            
            if eval_config.get('generate_report', True):
                cmd.append("--generate-report")
            if eval_config.get('save_predictions', True):
                cmd.append("--save-predictions")
            if data_config.get('data_dir'):
                cmd.extend(["--data-dir", data_config['data_dir']])
            
            if run_command(cmd, f"Evaluation on {split} split", args.dry_run):
                success_count += 1
    
    # Step 2: Generate visualizations
    if steps_config.get('visualization', {}).get('enabled', True):
        vis_config = steps_config['visualization']
        
        for split in data_config['splits']:
            total_steps += 1
            eval_dir = output_dir / f"evaluation_{split}"
            vis_dir = output_dir / f"visualizations_{split}"
            
            # Skip if exists and configured to do so
            if vis_config.get('skip_if_exists', False) and vis_dir.exists():
                print(f"⏭️  Skipping visualization for {split} (already exists)")
                success_count += 1
                continue
            
            pred_dir = eval_dir / "predictions"
            if pred_dir.exists() or args.dry_run:
                cmd = [
                    python_exe, "scripts/visualize_results.py",
                    "--predictions-dir", str(pred_dir),
                    "--ground-truth-dir", data_config.get('data_dir', 'data'),
                    "--images-dir", data_config.get('data_dir', 'data'),
                    "--output-dir", str(vis_dir),
                    "--num-samples", str(vis_config.get('num_samples', 10)),
                    "--modality", vis_config.get('modality', 'flair'),
                    "--slice-selection", vis_config.get('slice_selection', 'center'),
                    "--split", split
                ]
                
                if run_command(cmd, f"Visualization generation for {split}", args.dry_run):
                    success_count += 1
            else:
                print(f"⚠️  Predictions directory not found: {pred_dir}")
    
    # Step 3: Analyze results and generate recommendations
    if steps_config.get('analysis', {}).get('enabled', True):
        total_steps += 1
        print(f"\n{'='*60}")
        print("ANALYZING RESULTS AND GENERATING RECOMMENDATIONS")
        print(f"{'='*60}")
        
        if args.dry_run:
            print("🔍 DRY RUN - Would analyze metrics and generate recommendations")
            success_count += 1
        else:
            # Find metrics file (use validation metrics as primary)
            metrics_file = output_dir / "evaluation_val" / "metrics.json"
            if not metrics_file.exists() and data_config['splits']:
                # Try first available split
                metrics_file = output_dir / f"evaluation_{data_config['splits'][0]}" / "metrics.json"
            
            if metrics_file.exists():
                analysis = analyze_metrics(metrics_file, config)
                
                # Save analysis
                analysis_file = output_dir / "recommendations.json"
                with open(analysis_file, 'w') as f:
                    json.dump(analysis, f, indent=4)
                
                # Generate report
                if steps_config.get('reporting', {}).get('enabled', True):
                    generate_report(output_dir, analysis, config)
                
                # Print summary
                print("\n📈 PERFORMANCE SUMMARY:")
                if "dice_score" in analysis:
                    print(f"   Dice Score: {analysis['dice_score']:.4f}")
                print(f"   Patients: {analysis.get('num_patients', 'N/A')}")
                
                print("\n💡 KEY RECOMMENDATIONS:")
                for rec in analysis.get("recommendations", [])[:5]:  # Top 5
                    print(f"   {rec}")
                
                success_count += 1
            else:
                print(f"⚠️  Metrics file not found: {metrics_file}")
                print("   Skipping analysis. Run evaluation first.")
    
    # Summary
    print(f"\n{'='*60}")
    print("WORKFLOW SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Completed: {success_count}/{total_steps} steps")
    print(f"📁 Results saved to: {output_dir}")
    print(f"⚙️  Configuration used: {args.config}")
    
    if args.dry_run:
        print("🔍 DRY RUN completed - no actual execution performed")
        return 0
    elif success_count == total_steps:
        print("🎉 All steps completed successfully!")
        return 0
    else:
        print("⚠️  Some steps failed. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
