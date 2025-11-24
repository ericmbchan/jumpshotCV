"""Utility functions for basketball shot analysis application."""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import os


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure logging for the application."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level = getattr(logging, log_level.upper(), logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    print(f"Logging level set to: {logging.getLevelName(log_level)}")


def create_results_directory(base_path: str = "results") -> str:
    """Create a timestamped results directory for output files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(base_path) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    logging.getLogger(__name__).info(f"Created results directory: {results_dir}")
    return str(results_dir)


def save_predictions_to_json(predictions: List[Dict[str, Any]], output_path: str) -> bool:
    """Save predictions to JSON file."""
    logger = logging.getLogger(__name__)

    try:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)

        logger.info(f"Saved predictions to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving predictions to JSON: {e}")
        return False


def format_predictions_report(predictions: List[Dict[str, Any]], video_info: Dict[str, Any]) -> str:
    """Format predictions into a human-readable report."""
    report = []
    report.append("\n" + "=" * 60)
    report.append("BASKETBALL SHOT ANALYSIS REPORT")
    report.append("=" * 60)

    report.append("\nVIDEO INFORMATION:")
    report.append(f"  File: {video_info.get('path', 'N/A')}")
    report.append(f"  Duration: {video_info.get('duration_seconds', 0):.2f} seconds")
    report.append(f"  Resolution: {video_info.get('width', 0)}x{video_info.get('height', 0)}")
    report.append(f"  FPS: {video_info.get('fps', 0):.2f}")

    makes = sum(1 for p in predictions if p.get("prediction") == "make")
    misses = sum(1 for p in predictions if p.get("prediction") == "miss")
    total = len(predictions)

    report.append("\nANALYSIS SUMMARY:")
    report.append(f"  Total Frames Analyzed: {total}")
    report.append(f"  Predicted Makes: {makes} ({100*makes/total if total > 0 else 0:.1f}%)")
    report.append(f"  Predicted Misses: {misses} ({100*misses/total if total > 0 else 0:.1f}%)")

    if predictions:
        avg_confidence = sum(p.get("confidence", 0) for p in predictions) / len(predictions)
        report.append(f"  Average Confidence: {avg_confidence:.4f}")

    report.append("\n" + "=" * 60 + "\n")

    return "\n".join(report)
