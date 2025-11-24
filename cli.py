"""Command-line interface for basketball shot analysis."""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

from video_processor import VideoProcessor
from model_inference import ShotClassifier
from utils import (
    setup_logging,
    save_predictions_to_json,
    format_predictions_report,
    create_results_directory,
)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze basketball videos to predict jump shot outcomes",
    )

    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to output JSON file")
    parser.add_argument("--frame-rate", type=int, default=2, help="Process every nth frame")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "resnet18"])
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("BASKETBALL SHOT ANALYSIS CLI")
    logger.info("=" * 70)

    try:
        logger.info(f"Video file: {args.video}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Frame sampling rate: {args.frame-rate}")

        if not Path(args.video).exists():
            logger.error(f"Video file not found: {args.video}")
            return 1

        if args.output:
            output_path = args.output
        else:
            results_dir = create_results_directory()
            output_path = str(Path(results_dir) / "predictions.json")

        logger.info(f"Output file: {output_path}")

        # Initialize video processor
        logger.info("Initializing video processor...")
        processor = VideoProcessor(frame_sampling_rate=args.frame_rate)

        if not processor.open_video(args.video):
            logger.error("Failed to open video file")
            return 1

        video_info = processor.get_video_info()
        logger.info(f"Video: {video_info['duration_seconds']:.2f}s, {video_info['width']}x{video_info['height']}")

        # Extract frames
        logger.info("Extracting frames...")
        frames_to_analyze = processor.extract_frames()
        logger.info(f"Extracted {len(frames_to_analyze)} frames")

        if not frames_to_analyze:
            logger.error("No frames extracted")
            return 1

        # Initialize model
        logger.info(f"Loading {args.model} model...")
        classifier = ShotClassifier(model_name=args.model)
        logger.info(f"Device: {classifier.device}")

        # Make predictions
        logger.info("Running inference...")
        frame_arrays = [f[0] for f in frames_to_analyze]
        frame_predictions = classifier.predict_frames_batch(frame_arrays, batch_size=args.batch_size)

        # Compile results
        predictions = []
        for (frame, timestamp), (prediction, confidence) in zip(frames_to_analyze, frame_predictions):
            predictions.append({
                "timestamp": round(timestamp, 2),
                "prediction": prediction,
                "confidence": round(confidence, 4),
            })

        # Save results
        logger.info("Saving results...")
        result_data = {
            "timestamp": datetime.now().isoformat(),
            "video_file": args.video,
            "video_info": video_info,
            "predictions": predictions,
            "summary": {
                "total_frames_analyzed": len(predictions),
                "makes": sum(1 for p in predictions if p["prediction"] == "make"),
                "misses": sum(1 for p in predictions if p["prediction"] == "miss"),
                "avg_confidence": round(sum(p["confidence"] for p in predictions) / len(predictions), 4) if predictions else 0,
            },
        }

        if not save_predictions_to_json(result_data, output_path):
            logger.error("Failed to save results")
            return 1

        # Print report
        report = format_predictions_report(predictions, video_info)
        logger.info(report)

        logger.info("=" * 70)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
