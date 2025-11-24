"""Flask web application for basketball shot analysis."""

import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify
from werkzeug.utils import secure_filename

from video_processor import VideoProcessor
from model_inference import ShotClassifier
from utils import setup_logging, save_predictions_to_json, format_predictions_report, create_results_directory

setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 500
app.config["UPLOAD_FOLDER"] = Path("uploads")
app.config["RESULTS_FOLDER"] = Path("results")

app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)
app.config["RESULTS_FOLDER"].mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}

classifier = None


def get_classifier():
    """Get or initialize the model classifier."""
    global classifier
    if classifier is None:
        logger.info("Initializing model classifier...")
        classifier = ShotClassifier(model_name="resnet50")
    return classifier


@app.route("/")
def index():
    """Render main page."""
    return jsonify({"message": "Basketball Shot Analysis API", "status": "running"})


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    try:
        classifier = get_classifier()
        return jsonify({
            "status": "healthy",
            "model_loaded": classifier.is_loaded,
            "device": classifier.device,
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server error."""
    logger.error(f"Internal error: {error}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("Starting Basketball Shot Analysis Web Application")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER'].absolute()}")
    logger.info(f"Results folder: {app.config['RESULTS_FOLDER'].absolute()}")
    app.run(debug=True, host="0.0.0.0", port=5000)
