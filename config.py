"""Configuration settings for the Basketball Shot Analysis application."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
UPLOADS_DIR = PROJECT_ROOT / "uploads"
RESULTS_DIR = PROJECT_ROOT / "results"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

UPLOADS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
DEFAULT_FRAME_SAMPLING_RATE = 2
DEFAULT_MOTION_THRESHOLD = 5000

DEFAULT_MODEL = "resnet50"
AVAILABLE_MODELS = ["resnet50", "resnet18", "efficientnet_b0"]

DEFAULT_BATCH_SIZE = 32
MODEL_INPUT_SIZE = 224

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
MAX_FILE_SIZE = 500 * 1024 * 1024

DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

USE_GPU = True
NUM_THREADS = os.getenv("OMP_NUM_THREADS", "4")

ENABLE_MOTION_DETECTION = True
ENABLE_TEMPORAL_ANALYSIS = True
ENABLE_GPU_OPTIMIZATION = True
