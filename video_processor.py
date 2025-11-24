"""
Video processing module for basketball shot analysis.

Handles video loading, frame extraction, and key frame detection
for jump shot analysis.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Process basketball videos and extract frames for analysis."""

    def __init__(self, frame_sampling_rate: int = 2):
        """
        Initialize the video processor.

        Args:
            frame_sampling_rate: Process every nth frame (default: 2)
        """
        self.frame_sampling_rate = frame_sampling_rate
        self.video_path = None
        self.fps = None
        self.total_frames = None
        self.frame_width = None
        self.frame_height = None

    def validate_video_file(self, video_path: str) -> bool:
        """
        Validate that the video file exists and is readable.

        Args:
            video_path: Path to video file

        Returns:
            bool: True if valid, False otherwise
        """
        path = Path(video_path)

        if not path.exists():
            logger.error(f"Video file not found: {video_path}")
            return False

        if not path.is_file():
            logger.error(f"Path is not a file: {video_path}")
            return False

        valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        if path.suffix.lower() not in valid_extensions:
            logger.warning(f"Unusual video extension: {path.suffix}")

        return True

    def open_video(self, video_path: str) -> bool:
        """
        Open and initialize video capture.

        Args:
            video_path: Path to video file

        Returns:
            bool: True if successfully opened, False otherwise
        """
        if not self.validate_video_file(video_path):
            return False

        try:
            self.video_path = video_path
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return False

            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if self.fps <= 0 or self.total_frames <= 0:
                logger.error(f"Invalid video properties - FPS: {self.fps}, Frames: {self.total_frames}")
                cap.release()
                return False

            cap.release()
            logger.info(f"Successfully opened video: {video_path}")
            return True

        except Exception as e:
            logger.error(f"Error opening video file: {e}")
            return False

    def extract_frames(self) -> List[Tuple[np.ndarray, float]]:
        """Extract frames from video at specified sampling rate."""
        if self.video_path is None:
            raise RuntimeError("Video must be opened first using open_video()")

        frames = []
        cap = cv2.VideoCapture(self.video_path)

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % self.frame_sampling_rate == 0:
                    timestamp = frame_count / self.fps
                    frames.append((frame, timestamp))

                frame_count += 1

        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            raise

        finally:
            cap.release()

        logger.info(f"Extracted {len(frames)} frames from video")
        return frames

    def get_video_info(self) -> dict:
        """Get information about the opened video."""
        if self.video_path is None:
            return {}

        return {
            "path": self.video_path,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_seconds": self.total_frames / self.fps if self.fps > 0 else 0,
            "width": self.frame_width,
            "height": self.frame_height,
        }
