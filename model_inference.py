"""Model inference module for basketball shot classification."""

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np
from typing import Tuple, Optional, Dict
import logging
import cv2

logger = logging.getLogger(__name__)


class ShotClassifier:
    """Basketball shot classification model using pretrained models."""

    def __init__(self, model_name: str = "resnet50", device: Optional[str] = None):
        """Initialize the shot classifier."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Install with: pip install torch torchvision")
            self.is_loaded = False
            return

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.transform = None
        self.is_loaded = False

        logger.info(f"Using device: {self.device}")
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the pretrained model and transformation pipeline."""
        if not TORCH_AVAILABLE:
            return

        try:
            if self.model_name == "resnet50":
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            elif self.model_name == "resnet18":
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 2),
            )

            self.model = model.to(self.device)
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            self.is_loaded = True
            logger.info(f"Successfully loaded {self.model_name} model")

        except Exception as e:
            logger.error(f"Error initializing model: {e}")

    def predict_frame(self, frame: np.ndarray) -> Tuple[str, float]:
        """Make prediction on a single frame."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)

            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

            class_names = ["miss", "make"]
            prediction_label = class_names[predicted_class]

            return prediction_label, confidence

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise

    def predict_frames_batch(self, frames: list, batch_size: int = 32) -> list:
        """Make predictions on multiple frames in batches."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")

        predictions = []

        try:
            for i in range(0, len(frames), batch_size):
                batch = frames[i : i + batch_size]

                batch_tensors = []
                for frame in batch:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tensor = self.transform(frame_rgb)
                    batch_tensors.append(tensor)

                input_tensor = torch.stack(batch_tensors).to(self.device)

                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)

                predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()
                confidences = probabilities.max(dim=1).values.cpu().numpy()

                class_names = ["miss", "make"]
                for pred_class, confidence in zip(predicted_classes, confidences):
                    label = class_names[pred_class]
                    predictions.append((label, float(confidence)))

            logger.info(f"Batch predictions complete for {len(frames)} frames")
            return predictions

        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {}

        return {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self.is_loaded,
        }
