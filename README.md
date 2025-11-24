# Basketball Jump Shot Classification MVP

A Python computer vision application that takes a basketball video file as input and leveraging CV video processing tool openCV2 and DL model resnet50 to predict jump shot outcomes using CNN architecture.

### Python API
```python
from video_processor import VideoProcessor
from model_inference import ShotClassifier

processor = VideoProcessor()
processor.open_video("video.mp4")
frames = processor.extract_frames()

classifier = ShotClassifier()
predictions = classifier.predict_frames_batch([f[0] for f in frames])
```

### Web App
```bash
python app.py
# Visit http://localhost:5000
```

## Features

- Video processing with OpenCV
- Pretrained model inference (ResNet50, ResNet18)
- Make/miss classification
- JSON output with timestamps & confidence
- GPU support
- CLI and Flask interfaces

## Output

```json
{
  "predictions": [
    {"timestamp": 0.0, "prediction": "make", "confidence": 0.945}
  ],
  "summary": {
    "makes": 35,
    "misses": 13,
    "avg_confidence": 0.823
  }
}
```

## Files

- `video_processor.py` - Video handling
- `model_inference.py` - Deep learning model
- `cli.py` - Command line tool
- `app.py` - Flask web app
- `utils.py` - Utilities
- `config.py` - Configuration


3. Implement batch video processing
4. Deploy to cloud

---
Status: ✅ Ready to use
Python: 3.9+
