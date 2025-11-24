# Basketball Jump Shot Classification MVP

A Python computer vision application that analyzes basketball videos to predict jump shot outcomes using deep learning.

## Quick Start

### 1. Install
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Test Video
```bash
python generate_test_video.py
```

### 3. Analyze
```bash
python cli.py --video test_video.mp4
```

## Usage

### Command Line
```bash
python cli.py --video basketball.mp4 --model resnet18 --frame-rate 2
```

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

## Troubleshooting

Video won't open?
```bash
ffmpeg -i video.mov -c:v libx264 video.mp4
```

Out of memory?
```bash
python cli.py --video shot.mp4 --batch-size 8
```

Module errors?
```bash
pip install -r requirements.txt --force-reinstall
```

## Next Steps

1. Fine-tune model on real basketball data
2. Add player/ball detection
3. Implement batch video processing
4. Deploy to cloud

---
Status: ✅ Ready to use
Python: 3.9+
