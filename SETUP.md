# Setup & Getting Started

## Installation (5 minutes)

### Step 1: Create Virtual Environment
```bash
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

## Testing Without a Video File

### Option 1: Generate Test Video
```bash
python generate_test_video.py
```

This creates `test_video.mp4` automatically!

### Option 2: Test Python Imports
```bash
python -c "from video_processor import VideoProcessor; print('âœ“ OK')"
python -c "from model_inference import ShotClassifier; print('âœ“ OK')"
python -c "from utils import setup_logging; print('âœ“ OK')"
```

## Running the Application

### 1. Command Line (Easiest)
```bash
python cli.py --video test_video.mp4
```

### 2. Web Interface
```bash
python app.py
# Then visit http://localhost:5000
```

### 3. Python API
```python
from video_processor import VideoProcessor
from model_inference import ShotClassifier

processor = VideoProcessor()
processor.open_video("test_video.mp4")
frames = processor.extract_frames()

classifier = ShotClassifier()
predictions = classifier.predict_frames_batch([f[0] for f in frames])

for (frame, ts), (pred, conf) in zip(frames, predictions):
    print(f"{ts:.2f}s: {pred} ({conf:.2%})")
```

## File Reference

| File | Purpose |
|------|---------|
| `video_processor.py` | Load videos, extract frames |
| `model_inference.py` | Run deep learning model |
| `cli.py` | Command-line interface |
| `app.py` | Flask web server |
| `utils.py` | Helper functions |
| `config.py` | Configuration |
| `generate_test_video.py` | Create test video |
| `requirements.txt` | Dependencies |
| `README.md` | Full documentation |

## Command Line Options

```bash
python cli.py --video <path> [options]

Options:
  --video FILE          Input video file (required)
  --model {resnet50,resnet18}  Model to use (default: resnet50)
  --frame-rate N        Process every Nth frame (default: 2)
  --batch-size N        Batch size for inference (default: 32)
  --output FILE         Output JSON file path
  --log-level {DEBUG,INFO,WARNING,ERROR}  Logging level
```

## Example Workflows

### Analyze a video with faster model
```bash
python cli.py --video basketball.mp4 --model resnet18
```

### Process every 3rd frame (faster)
```bash
python cli.py --video basketball.mp4 --frame-rate 3
```

### Debug mode with detailed logging
```bash
python cli.py --video test_video.mp4 --log-level DEBUG
```

### Save results to specific path
```bash
python cli.py --video basketball.mp4 --output results.json
```

## Common Issues

### "No module named 'torch'"
```bash
pip install torch torchvision
```

### "OpenCV not found"
```bash
pip install opencv-python
```

### "Video won't open"
```bash
# Try converting to MP4
ffmpeg -i video.mov -c:v libx264 video.mp4
python cli.py --video video.mp4
```

### Out of memory
```bash
# Reduce batch size
python cli.py --video shot.mp4 --batch-size 8

# Or use faster model
python cli.py --video shot.mp4 --model resnet18
```

## Performance Tips

1. **Use ResNet18** for CPU systems (much faster)
2. **Increase frame-rate** to skip more frames (faster but less accurate)
3. **Reduce batch-size** if running out of memory
4. **Enable GPU** if available (5-10x faster)

## Next Steps

1. Generate test video: `python generate_test_video.py`
2. Try CLI: `python cli.py --video test_video.mp4`
3. Check results in `results/` directory
4. Explore the code and customize as needed

---
Ready to go! íº€
