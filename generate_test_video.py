"""Generate a test video for the application."""

import cv2
import numpy as np
from pathlib import Path


def generate_test_video(filename="test_video.mp4", duration_seconds=5, fps=30):
    """Generate a simple test video with random frames."""
    width, height = 640, 480
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ Failed to create video writer")
        return False
    
    total_frames = duration_seconds * fps
    
    print(f"Generating {duration_seconds}s test video ({total_frames} frames)...")
    
    for i in range(total_frames):
        # Create a frame with some pattern
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Add some text to indicate frame number
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        
        if (i + 1) % 30 == 0:
            print(f"  Progress: {i + 1}/{total_frames} frames")
    
    out.release()
    
    print(f"✓ Test video created: {filename}")
    print(f"  Duration: {duration_seconds}s")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Frames: {total_frames}")
    
    return True


if __name__ == "__main__":
    import sys
    
    filename = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"
    
    print("=" * 60)
    print("TEST VIDEO GENERATOR")
    print("=" * 60)
    
    if generate_test_video(filename):
        print("\nReady to test with:")
        print(f"  python cli.py --video {filename}")
        sys.exit(0)
    else:
        print("\n❌ Failed to generate test video")
        sys.exit(1)
