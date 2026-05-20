"""
extract_frames.py
-----------------
Split a video file into sequential PNG frames.
Usage: python extract_frames.py <video_path> [frames_dir] [max_frames]
"""

import cv2
import os
import sys

def extract_frames(video_path, max_frames, output_dir=None):
    """Extract frames from video, capped at max_frames for speed."""
    if output_dir is None:
        output_dir = "frames"
    
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(__file__), "..", output_dir)
    
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)

    saved = 0
    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        path = os.path.join(output_dir, f"frame_{saved:03d}.png")
        cv2.imwrite(path, frame)
        saved += 1

    cap.release()
    print(f"Extracted {saved} frames from {total} total (fps={fps:.1f})")
    print(f"Frames saved to: {output_dir}/")

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "akiyo.mp4"
    out   = sys.argv[2] if len(sys.argv) > 2 else "frames"
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    extract_frames(video, max_frames, out)