"""
extract_frames.py
-----------------
Split a video file into sequential PNG frames.
Usage: python extract_frames.py <video_path> [frames_dir]
"""

import cv2
import os
import sys

def extract_frames(video_path, output_dir="frames", max_frames=60):
    """Extract frames from video, capped at max_frames for speed."""
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
        path = os.path.join(output_dir, f"frame_{saved:04d}.png")
        cv2.imwrite(path, frame)
        saved += 1

    cap.release()
    print(f"Extracted {saved} frames from {total} total (fps={fps:.1f})")
    print(f"Frames saved to: {output_dir}/")

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "sample-5s.mp4"
    out   = sys.argv[2] if len(sys.argv) > 2 else "frames"
    extract_frames(video, out)