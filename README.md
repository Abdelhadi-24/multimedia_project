# 🎬 Simplified MPEG-4 Encoder

A from-scratch implementation of a simplified MPEG-4 video encoder/decoder pipeline in Python, covering colour space conversion, intra-frame DCT coding (I-frames), inter-frame motion-compensated coding (P-frames), and entropy compression via zlib. (for study details pls see `report.pdf`)

---

## 📁 Project Structure

```
project-root/
│
├── frames/                  # Input frames (PNG or JPG images)
│   ├── frame_001.png
│   ├── frame_002.png
│   └── ...
│
├── results/                 # Auto-created on first run
│   ├── output.bin                    # Encoded bitstream
│   ├── reconstructed_frames/         # Decoded output frames
│   └── pipeline_visualisation.png    # Visual summary of the pipeline
│
├── src/
│   ├── preprocess.py        # Part 1 — YCbCr 4:2:0 colour space conversion
│   ├── intra_frame.py       # Part 2 — I-frame DCT + quantisation
│   ├── inter_frame.py       # Part 3 — P-frame motion estimation & residuals
│   ├── entropy_coding.py    # Part 4 — Serialisation + zlib compression
│   ├── evaluate.py          # Part 5a — Compression ratio metrics
│   └── visualise.py         # Part 5b — Pipeline visualisation plot
│
├── extras/                  # Optional utilities (for MP4 testing and report graphs)
│   ├── extract_frames.py    # Extract frames from MP4 video
│   ├── graphs_for_report.py # Generate graphs
│   └── video.mp4            # Sample MP4 video file
│
├── main.py                  # Pipeline entry point
├── requirements.txt
├── report.pdf               # Project report (all the details are here)
└── README.md
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🎞️ Step 0 (Optional) — Extract Frames from a Video

If your source is a `.y4m` file, delete what's inside `frames/` directory and use `ffmpeg` to split it into sequential PNG frames. You control exactly how many frames to extract with the `-frames:v` flag:

```bash
# Extract the first N frames (e.g. 30 frames)
ffmpeg -i input.y4m -frames:v 30 frames/frame_%03d.png
```

```bash
# Extract ALL frames
ffmpeg -i input.y4m frames/frame_%03d.png
```

```bash
# Extract frames at a specific rate (e.g. 1 frame per second)
ffmpeg -i input.y4m -vf fps=1 frames/frame_%03d.png
```

Frames will be saved into the `frames/` directory, which is the default input for the encoder.

---

## � Bonus — Frame Extraction from MP4 (Optional)

If you prefer to test with an MP4 video file instead of Y4M, the `extras/` directory provides `extract_frames.py`:

```bash
cd extras
python extract_frames.py your_video.mp4 [max_frames]
```

This will extract frames to `../frames/` with the naming format `frame_001.png`, `frame_002.png`, etc.

> **Note:** Testing with `.y4m` files via ffmpeg is recommended for best results, as MP4 involves additional codec processing. Use MP4 extraction only if Y4M is unavailable.
> **Why `.y4m` and not `.mp4`?**
> Y4M (YUV4MPEG2) is a raw, uncompressed video container. Unlike MP4, it stores frames without any codec layer, so the pixel values you extract are exactly what the camera captured — no hidden re-compression, no colour re-mapping, no codec artefacts. This guarantees a clean, lossless baseline for evaluating the encoder's own compression quality, independent of any upstream codec decisions.
---

## �🚀 Usage

All modes are accessible through `main.py`:

```bash
python main.py <mode> [options]
```

### Modes

| Mode        | Description                                              |
|-------------|----------------------------------------------------------|
| `encode`    | Encode frames → `results/output.bin`                     |
| `decode`    | Decode `results/output.bin` → reconstructed frames       |
| `evaluate`  | Print compression ratio metrics per frame                |
| `visualise` | Generate the pipeline visualisation PNG                  |
| `all`       | Run all four stages in sequence                          |

### Options

| Flag       | Default  | Description                                     |
|------------|----------|-------------------------------------------------|
| `--dir`    | `frames` | Path to the input frames directory              |
| `--fq`     | `4`      | Quantisation factor (higher = more compression) |
| `--gop`    | `8`      | GOP size (I-frame every N frames)               |
| `--search` | `8`      | Motion search window radius (pixels)            |

### Examples

```bash
# Full pipeline with default settings
python main.py all

# Encode only, with custom quality and GOP
python main.py encode --fq 6 --gop 16 --search 12

# Decode a previously encoded bitstream
python main.py decode

# Evaluate compression ratios
python main.py evaluate

# Generate visualisation from a custom frames folder
python main.py visualise --dir my_frames
```

---

## 🔬 Pipeline Overview

```
Input Frames (PNG/JPG)
        │
        ▼
① Preprocess      bgr_to_ycbcr420()    — BGR → Y, Cb, Cr with 4:2:0 subsampling
        │
        ▼
② Intra Coding    encode_frame()       — 8×8 DCT + quantisation  →  I-frame coefficients
        │
        ▼
③ Inter Coding    encode_pframe()      — Block matching (MSE) + residual DCT  →  P-frames
        │
        ▼
④ Entropy Coding  compress_frame()     — pickle + zlib level-9  →  output.bin
        │
        ▼
⑤a Evaluate       evaluate()           — Per-frame compression ratio table
⑤b Visualise      visualise()          — Multi-panel matplotlib figure
```

---

## 📊 Output Example

```
=================================================================
                  EVALUATION — PART 5a
=================================================================
  Frame                Type      Original   Compressed    Ratio
-----------------------------------------------------------------
  frame_001.png        I         312.4KB      48.7KB      6.41x
  frame_002.png        P         312.4KB      12.3KB     25.40x
  ...
=================================================================
  TOTAL                          2498.0KB     198.4KB     12.59x
-----------------------------------------------------------------
  I-frames :  2 / 16  (12.5%)
  P-frames : 14 / 16  (87.5%)
=================================================================
```

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt).

