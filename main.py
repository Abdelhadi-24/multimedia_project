"""
main.py — Full pipeline runner for the Simplified MPEG-4 Encoder
Usage:
  python main.py encode [--dir frames] [--fq 4] [--gop 8] [--search 8]
  python main.py decode
  python main.py evaluate
  python main.py visualise
  python main.py all [--dir frames] [--fq 4] [--gop 8] [--search 8]
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from entropy_coding import encode, decode
from evaluate       import evaluate
from visualise      import visualise

RESULTS_DIR   = "results"
RECONSTRUCTED = os.path.join(RESULTS_DIR, "reconstructed_frames")
BIN_FILE      = os.path.join(RESULTS_DIR, "output.bin")

os.makedirs(RECONSTRUCTED, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="MPEG-4 Simplified Encoder Pipeline")
    parser.add_argument("mode", choices=["encode", "decode", "evaluate", "visualise", "all"])
    parser.add_argument("--dir",    default="frames")
    parser.add_argument("--fq",     default=4,  type=int)
    parser.add_argument("--gop",    default=8,  type=int)
    parser.add_argument("--search", default=8,  type=int)
    args = parser.parse_args()

    if args.mode in ("encode", "all"):
        print("\n── Step 1: Encoding → results/output.bin ──────────────────────")
        encode(args.dir, args.fq, args.gop, args.search, BIN_FILE)

    if args.mode in ("decode", "all"):
        print("\n── Step 2: Decoding results/output.bin ────────────────────────")
        decode(BIN_FILE, RECONSTRUCTED)

    if args.mode in ("evaluate", "all"):
        print("\n── Step 3: Evaluation ─────────────────────────────────────────")
        evaluate(args.dir, BIN_FILE)

    if args.mode in ("visualise", "all"):
        print("\n── Step 4: Pipeline Visualisation ─────────────────────────────")
        visualise(args.dir, RECONSTRUCTED, BIN_FILE, fq=args.fq)

if __name__ == "__main__":
    main()