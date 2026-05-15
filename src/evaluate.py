"""
evaluate.py — Part 5a: Quality Metrics
"""
import os
import struct


def read_blob_sizes(bin_file):
    sizes = []
    with open(bin_file, "rb") as f:
        num_frames = struct.unpack(">I", f.read(4))[0]
        for _ in range(num_frames):
            size = struct.unpack(">I", f.read(4))[0]
            f.seek(size, 1)
            sizes.append(size)
    return sizes


def read_frame_types(bin_file):
    import pickle, zlib
    types = []
    with open(bin_file, "rb") as f:
        num_frames = struct.unpack(">I", f.read(4))[0]
        for _ in range(num_frames):
            size = struct.unpack(">I", f.read(4))[0]
            blob = f.read(size)
            d = pickle.loads(zlib.decompress(blob))
            types.append(str(d["frame_type"]))
    return types


def evaluate(frames_dir="frames", bin_file="output.bin"):
    if not os.path.exists(bin_file):
        print(f"Error: '{bin_file}' not found."); return
    if not os.path.isdir(frames_dir):
        print(f"Error: '{frames_dir}' not found."); return

    orig_files  = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg")))
    blob_sizes  = read_blob_sizes(bin_file)
    frame_types = read_frame_types(bin_file)

    print("=" * 65)
    print("                  EVALUATION — PART 5a")
    print("=" * 65)
    print(f"  {'Frame':<20} {'Type':<8} {'Original':>12} {'Compressed':>12} {'Ratio':>8}")
    print("-" * 65)

    i_count = p_count = 0
    total_orig = total_comp = 0

    for orig_fname, ftype, comp_size in zip(orig_files, frame_types, blob_sizes):
        orig_size = os.path.getsize(os.path.join(frames_dir, orig_fname))
        ratio = orig_size / comp_size
        if ftype == "I": i_count += 1
        else:            p_count += 1
        total_orig += orig_size
        total_comp += comp_size
        print(f"  {orig_fname:<20} {ftype:<8} {orig_size/1024:>10.1f}KB {comp_size/1024:>10.1f}KB {ratio:>7.2f}x")

    total = i_count + p_count
    print("=" * 65)
    print(f"  {'TOTAL':<20} {'':8} {total_orig/1024:>10.1f}KB {total_comp/1024:>10.1f}KB {total_orig/total_comp:>7.2f}x")
    print("-" * 65)
    print(f"  I-frames : {i_count} / {total}  ({100*i_count/total:.1f}%)")
    print(f"  P-frames : {p_count} / {total}  ({100*p_count/total:.1f}%)")
    print("=" * 65)

    return total_orig / total_comp, i_count, p_count