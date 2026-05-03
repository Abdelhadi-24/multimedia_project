"""
entropy_coding.py — Part 4: Entropy coding (lossless compression)
Serialises all coded frames into a single .bin file using zlib.
"""

import numpy as np
import os
import sys
import zlib
import struct
import pickle
import cv2

from intra_frame import decode_frame as intra_decode_frame, make_quant
from inter_frame import decode_pframe, apply_mv
from preprocess  import ycbcr420_to_bgr


def compress_frame(frame_dict):
    return zlib.compress(pickle.dumps(frame_dict), level=9)


def decompress_frame(data):
    return pickle.loads(zlib.decompress(data))



def encode(frames_dir, fq=4, gop_size=8, search_win=8, out_bin="output.bin"):
    from preprocess  import bgr_to_ycbcr420
    from intra_frame import encode_frame, make_quant
    from inter_frame import encode_pframe, decode_pframe

    files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg")))
    if not files:
        print(f"No frames in '{frames_dir}'"); sys.exit(1)

    Q_luma   = make_quant(fq)
    Q_chroma = make_quant(fq * 2)

    frame_blobs = []
    ref_Y = ref_Cb = ref_Cr = None
    frame_types = []

    for idx, fname in enumerate(files):
        bgr = cv2.imread(os.path.join(frames_dir, fname))
        Y, Cb, Cr = bgr_to_ycbcr420(bgr)

        if idx % gop_size == 0:  # I-frame
            (Yq, Cbq, Crq), hws, ohws = encode_frame(Y, Cb, Cr, Q_luma, Q_chroma)
            frame_data = dict(
                frame_type="I", fq=fq,
                Yq=Yq, Cbq=Cbq, Crq=Crq,
                hw_Y=hws[0],  hw_Cb=hws[1],  hw_Cr=hws[2],
                ohw_Y=ohws[0], ohw_Cb=ohws[1], ohw_Cr=ohws[2],
            )
            ref_Y, ref_Cb, ref_Cr = intra_decode_frame(Yq, Cbq, Crq, hws, ohws, Q_luma, Q_chroma)
            frame_types.append("I")
            print(f"  [{idx:03d}] I-frame: {fname}")

        else:  # P-frame
            mv_map, results = encode_pframe(Y, Cb, Cr, ref_Y, ref_Cb, ref_Cr, Q_luma, Q_chroma, search_win=search_win)
            frame_data = dict(frame_type="P", fq=fq, mv_map=mv_map)
            for key, (C, HW, ohw) in results.items():
                frame_data[f"{key}q"]    = C
                frame_data[f"hw_{key}"]  = HW
                frame_data[f"ohw_{key}"] = ohw
            ref_Y, ref_Cb, ref_Cr = decode_pframe(mv_map, results, ref_Y, ref_Cb, ref_Cr, Q_luma, Q_chroma)
            frame_types.append("P")
            print(f"  [{idx:03d}] P-frame: {fname}")

        frame_blobs.append(compress_frame(frame_data))

    with open(out_bin, "wb") as f:
        f.write(struct.pack(">I", len(frame_blobs)))
        for blob in frame_blobs:
            f.write(struct.pack(">I", len(blob)))
            f.write(blob)

    print(f"\nEncoded {len(files)} frame(s) → '{out_bin}'  (GOP={gop_size}, S={search_win}, fq={fq})")
    return frame_types



def decode(in_bin="output.bin", out_dir="reconstructed_frames"):
    os.makedirs(out_dir, exist_ok=True)

    with open(in_bin, "rb") as f:
        num_frames = struct.unpack(">I", f.read(4))[0]
        blobs = []
        for _ in range(num_frames):
            size = struct.unpack(">I", f.read(4))[0]
            blobs.append(f.read(size))

    ref_Y = ref_Cb = ref_Cr = None

    for idx, blob in enumerate(blobs):
        d    = decompress_frame(blob)
        fq   = int(d["fq"])
        Ql   = make_quant(fq)
        Qc   = make_quant(fq * 2)
        stem = f"frame_{idx+1:03d}"
        frame_type = str(d["frame_type"])

        if frame_type == "I":
            hws  = (d["hw_Y"],  d["hw_Cb"],  d["hw_Cr"])
            ohws = (d["ohw_Y"], d["ohw_Cb"], d["ohw_Cr"])
            Y, Cb, Cr = intra_decode_frame(d["Yq"], d["Cbq"], d["Crq"], hws, ohws, Ql, Qc)
        else:
            mv = d["mv_map"]
            results = {k: (d[f"{k}q"], tuple(d[f"hw_{k}"]), tuple(d[f"ohw_{k}"])) for k in ("Y", "Cb", "Cr")}
            Y, Cb, Cr = decode_pframe(mv, results, ref_Y, ref_Cb, ref_Cr, Ql, Qc)

        ref_Y, ref_Cb, ref_Cr = Y, Cb, Cr
        cv2.imwrite(os.path.join(out_dir, f"{stem}.png"), ycbcr420_to_bgr(Y, Cb, Cr))

    print(f"Decoded {num_frames} frame(s) → '{out_dir}/'")