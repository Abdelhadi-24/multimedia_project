"""
inter_frame.py — Part 3: Inter-frame coding (P-frames)
Motion estimation (block matching with MSE) + residual DCT coding
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from intra_frame import make_quant, encode_channel, decode_channel, encode_frame, decode_frame


# Residual encode/decode

def encode_residual(residual, Q):
    return encode_channel(residual + 128, Q)


def decode_residual(coeffs, HW, orig_hw, Q):
    return decode_channel(coeffs, HW, orig_hw, Q).astype(np.float32) - 128



# Motion Estimation

def motion_estimate(cur_Y, ref_Y, mb_size=16, search_win=8):
    h, w = cur_Y.shape
    mh = (h + mb_size - 1) // mb_size
    mw = (w + mb_size - 1) // mb_size

    mv_map = np.zeros((mh, mw, 2), dtype=np.int16)
    pred_Y = np.zeros_like(cur_Y, dtype=np.float32)

    pad = search_win
    ref_pad = np.pad(ref_Y.astype(np.float32), pad, mode='edge')

    for mi in range(mh):
        for mj in range(mw):
            y0, x0 = mi * mb_size, mj * mb_size
            y1, x1 = min(y0 + mb_size, h), min(x0 + mb_size, w)
            bh, bw  = y1 - y0, x1 - x0

            cur_block = cur_Y[y0:y1, x0:x1].astype(np.float32)

            ry0 = y0 + pad - search_win
            rx0 = x0 + pad - search_win
            search_region = ref_pad[ry0:ry0 + bh + 2*search_win,
                                    rx0:rx0 + bw + 2*search_win]

            candidates = sliding_window_view(search_region, (bh, bw))
            mse_map    = np.mean((candidates - cur_block) ** 2, axis=(-2, -1))

            best_dy, best_dx = np.unravel_index(np.argmin(mse_map), mse_map.shape)
            dy, dx = best_dy - search_win, best_dx - search_win

            mv_map[mi, mj] = (dy, dx)
            ry = y0 + pad + dy
            rx = x0 + pad + dx
            pred_Y[y0:y1, x0:x1] = ref_pad[ry:ry + bh, rx:rx + bw]

    return mv_map, pred_Y.astype(np.uint8)


def apply_mv(ref_ch, mv_map, mb_size=16, scale=1.0):
    h, w = ref_ch.shape
    pad  = mb_size
    ref_pad = np.pad(ref_ch.astype(np.float32), pad, mode='edge')
    pred = np.zeros((h, w), dtype=np.float32)
    bsize = max(1, int(mb_size * scale))

    for mi in range(mv_map.shape[0]):
        for mj in range(mv_map.shape[1]):
            y0, x0 = int(mi * bsize), int(mj * bsize)
            y1, x1 = min(y0 + bsize, h), min(x0 + bsize, w)
            dy = int(round(int(mv_map[mi, mj, 0]) * scale))
            dx = int(round(int(mv_map[mi, mj, 1]) * scale))
            ry, rx = y0 + pad + dy, x0 + pad + dx
            pred[y0:y1, x0:x1] = ref_pad[ry:ry + y1 - y0, rx:rx + x1 - x0]

    return np.clip(pred, 0, 255).astype(np.uint8)


# Encode / Decode one P-frame

def encode_pframe(cur_Y, cur_Cb, cur_Cr, ref_Y, ref_Cb, ref_Cr, Q_luma, Q_chroma, mb_size=16, search_win=8):
    mv_map, pred_Y = motion_estimate(cur_Y, ref_Y, mb_size, search_win)
    pred_Cb = apply_mv(ref_Cb, mv_map, mb_size, scale=0.5)
    pred_Cr = apply_mv(ref_Cr, mv_map, mb_size, scale=0.5)

    results = {}
    for key, cur, pred, Q in [("Y", cur_Y, pred_Y, Q_luma),
                               ("Cb", cur_Cb, pred_Cb, Q_chroma),
                               ("Cr", cur_Cr, pred_Cr, Q_chroma)]:
        res = cur.astype(np.float32) - pred.astype(np.float32)
        results[key] = encode_residual(res, Q)

    return mv_map, results


def decode_pframe(mv_map, results, ref_Y, ref_Cb, ref_Cr, Q_luma, Q_chroma, mb_size=16):
    pred_Y  = apply_mv(ref_Y,  mv_map, mb_size, scale=1.0)
    pred_Cb = apply_mv(ref_Cb, mv_map, mb_size, scale=0.5)
    pred_Cr = apply_mv(ref_Cr, mv_map, mb_size, scale=0.5)

    out = {}
    for key, pred, Q in [("Y", pred_Y, Q_luma), ("Cb", pred_Cb, Q_chroma), ("Cr", pred_Cr, Q_chroma)]:
        C, HW, ohw = results[key]
        out[key] = np.clip(pred.astype(np.float32) + decode_residual(C, HW, ohw, Q), 0, 255).astype(np.uint8)

    return out["Y"], out["Cb"], out["Cr"]
