"""
preprocess.py — Part 1: Color space conversion + chroma subsampling
"""

import cv2
import numpy as np


def bgr_to_ycbcr420(bgr):
    """BGR uint8 → (Y, Cb, Cr) with 4:2:0 chroma subsampling."""
    ycbcr = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = ycbcr[:,:,0], ycbcr[:,:,1], ycbcr[:,:,2]
    down = lambda ch: cv2.resize(ch, (ch.shape[1]//2, ch.shape[0]//2), interpolation=cv2.INTER_AREA)
    return Y, down(Cb), down(Cr)


def ycbcr420_to_bgr(Y, Cb, Cr):
    """Invert: upsample chroma → YCbCr → BGR."""
    h, w = Y.shape
    up = lambda ch: cv2.resize(ch, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(cv2.merge([Y, up(Cr), up(Cb)]), cv2.COLOR_YCrCb2BGR)
