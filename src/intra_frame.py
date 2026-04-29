"""
intra_frame.py — Part 2: Intra-frame coding (I-frames) with DCT + quantisation
"""

import numpy as np
import cv2



# Quantisation matrix

def make_quant(fq: int) -> np.ndarray:
    return np.fromfunction(lambda i, j: 1 + (1 + i + j) * fq, (8, 8), dtype=np.float32)



# Padding

def pad_channel(channel):
    h, w = channel.shape
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    padded = np.pad(channel.astype(np.float32), ((0, ph), (0, pw)), 'constant')
    return padded, (h, w)



# Block-by-block DCT + Quantisation

def encode_channel(channel, Q):
    padded, orig_hw = pad_channel(channel)
    H, W = padded.shape
    blocks_h = H // 8
    blocks_w = W // 8
    coeffs = np.zeros((blocks_h, blocks_w, 8, 8), dtype=np.int16)

    for i in range(blocks_h):
        for j in range(blocks_w):
            block = padded[i*8:(i+1)*8, j*8:(j+1)*8] - 128
            dct_block = cv2.dct(block.astype(np.float32))
            coeffs[i, j] = np.round(dct_block / Q).astype(np.int16)

    return coeffs, padded.shape, orig_hw


def decode_channel(coeffs, HW, orig_hw, Q):
    H, W = HW
    recon = np.zeros((H, W), dtype=np.float32)

    for i in range(coeffs.shape[0]):
        for j in range(coeffs.shape[1]):
            block = cv2.idct(coeffs[i, j].astype(np.float32) * Q) + 128
            recon[i*8:(i+1)*8, j*8:(j+1)*8] = block

    h, w = orig_hw
    return np.clip(recon[:h, :w], 0, 255).astype(np.uint8)



# Encode / Decode Frame

def encode_frame(Y, Cb, Cr, Q_luma, Q_chroma):
    Yq,  hw_Y,  ohw_Y  = encode_channel(Y,  Q_luma)
    Cbq, hw_Cb, ohw_Cb = encode_channel(Cb, Q_chroma)
    Crq, hw_Cr, ohw_Cr = encode_channel(Cr, Q_chroma)
    return (Yq, Cbq, Crq), (hw_Y, hw_Cb, hw_Cr), (ohw_Y, ohw_Cb, ohw_Cr)


def decode_frame(Yq, Cbq, Crq, hws, ohws, Q_luma, Q_chroma):
    Y  = decode_channel(Yq,  hws[0], ohws[0], Q_luma)
    Cb = decode_channel(Cbq, hws[1], ohws[1], Q_chroma)
    Cr = decode_channel(Crq, hws[2], ohws[2], Q_chroma)
    return Y, Cb, Cr
