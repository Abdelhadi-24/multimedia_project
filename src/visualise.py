"""
visualise.py — Part 5b: Pipeline Visualisation
Produces a single figure with 5 rows, one per pipeline stage.
"""

import os, pickle, zlib, struct
import cv2, numpy as np, matplotlib, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from preprocess  import bgr_to_ycbcr420, ycbcr420_to_bgr
from intra_frame import make_quant
from inter_frame import motion_estimate, apply_mv



def _read_bin(bin_file):
    frames = []
    with open(bin_file, "rb") as f:
        for _ in range(struct.unpack(">I", f.read(4))[0]):
            sz = struct.unpack(">I", f.read(4))[0]
            frames.append(pickle.loads(zlib.decompress(f.read(sz))))
    return frames

def _load_bgrs(d):
    files = sorted(f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg")))
    return [cv2.imread(os.path.join(d, f)) for f in files], files

def _show(ax, img, title="", cmap=None, vmin=None, vmax=None, colorbar=False, fig=None):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=7, color="#ddd", pad=3)
    ax.axis("off")
    if colorbar and fig is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=5, colors="#aaa")

def _mv_overlay(frame_Y, mv_map, max_mag=8.0):
    vis    = cv2.cvtColor(frame_Y, cv2.COLOR_GRAY2BGR)
    mh, mw = mv_map.shape[:2]
    SCALE  = 8
    for mi in range(mh):
        for mj in range(mw):
            dy, dx = int(mv_map[mi, mj, 0]), int(mv_map[mi, mj, 1])
            cx, cy = mj * 16 + 8, mi * 16 + 8
            mag    = np.sqrt(dx**2 + dy**2)
            r, g, b = plt.cm.plasma(min(mag / max_mag, 1.0))[:3]
            color   = (int(b*255), int(g*255), int(r*255))
            if mag == 0:
                cv2.circle(vis, (cx, cy), 2, color, -1)
            else:
                ex = int(cx + dx * SCALE)
                ey = int(cy + dy * SCALE)
                cv2.arrowedLine(vis, (cx, cy), (ex, ey), color, 1, tipLength=0.4)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)



def visualise(frames_dir="frames", reconstructed_dir="reconstructed_frames",
              bin_file="output.bin", fq=4):

    for p in (frames_dir, reconstructed_dir, bin_file):
        if not os.path.exists(p):
            print(f"ERROR: '{p}' not found."); return

    orig_bgrs, _  = _load_bgrs(frames_dir)
    recon_bgrs, _ = _load_bgrs(reconstructed_dir)
    bin_frames    = _read_bin(bin_file)

    fq_actual = int(bin_frames[0].get("fq", fq))
    Q         = make_quant(fq_actual)
    p_indices = [i for i, d in enumerate(bin_frames) if str(d["frame_type"]) == "P"]
    def _avg_mag(i):
        mv = bin_frames[i]["mv_map"].astype(float)
        return np.mean(np.sqrt(mv[..., 0]**2 + mv[..., 1]**2))
    p_idx = max(p_indices, key=_avg_mag) if p_indices else None


    n = len(orig_bgrs)
    preview = [int(i * (n-1) / 4) for i in range(5)] if n >= 5 else list(range(n))

    Y0, Cb0, Cr0 = bgr_to_ycbcr420(orig_bgrs[0])
    h, w = Y0.shape
    up = lambda c: cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)

    raw_b = Y0[:8, :8].astype(np.float32)
    dct_b = cv2.dct(raw_b - 128)
    q_b   = np.round(dct_b / Q).astype(np.int16)
    rec_b = np.clip(cv2.idct(q_b.astype(np.float32) * Q) + 128, 0, 255).astype(np.uint8)

    if p_idx is not None:
        cur_bgr, ref_bgr   = orig_bgrs[p_idx], orig_bgrs[p_idx - 1]
        cur_Y, cur_Cb, cur_Cr = bgr_to_ycbcr420(cur_bgr)
        ref_Y, ref_Cb, ref_Cr = bgr_to_ycbcr420(ref_bgr)
        mv_map, pred_Y     = motion_estimate(cur_Y, ref_Y, mb_size=16, search_win=8)
        mag_map            = np.sqrt(mv_map[:,:,0].astype(float)**2 + mv_map[:,:,1].astype(float)**2)
        vmax_mag           = max(float(mag_map.max()), 1.0)
        res_Y              = cur_Y.astype(np.float32) - pred_Y.astype(np.float32)
        mx                 = max(int(np.percentile(np.abs(res_Y), 99)), 5)
        recon_p            = recon_bgrs[p_idx] if p_idx < len(recon_bgrs) else cur_bgr
        pred_Cb            = apply_mv(ref_Cb, mv_map, mb_size=16, scale=0.5)
        pred_Cr            = apply_mv(ref_Cr, mv_map, mb_size=16, scale=0.5)
        pred_rgb           = cv2.cvtColor(ycbcr420_to_bgr(pred_Y, pred_Cb, pred_Cr), cv2.COLOR_BGR2RGB)


    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 25), facecolor="#0d0d1a")

    fig.text(0.5, 0.995, "MPEG-4 Pipeline Visualisation",
             ha="center", va="top", fontsize=15, fontweight="bold",
             color="#ffffff", fontfamily="monospace")

    gs = GridSpec(5, 1, figure=fig, hspace=0.45,
                  top=0.97, bottom=0.01, left=0.02, right=0.98)

    LABEL_X  = 0.012   
    row_tops = [None] * 5  

    # Row 1: Original Frames 
    gs1 = GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0], wspace=0.08)
    for col, k in enumerate(preview):
        ax    = fig.add_subplot(gs1[0, col])
        ftype = str(bin_frames[k]["frame_type"])
        _show(ax, cv2.cvtColor(orig_bgrs[k], cv2.COLOR_BGR2RGB),
              title=f"frame {k+1:03d}  [{ftype}]", fig=fig)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor("#f0c060" if ftype == "I" else "#60c0f0")
            sp.set_linewidth(1.5)
        ax.set_xticks([]); ax.set_yticks([])

    # Row 2: Colour Space 
    gs2 = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1], wspace=0.15)
    cs_items = [
        (cv2.cvtColor(orig_bgrs[0], cv2.COLOR_BGR2RGB), "Original RGB",    None),
        (Y0,       "Y  (Luma)",        "inferno"),
        (up(Cb0),  "Cb (Chroma Blue)", "Blues"),
        (up(Cr0),  "Cr (Chroma Red)",  "Reds"),
    ]
    for col, (img, title, cmap) in enumerate(cs_items):
        ax = fig.add_subplot(gs2[0, col])
        _show(ax, img, title, cmap,
              vmin=(0 if cmap else None), vmax=(255 if cmap else None),
              colorbar=(cmap is not None), fig=fig)

    # Row 3: DCT 
    gs3 = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[2], wspace=0.22)
    dct_items = [
        (raw_b,             "Raw pixels",           "viridis",  0,    255),
        (dct_b,             "DCT coefficients",     "magma",    None, None),
        (q_b.astype(float), "Quantised coeffs",     "coolwarm", None, None),
        (rec_b,             "Reconstructed (IDCT)", "viridis",  0,    255),
    ]
    for col, (data, title, cmap, vmin, vmax) in enumerate(dct_items):
        ax = fig.add_subplot(gs3[0, col])
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="equal", interpolation="nearest")
        ax.set_title(title, fontsize=7, color="#ddd", pad=3)
        for x in range(9):
            ax.axvline(x - .5, color="#ffffff22", lw=0.5)
            ax.axhline(x - .5, color="#ffffff22", lw=0.5)
        for r in range(8):
            for c in range(8):
                ax.text(c, r, f"{int(round(data[r, c]))}", ha="center", va="center",
                        fontsize=4.5, color="white", fontweight="bold")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=5, colors="#aaa")
        ax.set_xticks([]); ax.set_yticks([])

    # Row 4: Motion Vectors 
    gs4 = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[3], wspace=0.12)
    if p_idx is None:
        ax = fig.add_subplot(gs4[0, :])
        ax.text(0.5, 0.5, "No P-frames (GOP=1) — all frames are I-frames",
                ha="center", va="center", fontsize=11, color="#f0c060",
                transform=ax.transAxes)
        ax.axis("off")
    else:
        ax1 = fig.add_subplot(gs4[0, 0])
        _show(ax1, cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB),
              title=f"Reference frame {p_idx:03d}", fig=fig)

        ax2 = fig.add_subplot(gs4[0, 1])
        _show(ax2, cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2RGB),
              title=f"Current frame {p_idx+1:03d}", fig=fig)

        ax3 = fig.add_subplot(gs4[0, 2])
        ax3.imshow(_mv_overlay(cur_Y, mv_map, vmax_mag), aspect="auto", interpolation="nearest")
        ax3.set_title("MV Field  (arrows ×8, circles=zero)", fontsize=7, color="#ddd", pad=3)
        ax3.axis("off")
        sm = plt.cm.ScalarMappable(cmap="plasma",
                                    norm=matplotlib.colors.Normalize(0, vmax_mag))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax3, fraction=0.035, pad=0.02)
        cb.set_label("MV magnitude (px)", fontsize=6, color="#aaa")
        cb.ax.tick_params(labelsize=5, colors="#aaa")

        ax4 = fig.add_subplot(gs4[0, 3])
        im4 = ax4.imshow(mag_map, cmap="hot", vmin=0, vmax=vmax_mag,
                         aspect="auto", interpolation="nearest")
        ax4.set_title("MV Magnitude Heatmap", fontsize=7, color="#ddd", pad=3)
        ax4.axis("off")
        cb4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.03)
        cb4.set_label("magnitude (px)", fontsize=6, color="#aaa")
        cb4.ax.tick_params(labelsize=5, colors="#aaa")

    # Row 5: Residuals 
    gs5 = GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[4], wspace=0.18)
    if p_idx is None:
        ax = fig.add_subplot(gs5[0, :])
        ax.text(0.5, 0.5, "No P-frames (GOP=1) — residual section not available",
                ha="center", va="center", fontsize=11, color="#f0c060",
                transform=ax.transAxes)
        ax.axis("off")
    else:
        res_items = [
            (res_Y,                                        "Residual Y (signed)",   "seismic", -mx,  mx,  True),
            (np.abs(res_Y),                                "Residual Y (abs)",      "hot",      0,   mx,  True),
            (pred_rgb,                                     "MC Prediction (color)", None,       None,None, False),
            (cv2.cvtColor(recon_p, cv2.COLOR_BGR2RGB),    "Reconstructed P-frame", None,       None,None, False),
            (cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2RGB),    "Original P-frame",      None,       None,None, False),
        ]
        for col, (data, title, cmap, vmin, vmax, cb) in enumerate(res_items):
            ax = fig.add_subplot(gs5[0, col])
            _show(ax, data, title, cmap, vmin, vmax, colorbar=cb, fig=fig)

    row_labels = [
        "Stage 1 — Original Frames",
        "Stage 2 — Colour Space",
        f"Stage 3 — DCT & Quantisation  (fq={fq_actual})",
        f"Stage 4 — Motion Vectors  (P-frame {p_idx+1:03d})" if p_idx is not None else "Stage 4 — Motion Vectors",
        f"Stage 5 — Residuals & Reconstruction  (P-frame {p_idx+1:03d})" if p_idx is not None else "Stage 5 — Residuals",
    ]
    for row, label in enumerate(row_labels):
        ss   = gs[row].get_position(fig)
        y    = ss.y1 + 0.012
        fig.text(LABEL_X, y, label, ha="left", va="bottom",
                 fontsize=8, fontweight="bold", color="#f0c060",
                 fontfamily="monospace")
        line = matplotlib.lines.Line2D(
            [0.01, 0.99], [y - 0.002, y - 0.002],
            transform=fig.transFigure, color="#333355", linewidth=0.8)
        fig.add_artist(line)

    out_path = os.path.join(os.path.dirname(bin_file), "pipeline_visualisation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nPipeline figure saved to '{out_path}'")