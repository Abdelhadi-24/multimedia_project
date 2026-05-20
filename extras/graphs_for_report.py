import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


fq_values     = [1, 2, 4, 8, 16]
global_ratios = [8.78, 13.15, 20.10, 32.14, 47.19]
iframe_ratios = [4.26, 5.96, 8.22, 11.54, 16.44]

gop_values = [1, 2, 4, 8, 16]
gop_ratios = [8.24, 12.16, 16.55, 20.10, 22.97]


def style_ax(ax):
    ax.grid(color="#dddddd", linewidth=0.7, linestyle="--")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")


fig1, ax1 = plt.subplots(figsize=(7, 5))
fig1.patch.set_facecolor("white")
style_ax(ax1)

ax1.plot(fq_values, global_ratios, marker="o", color="#1a5fa8", linewidth=2,
         markersize=7, label="Global ratio")
ax1.fill_between(fq_values, global_ratios, alpha=0.07, color="#1a5fa8")

ax1.plot(fq_values, iframe_ratios, marker="s", color="#c0392b", linewidth=2,
         markersize=7, linestyle="--", label="Avg I-frame ratio")
ax1.fill_between(fq_values, iframe_ratios, alpha=0.07, color="#c0392b")

ax1.set_xlabel("Quantisation factor (fq)", fontsize=11)
ax1.set_ylabel("Compression ratio (×)", fontsize=11)
ax1.set_title("Compression ratio vs Quantisation factor\n(GOP=8, S=8)", fontsize=11)
ax1.legend(fontsize=10, framealpha=0.9)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}×"))
ax1.set_xticks(fq_values)

for x, y in zip(fq_values, global_ratios):
    ax1.annotate(f"{y}×", (x, y), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=8, color="#1a5fa8")
for x, y in zip(fq_values, iframe_ratios):
    ax1.annotate(f"{y}×", (x, y), textcoords="offset points",
                 xytext=(0, -15), ha="center", fontsize=8, color="#c0392b")

plt.tight_layout()
plt.savefig("graph_fq.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved → graph_fq.png")


fig2, ax2 = plt.subplots(figsize=(7, 5))
fig2.patch.set_facecolor("white")
style_ax(ax2)

ax2.plot(gop_values, gop_ratios, marker="o", color="#1a5fa8", linewidth=2,
         markersize=7, label="Global ratio")
ax2.fill_between(gop_values, gop_ratios, alpha=0.07, color="#1a5fa8")

ax2.set_xlabel("GOP size (G)", fontsize=11)
ax2.set_ylabel("Compression ratio (×)", fontsize=11)
ax2.set_title("Compression ratio vs GOP size\n(fq=4, S=8)", fontsize=11)
ax2.legend(fontsize=10, framealpha=0.9)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}×"))
ax2.set_xticks(gop_values)

for x, y in zip(gop_values, gop_ratios):
    ax2.annotate(f"{y}×", (x, y), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=8, color="#1a5fa8")

plt.tight_layout()
plt.savefig("graph_gop.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved → graph_gop.png")