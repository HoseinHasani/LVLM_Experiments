import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -----------------------------
# Configuration
# -----------------------------
data_dir = "../data"
save_dir = "figs"
os.makedirs(save_dir, exist_ok=True)

n_layers = 32
n_heads = 32

attn_choice = "vals1"

K = 10
early_layers = (0, K)              # first 5 layers
late_layers  = (32-K, 32)  # last 5 layers

min_position = None
max_position = None

# -----------------------------
# Load data
# -----------------------------
X   = np.load(os.path.join(data_dir, "x.npy"))
y   = np.load(os.path.join(data_dir, "y.npy"))
rep = np.load(os.path.join(data_dir, "f.npy"))
pos = np.load(os.path.join(data_dir, "pos.npy"))

# -----------------------------
# Recover attention
# -----------------------------
ATTN_OFFSET = 1
ATTN_SIZE = 4 * n_layers * n_heads

attn_block = X[:, ATTN_OFFSET : ATTN_OFFSET + ATTN_SIZE]
attn_block = attn_block.reshape(-1, n_layers, n_heads, 4)

if attn_choice == "vals1":
    attn = attn_block[..., 0]
elif attn_choice == "vals2":
    attn = attn_block[..., 1]
else:
    raise ValueError("attn_choice must be 'vals1' or 'vals2'")

# -----------------------------
# Utilities
# -----------------------------
def smooth(x, kernel=[0.2, 0.6, 0.2], k=8):
    x = x.copy()
    for _ in range(k):
        for j in range(1, len(x) - 1):
            x[j] = kernel[0] * x[j-1] + kernel[1] * x[j] + kernel[2] * x[j+1]
    return x

def fill_nan_nearest(x):
    x = x.copy()
    idx = np.arange(len(x))
    valid = ~np.isnan(x)
    if valid.sum() == 0:
        return x
    x[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    return x

def conditional_mean(values, pos, mask, positions):
    out = []
    for t in positions:
        vals = values[(pos == t) & mask]
        out.append(np.mean(vals) if len(vals) > 0 else np.nan)
    return np.array(out)

# -----------------------------
# Masks
# -----------------------------
tp = (y == 1)
fp = (y == 0)
first = (rep == 1)

if min_position is None:
    min_position = int(pos.min())
if max_position is None:
    max_position = int(pos.max())

positions = np.arange(min_position, max_position + 1)

# -----------------------------
# Early / late layer averages
# -----------------------------
attn_early = attn[:, early_layers[0]:early_layers[1], :].mean(axis=(1, 2))
attn_late  = attn[:, late_layers[0]:late_layers[1], :].mean(axis=(1, 2))

# -----------------------------
# Conditional curves (first occurrence only)
# -----------------------------
mean_tp_early = conditional_mean(attn_early, pos, tp & first, positions)
mean_tp_late  = conditional_mean(attn_late,  pos, tp & first, positions)

mean_fp_early = conditional_mean(attn_early, pos, fp & first, positions)
mean_fp_late  = conditional_mean(attn_late,  pos, fp & first, positions)

# Fill + smooth
for name in [
    "mean_tp_early", "mean_tp_late",
    "mean_fp_early", "mean_fp_late"
]:
    locals()[name] = smooth(fill_nan_nearest(locals()[name]))

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8, 4))

# Correct (blue)
plt.plot(positions, mean_tp_late,  color="blue", lw=2.5)
plt.plot(positions, mean_tp_early, color="blue", lw=2.5, ls="--")

# Hallucinated (red)
plt.plot(positions, mean_fp_late,  color="red", lw=2.5)
plt.plot(positions, mean_fp_early, color="red", lw=2.5, ls="--")

# -----------------------------
# Legends
# -----------------------------
class_handles = [
    Line2D([0], [0], color="blue", lw=2.5, label="Correct"),
    Line2D([0], [0], color="red", lw=2.5, label="Hallucinated"),
]

legend1 = plt.legend(
    handles=class_handles,
    loc="lower left",
    fontsize=13,
    frameon=True,
)
plt.gca().add_artist(legend1)

layer_handles = [
    Line2D([0], [0], color="black", lw=2.5, ls="-",  label=f"Last {K} layers"),
    Line2D([0], [0], color="black", lw=2.5, ls="--", label=f"First {K} layers"),
]

plt.legend(
    handles=layer_handles,
    loc="center",
    fontsize=13,
    frameon=True,
)

# -----------------------------
# Labels
# -----------------------------
plt.xlabel("Token Position", fontsize=13)
plt.ylabel("Avg. Image Attention", fontsize=13)
plt.title(
    "Average Image Attention for Early vs Late Layers (First Occurrences)",
    fontsize=13,
    fontweight="bold",
)

plt.tight_layout()

save_path = os.path.join(
    save_dir,
    f"avg_attn_vs_pos_first_early_vs_late.pdf"
)
plt.savefig(save_path)

print(f"Saved figure to {save_path}")
