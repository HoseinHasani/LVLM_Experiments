import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
data_dir = "../data"       
save_dir = "figs"
os.makedirs(save_dir, exist_ok=True)

n_layers = 32
n_heads = 32

attn_choice = "vals1"    
start_layer = 5
end_layer = 18

show_std = False          

min_position = None
max_position = None


X = np.load(os.path.join(data_dir, "x.npy"))
y = np.load(os.path.join(data_dir, "y.npy"))     
rep = np.load(os.path.join(data_dir, "f.npy"))
pos = np.load(os.path.join(data_dir, "pos.npy"))


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


assert 0 <= start_layer < end_layer <= n_layers
attn = attn[:, start_layer:end_layer, :]

attn_avg = attn.mean(axis=(1, 2))

if min_position is None:
    min_position = int(pos.min())
if max_position is None:
    max_position = int(pos.max())

positions = np.arange(min_position, max_position + 1)

def smooth(x, kernel=[0.2, 0.6, 0.2], k=8):
    x = x.copy()
    for _ in range(k):
        for j in range(1, len(x) - 1):
            x[j] = kernel[0] * x[j-1] + kernel[1] * x[j] + kernel[2] * x[j+1]
    return x

def fill_nan_nearest(x):
    x = x.copy()
    n = len(x)
    idx = np.arange(n)

    valid = ~np.isnan(x)
    if valid.sum() == 0:
        return x  # all NaN, nothing to do

    x[~valid] = np.interp(
        idx[~valid],
        idx[valid],
        x[valid],
    )
    return x


def conditional_mean_std(values, pos, mask):
    means, stds = [], []
    for t in positions:
        vals = values[(pos == t) & mask]
        if len(vals) == 0:
            means.append(np.nan)
            stds.append(np.nan)
        else:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
    return np.array(means), np.array(stds)


def conditional_mean_std(values, pos, mask):
    means, stds = [], []
    for t in positions:
        vals = values[(pos == t) & mask]
        if len(vals) == 0:
            means.append(np.nan)
            stds.append(np.nan)
        else:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
    return np.array(means), np.array(stds)


tp = (y == 1)
fp = (y == 0)

first = (rep == 1)
nonfirst = (rep == 0)

mean_tp_first, _ = conditional_mean_std(attn_avg, pos, tp & first)
mean_tp_nonfirst, _ = conditional_mean_std(attn_avg, pos, tp & nonfirst)

mean_fp_first, _ = conditional_mean_std(attn_avg, pos, fp & first)
mean_fp_nonfirst, _ = conditional_mean_std(attn_avg, pos, fp & nonfirst)

# Fill empty positions
mean_tp_first = fill_nan_nearest(mean_tp_first)
mean_tp_nonfirst = fill_nan_nearest(mean_tp_nonfirst)
mean_fp_first = fill_nan_nearest(mean_fp_first)
mean_fp_nonfirst = fill_nan_nearest(mean_fp_nonfirst)

# Smooth AFTER filling
mean_tp_first = smooth(mean_tp_first)
mean_tp_nonfirst = smooth(mean_tp_nonfirst)
mean_fp_first = smooth(mean_fp_first)
mean_fp_nonfirst = smooth(mean_fp_nonfirst)




plt.figure(figsize=(8, 4))

# Correct (blue)
# Correct (blue)
plt.plot(
    positions,
    mean_tp_first,
    color="blue",
    linewidth=2.5,
)
plt.plot(
    positions,
    mean_tp_nonfirst,
    color="blue",
    linewidth=2.5,
    linestyle="--",
)

# Hallucinated (red)
plt.plot(
    positions,
    mean_fp_first,
    color="red",
    linewidth=2.5,
)
plt.plot(
    positions,
    mean_fp_nonfirst,
    color="red",
    linewidth=2.5,
    linestyle="--",
)


from matplotlib.lines import Line2D

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

rep_handles = [
    Line2D([0], [0], color="black", lw=2.5, linestyle="-", label="First"),
    Line2D([0], [0], color="black", lw=2.5, linestyle="--", label="Non-first"),
]

plt.legend(
    handles=rep_handles,
    loc="upper right",
    fontsize=13,
    frameon=True,
)



plt.xlabel("Token Position", fontsize=13)
plt.ylabel("Avg. Image Attention", fontsize=13)

plt.title(
    "Average Image Attention of First and Non-first Occurrences",
    fontsize=13,
    fontweight="bold",
)

plt.tight_layout()

save_path = os.path.join(
    save_dir,
    f"avg_attn_vs_pos_first.pdf"
)
plt.savefig(save_path)

print(f"Saved figure to {save_path}")


