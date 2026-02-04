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
end_layer = 17

show_std = False          

min_position = None
max_position = None


X = np.load(os.path.join(data_dir, "x.npy"))
y = np.load(os.path.join(data_dir, "y.npy"))     
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

mean_tp, std_tp = conditional_mean_std(attn_avg, pos, y == 1)
mean_fp, std_fp = conditional_mean_std(attn_avg, pos, y == 0)

exp_tp = attn_avg[y == 1].mean()
exp_fp = attn_avg[y == 0].mean()


mean_tp = smooth(mean_tp)
mean_fp = smooth(mean_fp)

plt.figure(figsize=(8, 4))

plt.plot(
    positions,
    mean_tp,
    label="Correct",
    linewidth=2.5,
    color="blue"
)

plt.plot(
    positions,
    mean_fp,
    label="Hallucinated",
    linewidth=2.5,
    color="red"
)

plt.axhline(
    exp_tp*1.005,
    linestyle="--",
    linewidth=2.6,
    label="Correct(Marginal Expectation)",
    color="blue",
    alpha=0.7,
)

plt.axhline(
    exp_fp*0.995,
    linestyle="--",
    linewidth=2.6,
    label="Hallucinated(Marginal Expectation)",
    color="red",
    alpha=0.7,
)


if show_std:
    plt.fill_between(
        positions,
        mean_tp - std_tp,
        mean_tp + std_tp,
        alpha=0.25,
    )
    plt.fill_between(
        positions,
        mean_fp - std_fp,
        mean_fp + std_fp,
        alpha=0.25,
    )

plt.xlabel("Token Position", fontsize=13)
plt.ylabel("Avg. Image Attention", fontsize=13)

plt.title(
    "Averaged Image Attention Across Token Position\n",
    fontweight="bold",
    fontsize=13,
)

plt.legend(fontsize=13)
plt.tight_layout()

save_path = os.path.join(
    save_dir,
    "avg_attn_vs_pos.pdf"
    # f"avg_attn_vs_pos_{attn_choice}_L{start_layer}-{end_layer-1}.pdf"
)

plt.savefig(save_path)

print(f"Saved figure to {save_path}")
