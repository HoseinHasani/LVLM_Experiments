import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style('darkgrid')
sns.set_palette('bright')

# Load data
pos = np.load("../data/pos.npy")
y = np.load("../data/y.npy")
f = np.load("../data/f.npy")  # 1 = first occurrence, 0 = non-first

def smooth(x, kernel=[0.2, 0.6, 0.2], k=10):
    x = x.copy()
    for _ in range(k):
        for j in range(1, len(x) - 1):
            x[j] = kernel[0] * x[j-1] + kernel[1] * x[j] + kernel[2] * x[j+1]
    return x

def pr_given_y_t(pos, y, f, cls, bins):
    centers = 0.5 * (bins[:-1] + bins[1:])
    probs = np.zeros(len(centers))

    for i in range(len(bins) - 1):
        mask = (
            (pos >= bins[i]) &
            (pos < bins[i + 1]) &
            (y == cls)
        )
        if mask.sum() > 0:
            probs[i] = f[mask].mean()
        else:
            probs[i] = 0.0

    probs = smooth(probs)
    return centers, probs

# Binning
bins = np.arange(pos.min(), pos.max() + 2) - 0.5

x_c, p_c = pr_given_y_t(pos, y, f, cls=1, bins=bins)
x_h, p_h = pr_given_y_t(pos, y, f, cls=0, bins=bins)

# Output path
out_dir = "figs"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "pr1_given_y_t.pdf")

# Plot
plt.figure(figsize=(6.5, 4))
plt.plot(x_c, p_c, label="Correct", color="blue", linewidth=2.5)
plt.plot(x_h, p_h, label="Hallucinated", color="red", linewidth=2.5)

plt.xlabel("Token Position", fontsize=15)
plt.ylabel(r"$p(o = \text{first} \mid y, t)$", fontsize=15)
plt.title(
    "First Occurrence Probability",
    fontsize=14,
    fontweight="bold"
)

plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(out_path, format="pdf", bbox_inches="tight")
