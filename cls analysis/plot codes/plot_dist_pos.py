import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style('darkgrid')
sns.set_palette('bright')

pos = np.load("../data/pos.npy")
y = np.load("../data/y.npy")

def smooth(x, kernel=[0.2, 0.6, 0.2], k=10):
    x = x.copy()
    for _ in range(k):
        for j in range(1, len(x) - 1):
            x[j] = kernel[0] * x[j-1] + kernel[1] * x[j] + kernel[2] * x[j+1]
    return x

def smooth_hist(values, bins):
    hist, edges = np.histogram(values, bins=bins, density=True)
    hist = smooth(hist)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist

bins = np.arange(pos.min(), pos.max() + 2) - 0.5

x_c, p_c = smooth_hist(pos[y == 1], bins)
x_h, p_h = smooth_hist(pos[y == 0], bins)

out_dir = "figs"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "pt_given_y.pdf")

plt.figure(figsize=(6.5, 4))
plt.plot(x_c, p_c, label="Correct", color="blue", linewidth=2.5)
plt.plot(x_h, p_h, label="Hallucinated", color="red", linewidth=2.5)

plt.xlabel("Token Position", fontsize=14)
plt.ylabel(r"$p(t \mid y)$", fontsize=14)
plt.title("Token Position Distribution", fontsize=14, fontweight="bold")
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(out_path, format="pdf", bbox_inches="tight")
