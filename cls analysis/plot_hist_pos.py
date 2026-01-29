import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Styling
sns.set_style('darkgrid')
sns.set_palette('bright')
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
})

# Load data
pos = np.load("data/pos.npy")
y = np.load("data/y.npy")

# Split by class
pos_correct = pos[y == 1]
pos_hall = pos[y == 0]

# Define bins (adjust if needed)
bins = np.arange(pos.min(), pos.max() + 2, 4) - 0.5

plt.figure(figsize=(6, 4))

plt.hist(
    pos_correct,
    bins=bins,
    density=True,
    alpha=0.6,
    label="Correct (y=1)",
    color="green"
)

plt.hist(
    pos_hall,
    bins=bins,
    density=True,
    alpha=0.6,
    label="Hallucinated (y=0)",
    color="red"
)

plt.xlabel("Token position $t$")
plt.ylabel(r"$p(t \mid y)$")
plt.legend()
plt.tight_layout()
plt.show()
