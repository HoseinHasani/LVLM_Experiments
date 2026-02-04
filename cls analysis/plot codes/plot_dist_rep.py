import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style('darkgrid')
sns.set_palette('bright')

f = np.load("../data/f.npy")
# Load data
y = np.load("../data/y.npy")[np.where(f)]
rep = np.load("../data/repeats.npy")[np.where(f)]

# Repetition values
rep_values = np.array([1, 2, 3, 4])

# Compute p(rep | y)
p_correct = []
p_hall = []

for r in rep_values:
    mask_r = (rep == r)

    # Correct class
    if (mask_r & (y == 1)).sum() > 0:
        p_correct.append(
            (mask_r & (y == 1)).sum() / (y == 1).sum()
        )
    else:
        p_correct.append(0.0)

    # Hallucinated class
    if (mask_r & (y == 0)).sum() > 0:
        p_hall.append(
            (mask_r & (y == 0)).sum() / (y == 0).sum()
        )
    else:
        p_hall.append(0.0)

p_correct = np.array(p_correct)
p_hall = np.array(p_hall)

# Output path
out_dir = "figs"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "prep_given_y.pdf")

# Plot
plt.figure(figsize=(5.4, 3.6))

bar_width = 0.3
x = np.arange(len(rep_values))

plt.bar(
    x - bar_width / 2,
    p_correct,
    width=bar_width,
    color="tab:blue",
    label="Correct"
)

plt.bar(
    x + bar_width / 2,
    p_hall,
    width=bar_width,
    color="tab:red",
    label="Hallucinated"
)

plt.xticks(
    x,
    [f"{r}" for r in rep_values],
    fontsize=12
)

plt.xlabel("Repetition", fontsize=13)
plt.ylabel(r"$p(\mathrm{r} \mid y)$", fontsize=13)
plt.title(
    "Object Repetition Distribution",
    fontsize=14,
    fontweight="bold"
)

plt.legend(fontsize=12, framealpha=1.0, facecolor="white")
plt.tight_layout()
plt.savefig(out_path, format="pdf", bbox_inches="tight")
