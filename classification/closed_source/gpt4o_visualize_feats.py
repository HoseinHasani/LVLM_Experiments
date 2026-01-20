import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
from scipy.stats import gaussian_kde

def load_prior_metadata(
    metadata_dir,
    min_position,
    max_position,
    max_repetition=3,
):
    X = []
    y = []

    meta_files = [
        os.path.join(metadata_dir, f)
        for f in os.listdir(metadata_dir)
        if f.endswith(".json")
    ]

    for mf in tqdm(meta_files, desc="Loading metadata"):
        with open(mf, "r") as f:
            data = json.load(f)

        for w in data["words_meta"]:
            if w['is_first_occ']:
                cls = w["type"]
                if cls not in {"tp", "fp"}:
                    continue
    
                # repetition r ∈ {0..3}
                r = min(w["total_reps"] - 1, max_repetition)
    
                # position t
                t = w["position"]
                if t < min_position or t > max_position:
                    continue
    
                # normalize
                r_norm = r / max_repetition
                t_norm = (t - min_position) / (max_position - min_position)
    
                X.append([r_norm, t_norm])
                y.append(1 if cls == "tp" else 0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y


METADATA_DIR = "gpt4o_meta_data_greedy"

X, y = load_prior_metadata(
    metadata_dir=METADATA_DIR,
    min_position=5,
    max_position=155,
)

X_int = 3 * X[:, 0] + 1
X[:, 0] = np.round(X_int).astype(int) # {0, 1, 2, 3}

# Split by class
X0 = X[y == 0]
X1 = X[y == 1]


out_dir = "./plots/"
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# 1. Histogram for Feature 0 (DISCRETE, 4 VALUES) — SHIFTED BARS
# ============================================================

feature0_values = np.sort(np.unique(X[:, 0]))
bar_width = 0.1

counts0 = np.array([np.sum(X0[:, 0] == v) for v in feature0_values])
counts1 = np.array([np.sum(X1[:, 0] == v) for v in feature0_values])

counts0 = counts0 / counts0.sum()
counts1 = counts1 / counts1.sum()

plt.figure(figsize=(6, 4))

plt.bar(
    feature0_values - bar_width / 2,
    counts0,
    width=bar_width,
    label="Class 0",
)

plt.bar(
    feature0_values + bar_width / 2,
    counts1,
    width=bar_width,
    label="Class 1",
)

plt.xlabel("Feature 0 (Discrete Values)")
plt.ylabel("Proportion")
plt.title("Histogram of Feature 0 by Class")
plt.xticks(feature0_values)
plt.legend()
plt.tight_layout()

plt.savefig(
    out_dir + "feature0_histogram.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(f'{out_dir}/feat0.png', dpi=200)

# ============================================================
# 2. Histogram for Feature 1 (CONTINUOUS)
# ============================================================

plt.figure(figsize=(6, 4))

plt.hist(
    X0[:, 1],
    bins=30,
    alpha=0.6,
    label="Class 0",
    density=True,
)

plt.hist(
    X1[:, 1],
    bins=30,
    alpha=0.6,
    label="Class 1",
    density=True,
)

plt.xlabel("Feature 1 (Continuous)")
plt.ylabel("Density")
plt.title("Histogram of Feature 1 by Class")
plt.legend()
plt.tight_layout()

plt.savefig(
    out_dir + "feature1_histogram.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(f'{out_dir}/feat1.png', dpi=200)

# ============================================================
# 3. SCATTER PLOT — JOINT FEATURES (WITH JITTER ON FEATURE 0)
# ============================================================

# Small jitter for discrete x-axis to avoid overplotting
jitter_strength = 0.04
x0_jitter = X[:, 0] + np.random.uniform(
    -jitter_strength, jitter_strength, size=len(X)
)

plt.figure(figsize=(6, 5))

plt.scatter(
    x0_jitter[y == 0],
    X[y == 0, 1],
    alpha=0.5,
    s=20,
    label="Class 0",
)

plt.scatter(
    x0_jitter[y == 1],
    X[y == 1, 1],
    alpha=0.2,
    s=20,
    label="Class 1",
)

plt.xlabel("Feature 0 (Discrete, Jittered)")
plt.ylabel("Feature 1")
plt.title("Joint Feature Scatter Plot")
plt.xticks(feature0_values)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(f"{out_dir}/joint.png", dpi=200)


# ============================================================
# 4. CONDITIONAL KDE: p(feature1 | feature0 = k)
# ============================================================

feature0_values = np.sort(np.unique(X[:, 0]))
n_rows = len(feature0_values)

x_grid = np.linspace(0.0, 1.0, 400)

fig, axes = plt.subplots(
    n_rows,
    1,
    figsize=(6, 2.5 * n_rows),
    sharex=True,
)

# If only one row, axes is not a list
if n_rows == 1:
    axes = [axes]

for ax, f0 in zip(axes, feature0_values):

    # Select samples for this discrete value
    mask0 = (X[:, 0] == f0) & (y == 0)
    mask1 = (X[:, 0] == f0) & (y == 1)

    print(f0, np.sum(mask0))
    
    x0_vals = X[mask0, 1]
    x1_vals = X[mask1, 1]

    # KDE (only if enough samples)
    if len(x0_vals) > 5:
        kde0 = gaussian_kde(x0_vals, bw_method="scott")
        ax.plot(x_grid, kde0(x_grid), label="Class 0", linewidth=2)

    if len(x1_vals) > 5:
        kde1 = gaussian_kde(x1_vals, bw_method="scott")
        ax.plot(x_grid, kde1(x_grid), label="Class 1", linewidth=2)

    ax.set_ylabel("Density")
    ax.set_title(f"#Rep = {int(f0)}")

    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Position (normalized to 1)")

axes[0].legend(loc="upper right")

plt.tight_layout()
plt.savefig(out_dir + "conditional_kde_feature1_given_feature0.png", dpi=200)
plt.show()



mask0 = y == 0
mask1 = y == 1

x0_vals = X[mask0, 1]
x1_vals = X[mask1, 1]

plt.figure(figsize=(6, 3))


if len(x0_vals) > 5:
    kde0 = gaussian_kde(x0_vals, bw_method="scott")
    plt.plot(x_grid, kde0(x_grid), label="Class 0", linewidth=2)

if len(x1_vals) > 5:
    kde1 = gaussian_kde(x1_vals, bw_method="scott")
    plt.plot(x_grid, kde1(x_grid), label="Class 1", linewidth=2)

plt.ylabel("Density")
plt.title("#Rep = {1, 2, 3, 4}")

plt.grid(alpha=0.3)

plt.xlabel("Position (normalized to 1)")

plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig(out_dir + "conditional_kde_feature1_all.png", dpi=200)
plt.show()

