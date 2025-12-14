import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from scipy.stats import sem
import seaborn as sns

# =========================
# Configuration
# =========================

data_dir = "../data/scores_1"                 # directory containing scores_*.pkl
file_pattern = "scores_*.pkl"
n_files = 4000

topk_entropy = 10                    # top-K for entropy
avg_win_size = 3
stride_size = 1

x_min, x_max = 0, 160

sns.set(style="darkgrid")
eps = 1e-12

# =========================
# Math utilities
# =========================

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x) + eps)

def entropy_topk(logits, k=10):
    """
    Entropy of top-K softmax probabilities.
    """
    probs = softmax(logits)
    topk = np.sort(probs)[-k:]
    topk = topk / (np.sum(topk) + eps)
    return -np.sum(topk * np.log(topk + eps))

# =========================
# Data extraction
# =========================

def extract_logit_metrics(data_dict, cls_, topk=10):
    """
    Extract (position, entropy, max_logit, max_softmax)
    """
    results = []
    entries = data_dict.get(cls_, [])

    for entry in entries:
        if len(entry) != 2:
            continue

        pos, logits = entry
        logits = np.asarray(logits).squeeze()

        if logits.ndim != 1:
            continue

        ent = entropy_topk(logits, k=topk)
        max_logit = float(np.max(logits))
        max_softmax = float(np.max(softmax(logits)))

        results.append((int(pos), ent, max_logit, max_softmax))

    return results


def aggregate_scores_across_files(files, n_files, topk=10):
    tp, fp, oth = [], [], []

    for f in tqdm(files[:n_files], desc="Reading score files"):
        try:
            with open(f, "rb") as handle:
                data_dict = pickle.load(handle)
        except Exception:
            continue

        tp.extend(extract_logit_metrics(data_dict, "tp", topk))
        fp.extend(extract_logit_metrics(data_dict, "fp", topk))
        oth.extend(extract_logit_metrics(data_dict, "other", topk))

    return tp, fp, oth

# =========================
# Aggregation by position
# =========================

def build_posmap(metric_data, metric_idx):
    """
    metric_idx:
      1 -> entropy
      2 -> max_logit
      3 -> max_softmax
    """
    posmap = {}
    for item in metric_data:
        pos = item[0]
        val = item[metric_idx]
        posmap.setdefault(pos, []).append(val)
    return posmap

# =========================
# Smoothing + CI
# =========================

def smooth_with_ci_from_posmap(pos_map, win=3, stride=1):
    if not pos_map:
        return np.array([]), np.array([]), np.array([])

    positions = sorted(pos_map.keys())
    x_arr = np.array(positions)

    xs, ys, cis = [], [], []

    for start in range(0, len(x_arr) - win + 1, stride):
        window_x = x_arr[start:start + win]
        vals = []

        for p in window_x:
            vals.extend(pos_map.get(p, []))

        if not vals:
            continue

        xs.append(np.mean(window_x))
        ys.append(np.mean(vals))
        cis.append(1.96 * sem(vals) if len(vals) > 1 else 0.0)

    return np.array(xs), np.array(ys), np.array(cis)

# =========================
# Plotting
# =========================

def plot_metric(tp_map, fp_map, oth_map,
                ylabel, title, savepath):

    tp_x, tp_y, tp_ci = smooth_with_ci_from_posmap(tp_map, avg_win_size, stride_size)
    fp_x, fp_y, fp_ci = smooth_with_ci_from_posmap(fp_map, avg_win_size, stride_size)
    oth_x, oth_y, oth_ci = smooth_with_ci_from_posmap(oth_map, avg_win_size, stride_size)

    plt.figure(figsize=(10, 6))

    if tp_x.size > 0:
        plt.plot(tp_x, tp_y, color="tab:green", linewidth=2, label="TP")
        plt.fill_between(tp_x, tp_y - tp_ci, tp_y + tp_ci,
                         color="tab:green", alpha=0.2)

    if fp_x.size > 0:
        plt.plot(fp_x, fp_y, color="tab:red", linewidth=2, label="FP")
        plt.fill_between(fp_x, fp_y - fp_ci, fp_y + fp_ci,
                         color="tab:red", alpha=0.2)

    if oth_x.size > 0:
        plt.plot(oth_x, oth_y, color="tab:gray", linewidth=2,
                 alpha=0.6, label="Others")

    plt.xlabel("Token Position (Generated Text)", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.xlim(x_min, x_max)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(savepath, dpi=140)
    plt.show()

# =========================
# Main
# =========================

if __name__ == "__main__":

    files = sorted(glob(os.path.join(data_dir, file_pattern)))
    print(f"Found {len(files)} score files")

    tp_scores, fp_scores, oth_scores = aggregate_scores_across_files(
        files, n_files=n_files, topk=topk_entropy
    )

    # ---- Entropy ----
    tp_ent = build_posmap(tp_scores, 1)
    fp_ent = build_posmap(fp_scores, 1)
    oth_ent = build_posmap(oth_scores, 1)

    plot_metric(
        tp_ent, fp_ent, oth_ent,
        ylabel="Entropy (Top-10 Softmax)",
        title="Entropy of Token Predictions (TP vs FP)",
        savepath="score_entropy_tp_fp.png"
    )

    # ---- Max Logit ----
    tp_ml = build_posmap(tp_scores, 2)
    fp_ml = build_posmap(fp_scores, 2)
    oth_ml = build_posmap(oth_scores, 2)

    plot_metric(
        tp_ml, fp_ml, oth_ml,
        ylabel="Max Logit",
        title="Maximum Logit per Token (TP vs FP)",
        savepath="score_max_logit_tp_fp.png"
    )

    # ---- Max Softmax ----
    tp_ms = build_posmap(tp_scores, 3)
    fp_ms = build_posmap(fp_scores, 3)
    oth_ms = build_posmap(oth_scores, 3)

    plot_metric(
        tp_ms, fp_ms, oth_ms,
        ylabel="Max Softmax Probability",
        title="Maximum Softmax Probability per Token (TP vs FP)",
        savepath="score_max_softmax_tp_fp.png"
    )
