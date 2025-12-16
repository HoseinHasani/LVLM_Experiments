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

target_rep = 1

sfx = ''

data_dir = "../data/double_scores_4"                 # directory containing scores_*.pkl

files = sorted(glob(os.path.join(data_dir, "scores_*.pkl")))

files_dict = {
    int(os.path.splitext(os.path.basename(f))[0].split('_')[1]): f
    for f in files
}



attn_data_dir = "../data/all layers all attention tp fp rep double"
attn_files = glob(os.path.join(attn_data_dir, "attentions_*.pkl"))

attn_files_dict = {
    int(os.path.splitext(os.path.basename(f))[0].split('_')[1]): f
    for f in attn_files
}

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
    logits = logits.astype(np.float32)

    topk_logits = np.partition(logits, -k)[-k:]
    topk_logits -= np.max(topk_logits)

    probs = np.exp(topk_logits)
    probs /= probs.sum() + 1e-12

    return -np.sum(probs * np.log(probs + 1e-12))

# =========================
# Data extraction
# =========================

def extract_logit_metrics(data_dict, cls_, topk=10, a_inds=[], a_reps=[]):
    results = []
    entries = data_dict.get(cls_, [])

    for k in range(len(entries)):
        entry = entries[k]
        pos = entry[0]
        logits = entry[1]
        
        a_pos = a_inds[k]
        
        if a_pos is None:
            continue
        
        if int(pos) != int(a_pos):
            # print('pos != a_pos', cls_, pos, a_pos)
            continue
            
            
        rep = a_reps[k]
        if rep != None:
            if rep != target_rep:
                continue
        
        logits = np.asarray(logits).squeeze()
        if logits.ndim != 1:
            continue

        logits = logits.astype(np.float32)

        ent = entropy_topk(logits, k=topk)
        max_logit = float(np.max(logits))
        max_softmax = float(np.max(softmax(logits)))

        if not np.isfinite(ent):
            continue
        if not np.isfinite(max_logit) or not np.isfinite(max_softmax):
            continue

        results.append((int(pos), ent, max_logit, max_softmax))
        
    return results



def aggregate_scores_across_files(files_dict, attn_files_dict, n_files, topk=10):
    tp, fp, oth = [], [], []

    file_names = list(files_dict.keys())
    attn_names = list(attn_files_dict.keys())
    
    
    intersect = list(set(file_names).intersection(set(attn_names)))
    
    print(f"Found {len(intersect)} score files")

    
    for j in tqdm(range(len(intersect)), desc="Reading score files"):
        file_id = intersect[j]
        f = files_dict[file_id]
        at = attn_files_dict[file_id]
        
        try:
            with open(f, "rb") as handle:
                data_dict = pickle.load(handle)
        except Exception:
            continue

        try:
            with open(at, "rb") as handle:
                at_data_dict = pickle.load(handle)
        except Exception:
            continue
        
        attn_tp_indices = []
        attn_tp_reps = []
        for k in range(len(at_data_dict['tp']['image'])):
            if len(at_data_dict['tp']['image'][k]['token_indices']) < 1:
                attn_tp_indices.append(None)
                attn_tp_reps.append(-10)
            else:
                attn_tp_indices.append(at_data_dict['tp']['image'][k]['token_indices'][0])
                attn_tp_reps.append(at_data_dict['tp']['image'][k]['rep_num'])

        attn_fp_indices = []
        attn_fp_reps = []
        for k in range(len(at_data_dict['fp']['image'])):
            if len(at_data_dict['fp']['image'][k]['token_indices']) < 1:
                attn_fp_indices.append(None)
                attn_fp_reps.append(-10)
            else:
                attn_fp_indices.append(at_data_dict['fp']['image'][k]['token_indices'][0])
                attn_fp_reps.append(at_data_dict['fp']['image'][k]['rep_num'])


        attn_oth_indices = []
        attn_oth_reps = []
        for k in range(len(at_data_dict['other']['image'])):
            if len(at_data_dict['other']['image'][k]['token_indices']) < 1:
                attn_oth_indices.append(None)
                attn_oth_reps.append(-10)
            else:
                attn_oth_indices.append(at_data_dict['other']['image'][k]['token_indices'][0])
                attn_oth_reps.append(at_data_dict['other']['image'][k]['rep_num'])

            
      
        tp.extend(extract_logit_metrics(data_dict, f"tp{sfx}", topk, attn_tp_indices, attn_tp_reps))
        fp.extend(extract_logit_metrics(data_dict, f"fp{sfx}", topk, attn_fp_indices, attn_fp_reps))
        oth.extend(extract_logit_metrics(data_dict, f"other{sfx}", topk, attn_oth_indices, attn_oth_reps))

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


tp_scores, fp_scores, oth_scores = aggregate_scores_across_files(
    files_dict, attn_files_dict, n_files=n_files, topk=topk_entropy
)

# ---- Entropy ----
tp_ent = build_posmap(tp_scores, 1)
fp_ent = build_posmap(fp_scores, 1)
oth_ent = build_posmap(oth_scores, 1)

plot_metric(
    tp_ent, fp_ent, oth_ent,
    ylabel="Entropy (Top-10 Softmax)",
    title="Entropy of Token Predictions (TP vs FP)",
    savepath=f"score_entropy_tp_fp{sfx}_{target_rep}.png"
)

# ---- Max Logit ----
tp_ml = build_posmap(tp_scores, 2)
fp_ml = build_posmap(fp_scores, 2)
oth_ml = build_posmap(oth_scores, 2)

plot_metric(
    tp_ml, fp_ml, oth_ml,
    ylabel="Max Logit",
    title="Maximum Logit per Token (TP vs FP)",
    savepath=f"score_max_logit_tp_fp{sfx}_{target_rep}.png"
)

# ---- Max Softmax ----
tp_ms = build_posmap(tp_scores, 3)
fp_ms = build_posmap(fp_scores, 3)
oth_ms = build_posmap(oth_scores, 3)

plot_metric(
    tp_ms, fp_ms, oth_ms,
    ylabel="Max Softmax Probability",
    title="Maximum Softmax Probability per Token (TP vs FP)",
    savepath=f"score_max_softmax_tp_fp{sfx}_{target_rep}.png"
)
