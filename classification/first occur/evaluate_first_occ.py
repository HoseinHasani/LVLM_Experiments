import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score
)


sns.set_style('darkgrid')
# sns.set_palette('bright')


min_position = 5
max_position = 155
position_margin = 2
use_logits = True


base_save_dir = "final_cl_results"
sfx = "" if use_logits else "_no_logits"
exp_name = f"exp__bayes_ce{sfx}"
save_dir = os.path.join(base_save_dir, exp_name)
results_dir = os.path.join(save_dir, "results")

y_true1 = np.load(f"{results_dir}/y_true_1.npy")
y_pred1 = np.load(f"{results_dir}/y_pred_1.npy")
y_probs1 = np.load(f"{results_dir}/y_probs_1.npy")
cls_test1 = np.load(f"{results_dir}/cls_test_1.npy")
pos_test1 = np.load(f"{results_dir}/pos_test_1.npy")

y_true2 = np.load(f"{results_dir}/y_true_2.npy")
y_pred2 = np.load(f"{results_dir}/y_pred_2.npy")
y_probs2 = np.load(f"{results_dir}/y_probs_2.npy")
cls_test2 = np.load(f"{results_dir}/cls_test_2.npy")
pos_test2 = np.load(f"{results_dir}/pos_test_2.npy")

y_true = np.concatenate([y_true1, y_true2])
y_pred = np.concatenate([y_pred1, y_pred2])
y_probs = np.concatenate([y_probs1, y_probs2])
cls_test = np.concatenate([cls_test1, cls_test2])
pos_test = np.concatenate([pos_test1, pos_test2])

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
acc = accuracy_score(y_true, y_pred)
metrics_text = f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | Acc: {acc:.3f}"
print(metrics_text)

metrics_file = os.path.join(results_dir, "global_metrics.txt")
with open(metrics_file, 'w') as f:
    f.write(metrics_text)
    
# -----------------------------
# ROC + PR Curves
# -----------------------------
fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)
precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
ap_score = average_precision_score(y_true, y_probs)

plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
plt.plot([0,1],[0,1],'--',c='gray')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve", fontweight='bold'); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(results_dir, "roc_curve.png"), dpi=300); plt.close()

plt.figure(figsize=(6,5))
plt.plot(recall_curve, precision_curve, lw=2, label=f"AP={ap_score:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision–Recall Curve"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(results_dir, "precision_recall_curve.png"), dpi=300); plt.close()

# -----------------------------
# Normalized Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)
cm_percent = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_percent*100, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=["Non-FP","FP"], yticklabels=["Non-FP","FP"],
            annot_kws={"size":12,"weight":"bold"}, cbar=False)
plt.title("Normalized Confusion Matrix (%)", fontweight='bold'); plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix_percent.png"), dpi=300); plt.close()


# FP / TP / Other confusion
conf_matrix = np.zeros((3, 2), dtype=int)
for true_label, predicted_label, cls in zip(y_true, y_pred, cls_test):
    if cls == 'fp':
        row = 0
    elif cls == 'tp':
        row = 1
    else:
        row = 2
    col = predicted_label
    conf_matrix[row, col] += 1

# Compute percentages (row-wise)
conf_percent = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
conf_percent = np.nan_to_num(conf_percent)  # avoid NaN if a row sums to zero

# Combine counts + percentages in display text
annot_text = np.empty_like(conf_matrix, dtype=object)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        count = conf_matrix[i, j]
        pct = conf_percent[i, j] * 100
        annot_text[i, j] = f"{pct:.1f}%"

# Plot heatmap
row_labels = ['FP', 'TP', 'Other']
col_labels = ['Class 0', 'Class 1']

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=annot_text, fmt='', cmap="Blues",
            xticklabels=col_labels, yticklabels=row_labels, cbar=False,
            annot_kws={"fontsize": 10, "ha": "center", "va": "center"})

plt.title("Confusion Matrix (FP, TP, Other vs. Predicted Labels)", fontsize=11, fontweight='bold')
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix_fp_tp_other_with_percentages.png"), dpi=300)
plt.close()


# -----------------------------
# Per-token metrics (with AUROC)
# -----------------------------

# Smoothing kernel
kernel = np.array([0.05, 0.1, 0.7, 0.1, 0.05])


def smooth(arr, kernel):

    sm = arr.copy()

    for j in range(2, len(arr)-2):
        sm[j] = np.sum(arr[j-2:j+3]*kernel)

    return sm

positions = np.arange(min_position, max_position+1)
accs, precs, recs, f1s, aurocs = [], [], [], [], []
for pos in positions:
    mask = np.abs(pos_test - pos) <= position_margin
    if np.sum(mask) == 0 or len(np.unique(y_true[mask])) < 2:
        accs.append(np.nan)
        precs.append(np.nan)
        recs.append(np.nan)
        f1s.append(np.nan)
        aurocs.append(np.nan)
        continue
    y_t, y_p, y_pr = y_true[mask], y_pred[mask], y_probs[mask]
    accs.append(accuracy_score(y_t, y_p))
    p, r, f, _ = precision_recall_fscore_support(y_t, y_p, average='binary', zero_division=0)
    precs.append(p)
    recs.append(r)
    f1s.append(f)
    aurocs.append(roc_auc_score(y_t, y_pr))

# Apply smoothing
accs_smoothed = smooth(np.array(accs), kernel)
precs_smoothed = smooth(np.array(precs), kernel)
recs_smoothed = smooth(np.array(recs), kernel)
f1s_smoothed = smooth(np.array(f1s), kernel)
aurocs_smoothed = smooth(np.array(aurocs), kernel)

# Plotting the results
fig, ax = plt.subplots(figsize=(8, 5))

# Plot all metrics on the same axis
ax.plot(positions, accs_smoothed, label="Accuracy", linewidth=2)
ax.plot(positions, precs_smoothed, label="Precision", linewidth=2)
ax.plot(positions, recs_smoothed, label="Recall", linewidth=2)
ax.plot(positions, f1s_smoothed, label="F1", linewidth=2)
ax.plot(positions, aurocs_smoothed, label="AUROC", linewidth=2, linestyle='--', color='k')

np.save(os.path.join(results_dir, "accs.npy"), accs)
np.save(os.path.join(results_dir, "precs.npy"), precs)
np.save(os.path.join(results_dir, "recs.npy"), recs)
np.save(os.path.join(results_dir, "f1s.npy"), f1s)
np.save(os.path.join(results_dir, "aurocs.npy"), aurocs_smoothed)


plt.legend()
ax.set_xlabel("Token Position")
ax.set_ylabel("Metric Value")
plt.xlim(8, 151)
plt.title("Performance by Token Position", fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "metrics_by_position_with_auroc.png"), dpi=300)
plt.close()

