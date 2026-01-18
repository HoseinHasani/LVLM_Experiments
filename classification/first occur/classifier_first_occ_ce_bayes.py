import os
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm

from scipy.stats import entropy as scipy_entropy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict



# -----------------------------
# Configuration
# -----------------------------


#######################
target_rep = 1

target_class = 'tp'
balanced_train = True
balanced_test = False
fp2tp_ratio = 1.0

train_size = 0.5
test_size = 0.5
pos_condition = True
use_logits = True
use_attns = True
#######################

attn_dir = "../../data/all layers all attention tp fp rep double"
score_dir = "../../data/double_scores"

base_save_dir = "final_cl_results"
os.makedirs(base_save_dir, exist_ok=True)
dataset_path = f"cls_data_{target_rep}"
# n_files = 3900

if target_class == 'tp':
    other_dropout = 1
    tp_dropout = 0.0
elif target_class == 'other':
    other_dropout = 0.0
    tp_dropout = 1
else:
    other_dropout = 0.2
    tp_dropout = 0.0    

n_layers, n_heads = 32, 32
min_position = 5
max_position = 155
position_margin = 2
n_top_k = 20
n_subtokens = 1
eps = 1e-10
use_entropy = True
n_epochs = 2
weight_decay = 1e-3
dropout_rate = 0.5
normalize_features = True

sfx = ""
sfx = sfx if use_logits else sfx + "_no_logits"
sfx = sfx if use_attns else sfx + "_no_attns"

exp_name = f"exp__bayes_ce{sfx}"
save_dir = os.path.join(base_save_dir, exp_name)
model_dir = os.path.join(save_dir, "model")
results_dir = os.path.join(save_dir, "results")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")


# -----------------------------
# Helper Functions
# -----------------------------
def compute_entropy(values):
    if len(values) == 0:
        return 0.0
    prob = np.array(values, dtype=float)
    prob = prob / (np.sum(prob, axis=-1, keepdims=True) + eps)
    return -np.sum(prob * np.log(prob + eps), axis=-1)


def logit_entropy(logits, k=100, eps=1e-10):
    logits = logits.astype(np.float32)
    

    topk_logits = np.partition(logits, -k)[-k:]
    topk_logits -= np.max(topk_logits)

    probs = np.exp(topk_logits)
    probs /= probs.sum() + eps

    return -np.sum(probs * np.log(probs + eps))

def softmax(x, eps=1e-10):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x) + eps)


def extract_attention_score_values(attn_dict, score_dict, cls_, source="image"):
    results = []
    attn_entries = attn_dict.get(cls_, {}).get(source, [])
    score_entries = score_dict.get(cls_, {})
    
    for k in range(min(len(attn_entries), len(score_entries))):
        
        a_e = attn_entries[k]
        s_e = score_entries[k]
        
        
        if len(a_e['token_indices']) < 1:
            continue
        
        if target_rep == 1:
            if a_e['rep_num'] != 1:
                continue
        else:
            if a_e['rep_num'] == 1:
                continue
        
        a_idx = a_e['token_indices'][0]
        s_idx = s_e[0]
        logits = np.sort(s_e[1])[-100:]
    
        if a_idx != s_idx:
            continue
        
        topk_vals = np.array(a_e["subtoken_results"][0]["topk_values"], dtype=float)
        if topk_vals.ndim != 3:
            continue
        
        topk_vals_next = np.array(a_e["subtoken_results"][0]["topk_values_next"], dtype=float)
        if topk_vals_next.ndim != 3:
            continue
        
        results.append((a_idx, topk_vals[..., :n_top_k], topk_vals_next[..., :n_top_k], logits))
        
    return results


def extract_all_features(attn_file_dict, score_file_dict, file_ids, n_layers, n_heads, min_position, max_position):
    X, y, pos_list, cls_list = [], [], [], []

    for j in tqdm(range(len(file_ids)), desc="Extracting features"):
        file_id = file_ids[j]
        at_f = attn_files_dict[file_id]
        sc_f = score_file_dict[file_id]

        try:
            with open(at_f, "rb") as handle:
                attn_data_dict = pickle.load(handle)
        except Exception:
            continue
        
        try:
            with open(sc_f, "rb") as handle:
                score_data_dict = pickle.load(handle)
        except Exception:
            continue

        for cls_, label in [("fp", 0), ("tp", 1), ("other", 1)]:
            img_samples = extract_attention_score_values(attn_data_dict, score_data_dict, cls_, "image")
            all_samples = []
            if img_samples:
                all_samples.extend([("image", *s) for s in img_samples])


            for src, idx, topk_arr1, topk_arr2, logits in all_samples:
                token_pos = int(idx)
                if token_pos < min_position or token_pos > max_position:
                    continue

                features = []
                
                for l in range(n_layers):
                    for h in range(n_heads):
                        vals1 = topk_arr1[l, h, :]
                        vals2 = topk_arr2[l, h, :]
                        mean_attention1 = np.mean(vals1)
                        mean_attention2 = np.mean(vals2)
                        features.append(mean_attention1)
                        features.append(mean_attention2)
                        
                        if use_entropy:
                            features.append(compute_entropy(vals1))
                            features.append(compute_entropy(vals2))
                            
                            
                # logit featuers:
                ent = logit_entropy(logits)
                max_logit = float(np.max(logits))
                max_softmax = float(np.max(softmax(logits)))
                
                features.extend([ent, max_logit, max_softmax])
                
                # normalized token position:
                features.append(token_pos / max_position)

                X.append(features)
                y.append(label)
                pos_list.append(token_pos)
                cls_list.append(cls_)
    if len(X) == 0:
        return None, None, None, None
    return np.array(X), np.array(y), np.array(pos_list), np.array(cls_list)


def compute_adaptive_fp_replication_factors(y_all, pos_all, win=5):
    """Compute adaptive FP replication factor per token position."""
    min_pos, max_pos = int(pos_all.min()), int(pos_all.max())
    replication_factors = {}
    pos_to_labels = defaultdict(list)

    for pos, label in zip(pos_all, y_all):
        pos_to_labels[int(pos)].append(int(label))

    for j in range(min_pos, max_pos + 1):
        local_labels = []
        for k in range(j - win, j + win + 1):
            if k in pos_to_labels:
                local_labels.extend(pos_to_labels[k])

        if len(local_labels) == 0:
            replication_factors[j] = 1
            continue

        n_0 = np.sum(np.array(local_labels) == 0)
        n_1 = np.sum(np.array(local_labels) == 1)
        if n_0 == 0:
            replication_factors[j] = 20
        else:
            replication_factors[j] = max(int(fp2tp_ratio * np.round(n_1 / n_0)), 1)

    print(f"Computed adaptive replication factors for positions {min_pos}–{max_pos}")
    return replication_factors


def balance_fp_samples_adaptive(X, y, pos, cls, fp_factors):
    """Replicate FP samples based on adaptive per-position factors."""
    X_bal, y_bal, pos_bal, cls_bal = [X], [y], [pos], [cls]

    for p, factor in fp_factors.items():
        mask = (y == 0) & (pos == p)
        if np.any(mask) and factor > 1:
            factor = min(factor, 30)
            X_rep = np.repeat(X[mask], factor, axis=0)
            y_rep = np.repeat(y[mask], factor, axis=0)
            pos_rep = np.repeat(pos[mask], factor, axis=0)
            cls_rep = np.repeat(cls[mask], factor, axis=0)
            X_bal.append(X_rep)
            y_bal.append(y_rep)
            pos_bal.append(pos_rep)
            cls_bal.append(cls_rep)

    X_bal = np.concatenate(X_bal, axis=0)
    y_bal = np.concatenate(y_bal, axis=0)
    pos_bal = np.concatenate(pos_bal, axis=0)
    cls_bal = np.concatenate(cls_bal, axis=0)

    return X_bal, y_bal, pos_bal, cls_bal

def drop_samples(X, y, pos, cls, target="other", dropout_ratio=0.5):
    """
    Randomly remove a fraction of 'target' class samples from the dataset.
    """
    mask_target = (cls == target)
    target_indices = np.where(mask_target)[0]

    if dropout_ratio <= 0 or len(target_indices) == 0:
        return X, y, pos, cls
    if dropout_ratio >= 1.0:
        keep_mask = ~mask_target
    else:
        n_drop = int(len(target_indices) * dropout_ratio)
        drop_indices = np.random.choice(target_indices, size=n_drop, replace=False)
        keep_mask = np.ones(len(cls), dtype=bool)
        keep_mask[drop_indices] = False

    print(f"Dropped {np.sum(~keep_mask)} / {len(cls)} ('{target}' samples removed: {100*dropout_ratio:.1f}%)")
    return X[keep_mask], y[keep_mask], pos[keep_mask], cls[keep_mask]



# -----------------------------
# Load or Extract Dataset
# -----------------------------
attn_files = sorted(glob(os.path.join(attn_dir, "attentions_*.pkl")))
score_files = sorted(glob(os.path.join(score_dir, "scores_*.pkl")))

attn_files_dict = {}
for f in attn_files:
    key = int(os.path.splitext(os.path.basename(f))[0].split('_')[1])
    attn_files_dict[key] = f

score_files_dict = {}
for f in score_files:
    key = int(os.path.splitext(os.path.basename(f))[0].split('_')[1])
    score_files_dict[key] = f


attn_ids = list(attn_files_dict.keys())
attn_ids = list(score_files_dict.keys())

intersect_ids = list(set(attn_files_dict).intersection(set(score_files_dict)))

n_files = len(intersect_ids)

print(f"Found {n_files} files")


if dataset_path and os.path.exists(f"{dataset_path}/x.npy"):
# if False:
    print("Loading saved dataset...")
    X_all = np.load(f"{dataset_path}/x.npy")
    y_all = np.load(f"{dataset_path}/y.npy")
    pos_all = np.load(f"{dataset_path}/pos.npy")
    cls_all = np.load(f"{dataset_path}/cls.npy")
else:
    X_all, y_all, pos_all, cls_all = extract_all_features(
        attn_files_dict, score_files_dict, intersect_ids, n_layers, n_heads, min_position, max_position
    )

    if X_all is not None:
        dataset_path = f"cls_data_{target_rep}"
        os.makedirs(dataset_path, exist_ok=True)
        np.save(os.path.join(dataset_path, "x.npy"), X_all)
        np.save(os.path.join(dataset_path, "y.npy"), y_all)
        np.save(os.path.join(dataset_path, "pos.npy"), pos_all)
        np.save(os.path.join(dataset_path, "cls.npy"), cls_all)
        print(f"Dataset saved in '{dataset_path}/'")
        


if not use_attns:
    X_all = X_all[:, -4:]    
elif not use_logits:
    X_all = np.concatenate([X_all[:, :-4], X_all[:, -1:]], -1)
elif not use_attns and not use_logits:
    X_all = X_all[:, -1:]
    
if not pos_condition:
    X_all = X_all[:, :-1]

if other_dropout > 0:
    X_all, y_all, pos_all, cls_all = drop_samples(
        X_all, y_all, pos_all, cls_all, target="other", dropout_ratio=other_dropout
    )

if tp_dropout > 0:
    X_all, y_all, pos_all, cls_all = drop_samples(
        X_all, y_all, pos_all, cls_all, target="tp", dropout_ratio=other_dropout
    )
# -----------------------------
# Train/Test Split
# -----------------------------

n_total = len(X_all)
n_train = int(n_total * train_size)
n_test = int(n_total * test_size)
if n_train + n_test > n_total:
    n_test = n_total - n_train

X_train, X_test = X_all[:n_train], X_all[-n_test:]
y_train, y_test = y_all[:n_train], y_all[-n_test:]
pos_train, pos_test = pos_all[:n_train], pos_all[-n_test:]
cls_train, cls_test = cls_all[:n_train], cls_all[-n_test:]




# -----------------------------
# Apply Adaptive FP Balancing
# -----------------------------
train_fp_factors = compute_adaptive_fp_replication_factors(y_all, pos_train, win=5)
test_fp_factors = train_fp_factors #compute_adaptive_fp_replication_factors(y_test, pos_test, win=5)


if balanced_train:
    X_train, y_train, pos_train, cls_train = balance_fp_samples_adaptive(
        X_train, y_train, pos_train, cls_train, train_fp_factors
    )

if balanced_test:
    X_test, y_test, pos_test, cls_test = balance_fp_samples_adaptive(
        X_test, y_test, pos_test, cls_test, test_fp_factors
    )




print(f"Train size: {len(y_train)} | TP={np.sum(y_train==1)}, FP={np.sum(y_train==0)}")
print(f"Test size:  {len(y_test)} | TP={np.sum(y_test==1)}, FP={np.sum(y_test==0)}")


# -----------------------------
# Build position-dependent priors
# -----------------------------

def fill_nans_linear(x):
    x = x.astype(float)
    n = len(x)
    idx = np.arange(n)

    valid = ~np.isnan(x)
    if valid.sum() == 0:
        return x  

    x_filled = x.copy()
    x_filled[~valid] = np.interp(
        idx[~valid],
        idx[valid],
        x[valid]
    )
    return x_filled


positions = np.arange(min_position, max_position + 1)

fp_factors = np.array([
    train_fp_factors.get(p, np.nan) for p in positions
], dtype=float)

fp_factors = fill_nans_linear(fp_factors)


kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
fp_factors_smoothed = fp_factors.copy()
fp_factors_smoothed[-2:2] = np.convolve(fp_factors, kernel, mode="same")[-2:2]

p_fp = 1.0 / (1.0 + fp_factors_smoothed)
p_tp = fp_factors_smoothed / (1.0 + fp_factors_smoothed)

eps = 1e-7
p_fp = np.clip(p_fp, eps, 1 - eps)
p_tp = np.clip(p_tp, eps, 1 - eps)

p_tp = fill_nans_linear(p_tp)
p_fp = fill_nans_linear(p_fp)

priors = np.stack([p_fp, p_tp], axis=1) # shape: (151, 2)
priors_t = torch.tensor(priors, dtype=torch.float32).to(device)


# -----------------------------
# Torch Model
# -----------------------------
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)
if normalize_features:
    mean, std = X_train_t.mean(0, keepdim=True), X_train_t.std(0, keepdim=True) + 1e-6
    X_train_t, X_test_t = (X_train_t - mean) / std, (X_test_t - mean) / std
    torch.save({"mean": mean, "std": std}, os.path.join(model_dir, "scaler.pt"))

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=128, shuffle=False)


pos_norm_min = float(X_train_t[:, -1].min().item())
pos_norm_max = float(X_train_t[:, -1].max().item())



class BayesianMLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        priors,
        pos_norm_min,
        pos_norm_max,
        min_position,
        max_position,
        hidden1=512,
        hidden2=64,
        dropout_rate=0.5,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, hidden1), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, 2)
        )

        # Position-dependent priors (probabilities)
        self.register_buffer("priors", priors)

        self.pos_norm_min = pos_norm_min
        self.pos_norm_max = pos_norm_max
        self.min_position = min_position
        self.max_position = max_position

    def decode_positions(self, x):
        pos_norm = x[:, -1]

        pos_real = (
            (pos_norm - self.pos_norm_min)
            / (self.pos_norm_max - self.pos_norm_min)
        ) * (self.max_position - self.min_position) + self.min_position

        pos_real = torch.round(pos_real).long()
        pos_real = torch.clamp(
            pos_real, self.min_position, self.max_position
        )

        return pos_real

    def forward(self, x, return_bayesian=True):
        """
        Returns:
          - raw_logits
          - raw_posteriors
          - bayesian_posteriors
          - bayesian_decision
        """
        # Raw network output
        raw_logits = self.net(x)
        raw_posteriors = torch.softmax(raw_logits, dim=1)

        if not return_bayesian:
            return raw_logits, raw_posteriors

        # Decode token positions
        pos_real = self.decode_positions(x)
        pos_idx = pos_real - self.min_position

        # Fetch priors for each sample
        priors = self.priors[pos_idx]  # [B, 2]

        # Bayesian correction (probability space)
        bayes_unnormalized = raw_posteriors * priors
        bayes_posteriors = bayes_unnormalized / (
            bayes_unnormalized.sum(dim=1, keepdim=True) + 1e-8
        )

        bayes_decision = torch.argmax(bayes_posteriors, dim=1)

        return (
            raw_logits,
            raw_posteriors,
            bayes_posteriors,
            bayes_decision,
        )



clf = BayesianMLPClassifier(
    input_dim=X_train_t.shape[1],
    priors=priors_t,
    pos_norm_min=pos_norm_min,
    pos_norm_max=pos_norm_max,
    min_position=min_position,
    max_position=max_position,
    dropout_rate=dropout_rate,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(clf.parameters(), lr=1e-3, weight_decay=weight_decay)

train_losses, test_losses = [], []
print(f"\nTraining PyTorch MLP for {n_epochs} epochs...")

for epoch in range(n_epochs):
    # -------------------------
    # Training
    # -------------------------
    clf.train()
    running_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.long().to(device)

        optimizer.zero_grad()

        raw_logits, _ = clf(xb, return_bayesian=False)

        loss = criterion(raw_logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # -------------------------
    # Validation
    # -------------------------
    clf.eval()
    val_loss = 0.0
    correct_raw, correct_bayes, total = 0, 0, 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.long().to(device)

            (
                raw_logits,
                raw_post,
                bayes_post,
                bayes_pred,
            ) = clf(xb)

            # Validation loss still uses RAW logits
            loss = criterion(raw_logits, yb)
            val_loss += loss.item() * xb.size(0)

            # Raw decision (balanced)
            raw_pred = torch.argmax(raw_post, dim=1)

            correct_raw += (raw_pred == yb).sum().item()
            correct_bayes += (bayes_pred == yb).sum().item()
            total += yb.size(0)

    val_loss /= len(test_loader.dataset)
    test_losses.append(val_loss)

    print(
        f"Epoch {epoch+1}/{n_epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc (raw): {correct_raw/total:.3f} | "
        f"Val Acc (bayes): {correct_bayes/total:.3f}"
    )


torch.save(clf.state_dict(), os.path.join(model_dir, "pytorch_mlp.pt"))
print("Model saved.\n")

# -----------------------------
# Evaluation (with probabilities)
# -----------------------------


clf.eval()
y_probs, y_pred, y_true = [], [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)

        (
            raw_logits,
            raw_post,
            bayes_post,
            bayes_pred,
        ) = clf(xb)

        y_probs.extend(bayes_post[:, 1].cpu().numpy())

        y_pred.extend(bayes_pred.cpu().numpy())

        y_true.extend(yb.numpy())

y_probs = np.array(y_probs)
y_pred  = np.array(y_pred)
y_true  = np.array(y_true)


np.save(f"{results_dir}/y_probs_{target_rep}.npy", y_probs)
np.save(f"{results_dir}/y_pred_{target_rep}.npy", y_pred)
np.save(f"{results_dir}/y_true_{target_rep}.npy", y_true)
np.save(f"{results_dir}/cls_test_{target_rep}.npy", cls_test)
np.save(f"{results_dir}/pos_test_{target_rep}.npy", pos_test)



