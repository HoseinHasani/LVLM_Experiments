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
load_from_existing = False
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

attn_dir = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/rz15dire/Ensemble/experiments/new_per_sample_outputs/attentions_greedy_rep_aware_double"
score_dir = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/rz15dire/Ensemble/experiments/new_per_sample_outputs/scores_greedy_rep_aware_double"
priors_dir = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/rz15dire/LVLM_Experiments/classification/final_cl_results/priors"
base_save_dir = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/rz15dire/LVLM_Experiments/classification/final_cl_results"

attn_dir = "../../data/all layers all attention tp fp rep double r"
score_dir = "../../data/double_scores r"

priors_dir = "priors_r"
base_save_dir = "final_cl_results_r"

os.makedirs(base_save_dir, exist_ok=True)
os.makedirs(priors_dir, exist_ok=True)
dataset_path = f"{base_save_dir}/cls_data_r_{target_rep}"


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
        meta = attn_dict.get(cls_, {}).get("meta", [])[k]
        total_reps = meta.get("total_reps", 1)
        lemmas = meta.get("lemma")
        is_first_occ = meta.get("is_first_occ", False)

        if is_first_occ:
            all_occ_indices = meta.get("all_occurrence_indices")
        else:
            all_occ_indices = None

        a_e = attn_entries[k]
        s_e = score_entries[k]
        
        
        if len(a_e['token_indices']) < 1:
            continue
        
        if target_rep == 1:
            if not is_first_occ:
                continue
        else:
            if is_first_occ:
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
        
        results.append((a_idx, topk_vals[..., :n_top_k], topk_vals_next[..., :n_top_k], logits, total_reps, lemmas, all_occ_indices))
        
    return results


def extract_all_features(attn_file_dict, score_file_dict, file_ids, n_layers, n_heads, min_position, max_position):
    X, y, pos_list, cls_list, lemma_list, all_occ_ind_list, rep_list = [], [], [], [], [], [], []

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


            for src, idx, topk_arr1, topk_arr2, logits, total_reps, lemma, all_occ_indices in all_samples:
                token_pos = int(idx)
                if token_pos < min_position or token_pos > max_position:
                    continue

                rep_list.append(total_reps)
                n_repeat = min(total_reps-1, 3)

                features = []
                
                features.append(n_repeat)
                
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
                lemma_list.append(lemma)
                all_occ_ind_list.append(all_occ_indices)
    if len(X) == 0:
        return None, None, None, None, None, None, None
    
    rep_list = np.clip(rep_list, 1, 4) # hard-coded values for now
    
    return np.array(X), np.array(y), np.array(pos_list), np.array(cls_list), np.array(lemma_list, dtype=object), np.array(all_occ_ind_list, dtype=object), rep_list


def upsample_to_target(indices, target_size, rng):
    """
    Upsample a set of indices to exactly target_size using:
    - zero or one full repetition
    - remaining samples without replacement
    """
    n = len(indices)
    if n == 0:
        return []

    if n >= target_size:
        return rng.choice(indices, size=target_size, replace=False).tolist()

    # At most one full repetition
    reps = target_size // n
    reps = min(reps, 1)

    out = indices * reps
    remaining = target_size - len(out)

    pool = indices.copy()
    if remaining > len(pool):
        # one extra repetition allowed, still capped
        pool = pool + indices

    out += rng.choice(pool, size=remaining, replace=False).tolist()
    return out



def balance_samples_by_position(
    X, y, pos, cls, lmm, occ, reps,
    classes=("fp", "tp"),
    random_state=0,
):
    """
    Balance samples per token position.
    
    Parameters
    ----------
    classes : tuple
        ("fp", "tp") or ("fp", "tp", "other")
    """
    rng = np.random.default_rng(random_state)

    X_out, y_out, pos_out, cls_out, lmm_out, occ_out, reps_out = [], [], [], [], [], [], []

    unique_positions = np.unique(pos)

    for p in unique_positions:
        mask_p = (pos == p)

        # collect indices per class
        class_to_indices = {}
        for c in classes:
            class_to_indices[c] = np.where(mask_p & (cls == c))[0].tolist()

        # target size = max class count at this position
        target = max(len(v) for v in class_to_indices.values())
        if target == 0:
            continue

        for c, idxs in class_to_indices.items():
            balanced_idxs = upsample_to_target(idxs, target, rng)

            X_out.append(X[balanced_idxs])
            y_out.append(y[balanced_idxs])
            pos_out.append(pos[balanced_idxs])
            cls_out.append(cls[balanced_idxs])
            lmm_out.append(lmm[balanced_idxs])
            occ_out.append(occ[balanced_idxs])
            reps_out.append(reps[balanced_idxs])

    return (
        np.concatenate(X_out, axis=0),
        np.concatenate(y_out, axis=0),
        np.concatenate(pos_out, axis=0),
        np.concatenate(cls_out, axis=0),
        np.concatenate(lmm_out, axis=0),
        np.concatenate(occ_out, axis=0),
        np.concatenate(reps_out, axis=0),
    )





def drop_samples(X, y, pos, cls, lmm, occ, reps,target="other", dropout_ratio=0.5):
    """
    Randomly remove a fraction of 'target' class samples from the dataset.
    """
    mask_target = (cls == target)
    target_indices = np.where(mask_target)[0]

    if dropout_ratio <= 0 or len(target_indices) == 0:
        return X, y, pos, cls, lmm, occ, reps
    if dropout_ratio >= 1.0:
        keep_mask = ~mask_target
    else:
        n_drop = int(len(target_indices) * dropout_ratio)
        drop_indices = np.random.choice(target_indices, size=n_drop, replace=False)
        keep_mask = np.ones(len(cls), dtype=bool)
        keep_mask[drop_indices] = False

    print(f"Dropped {np.sum(~keep_mask)} / {len(cls)} ('{target}' samples removed: {100*dropout_ratio:.1f}%)")
    return X[keep_mask], y[keep_mask], pos[keep_mask], cls[keep_mask], lmm[keep_mask], occ[keep_mask], reps[keep_mask]



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
score_ids = list(score_files_dict.keys())

intersect_ids = list(set(attn_ids).intersection(set(score_ids)))

n_files = len(intersect_ids)

print(f"Found {n_files} files")


if dataset_path and os.path.exists(f"{dataset_path}/x.npy") and load_from_existing:
    print("Loading saved dataset...")
    X_all = np.load(f"{dataset_path}/x.npy")
    y_all = np.load(f"{dataset_path}/y.npy")
    pos_all = np.load(f"{dataset_path}/pos.npy")
    cls_all = np.load(f"{dataset_path}/cls.npy")
    lemma_all = np.load(f"{dataset_path}/lemma.npy")
else:
    X_all, y_all, pos_all, cls_all, lemma_all, all_occ_list_all, reps_all = extract_all_features(
        attn_files_dict, score_files_dict, intersect_ids, n_layers, n_heads, min_position, max_position
    )

    if X_all is not None:
        # dataset_path = f"{base_save_dir}/cls_data_{target_rep}"
        os.makedirs(dataset_path, exist_ok=True)
        np.save(os.path.join(dataset_path, "x.npy"), X_all)
        np.save(os.path.join(dataset_path, "y.npy"), y_all)
        np.save(os.path.join(dataset_path, "pos.npy"), pos_all)
        np.save(os.path.join(dataset_path, "cls.npy"), cls_all)
        np.save(os.path.join(dataset_path, "lemma.npy"), lemma_all)
        np.save(os.path.join(dataset_path, "occ_indices.npy"), all_occ_list_all)
        np.save(os.path.join(dataset_path, "repeats.npy"), reps_all)
        print(f"Dataset saved in '{dataset_path}/'")
        


if not use_attns:
    X_all = X_all[:, -4:]    
elif not use_logits:
    X_all = np.concatenate([X_all[:, :-4], X_all[:, -1:]], -1)
elif not use_attns and not use_logits:
    assert False
    X_all = X_all[:, -1:]
    
if not pos_condition:
    assert pos_condition
    X_all = X_all[:, :-1]
    
    


if other_dropout > 0:
    X_all, y_all, pos_all, cls_all, lemma_all, all_occ_list_all, reps_all = drop_samples(
        X_all, y_all, pos_all, cls_all, lemma_all, all_occ_list_all, reps_all, target="other", dropout_ratio=other_dropout
    )

if tp_dropout > 0:
    X_all, y_all, pos_all, cls_all, lemma_all, all_occ_list_all, reps_all = drop_samples(
        X_all, y_all, pos_all, cls_all, lemma_all, all_occ_list_all, reps_all, target="tp", dropout_ratio=other_dropout
    )


reps_org = reps_all.copy()

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
lemma_train, lemma_test = lemma_all[:n_train], lemma_all[-n_test:]
all_occ_train, all_occ_test = all_occ_list_all[:n_train], all_occ_list_all[-n_test:]
reps_train, reps_test = reps_all[:n_train], reps_all[-n_test:]

X_train_original = X_all #X_train.copy()
y_train_original = y_all #y_train.copy()

# -----------------------------
# Apply Adaptive FP Balancing
# -----------------------------

if balanced_train:
    X_train, y_train, pos_train, cls_train, lemma_train, occ_train, reps_train = \
        balance_samples_by_position(
            X_train, y_train, pos_train, cls_train,
            lemma_train, all_occ_train, reps_train,
            classes=("fp", "tp")
        )

if balanced_test:
    X_test, y_test, pos_test, cls_test, lemma_test, occ_test, reps_test = \
        balance_samples_by_position(
            X_test, y_test, pos_test, cls_test,
            lemma_test, all_occ_test, reps_test,
            classes=("fp", "tp")
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

# -----------------------------
# Empirical position priors p(y | t) from original dataset
# -----------------------------

# Use ORIGINAL (unbalanced) data
pos_orig = pos_all          # token positions
labels_orig = y_all         # 0 = FP, 1 = TP

positions = np.arange(min_position, max_position + 1)
n_pos = len(positions)

# counts[t, y]
counts = np.zeros((n_pos, 2), dtype=np.float32)

for i, t in enumerate(positions):
    mask = (pos_orig == t)
    if mask.any():
        counts[i, 0] = np.sum(labels_orig[mask] == 0)
        counts[i, 1] = np.sum(labels_orig[mask] == 1)
    else:
        counts[i, :] = np.nan

# Convert counts → priors
priors_tokenposition = np.full((n_pos, 2), np.nan, dtype=np.float32)

total = counts.sum(axis=1)

valid = total > 0
priors_tokenposition[valid, 0] = counts[valid, 0] / total[valid]
priors_tokenposition[valid, 1] = counts[valid, 1] / total[valid]

# Fill missing positions using nearest non-NaN
priors_tokenposition[:, 0] = fill_nans_linear(priors_tokenposition[:, 0])
priors_tokenposition[:, 1] = fill_nans_linear(priors_tokenposition[:, 1])

# Numerical safety
eps = 1e-7
priors_tokenposition = np.clip(priors_tokenposition, eps, 1 - eps)
priors_tokenposition /= priors_tokenposition.sum(axis=1, keepdims=True)

# Torch tensor
priors_t = torch.tensor(
    priors_tokenposition,
    dtype=torch.float32
).to(device)

# Save
np.save(
    os.path.join(priors_dir, "priors_token_position.npy"),
    priors_tokenposition
)


######## repition prior ##########
rep_vals = reps_org.astype(int) - 1
labels   = y_train_original


n_bins = 4
eps = 1e-6

# counts[r, y]
counts = np.zeros((n_bins, 2), dtype=np.float32)

for r in range(n_bins):
    for y in [0, 1]:
        counts[r, y] = np.sum((rep_vals == r) & (labels == y))

# add smoothing
counts += eps

# COLUMN-WISE normalization
priors_r = counts / counts.sum(axis=0, keepdims=True)

priors_r = torch.tensor(priors_r, dtype=torch.float32).to(device)


np.save(
    os.path.join(priors_dir, "priors_repetition.npy"),
    priors_r.cpu().numpy()
)




# ============================================================
# Learn joint prior p(y | r, t) using a small MLP
# ============================================================

class PriorRTNet(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)  # logits for y
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------
# Prepare prior training data (USE ORIGINAL DATA, NO BALANCING)
# ------------------------------------------------------------

# Decode repetition (0..3) and position (min_position..max_position)
rep_vals = (reps_org.astype(np.int64) - 1)
pos_vals = pos_all.astype(np.float32)

# Normalize inputs
rep_norm = rep_vals / 3.0
pos_norm = (pos_vals - min_position) / (max_position - min_position)

X_prior = np.stack([rep_norm, pos_norm], axis=1)
y_prior = y_all

X_prior_t = torch.tensor(X_prior, dtype=torch.float32).to(device)
y_prior_t = torch.tensor(y_prior, dtype=torch.long).to(device)


prior_net = PriorRTNet(hidden=32).to(device)
prior_optimizer = torch.optim.Adam(
    prior_net.parameters(), lr=1e-3, weight_decay=1e-4
)
prior_criterion = nn.CrossEntropyLoss(label_smoothing=0.02)


prior_dataset = TensorDataset(X_prior_t, y_prior_t)
prior_loader = DataLoader(
    prior_dataset,
    batch_size=160,
    shuffle=True,
    drop_last=False
)

prior_net.train()

for epoch in range(10): # hard code for now
    epoch_loss = 0.0
    n_samples = 0

    for xb, yb in prior_loader:
        logits = prior_net(xb)
        loss = prior_criterion(logits, yb)

        prior_optimizer.zero_grad()
        loss.backward()
        prior_optimizer.step()

        epoch_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)

    epoch_loss /= n_samples

    if (epoch + 1) % 2 == 0:
        print(
            f"[Prior] Epoch {epoch+1:02d} | "
            f"Avg Loss: {epoch_loss:.4f}"
        )


prior_net.eval()
for p in prior_net.parameters():
    p.requires_grad = False

torch.save(prior_net.state_dict(), os.path.join(priors_dir, "prior_rt_mlp.pt"))
print("Joint prior network trained and saved.")






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

rep_norm_min = float(X_train_t[:, 0].min().item())
rep_norm_max = float(X_train_t[:, 0].max().item())


class BayesianMLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        prior_net,
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

        # Learned joint prior network
        self.prior_net = prior_net

        self.pos_norm_min = pos_norm_min
        self.pos_norm_max = pos_norm_max
        self.min_position = min_position
        self.max_position = max_position

        self.rep_norm_min = rep_norm_min
        self.rep_norm_max = rep_norm_max
        self.min_repetition = 0
        self.max_repetition = 3

        

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
    
    def decode_repetitions(self, x):
        rep_norm = x[:, 0]

        rep_real = (
            (rep_norm - self.rep_norm_min)
            / (self.rep_norm_max - self.rep_norm_min)
        ) * (self.max_repetition - self.min_repetition) + self.min_repetition

        rep_real = torch.round(rep_real).long()
        rep_real = torch.clamp(
            rep_real, self.min_repetition, self.max_repetition
        )

        return rep_real

    def forward(self, x, return_bayesian=True):
        raw_logits = self.net(x)
        raw_posteriors = torch.softmax(raw_logits, dim=1)

        if not return_bayesian:
            return raw_logits, raw_posteriors

        # ----------------------------------
        # Decode repetition and position
        # ----------------------------------
        pos_real = self.decode_positions(x)
        rep_real = self.decode_repetitions(x)

        # Normalize for prior network
        pos_norm = (pos_real - self.min_position) / (
            self.max_position - self.min_position
        )
        rep_norm = rep_real / self.max_repetition

        rt = torch.stack([rep_norm, pos_norm], dim=1)

        # ----------------------------------
        # Learned joint prior p(y | r, t)
        # ----------------------------------
        prior_logits = self.prior_net(rt)
        priors_rt = torch.softmax(prior_logits, dim=1)

        # ----------------------------------
        # Bayesian correction
        # ----------------------------------
        bayes_unnormalized = raw_posteriors * priors_rt
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
    prior_net=prior_net,
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
np.save(f"{results_dir}/lemma_test_{target_rep}.npy", lemma_test)


y_probs2 = []
y_pred2 = []
y_true2 = []
cls_test2 = []
pos_test2 = []
lemma_test2 = []

for i, r in enumerate(reps_test):
    if r < 2:
        continue
    for k in range(r-1):
        y_probs2.append(y_probs[i])
        y_pred2.append(y_pred[i])
        y_true2.append(y_true[i])
        cls_test2.append(cls_test[i])
        pos_test2.append(pos_test[i])
        lemma_test2.append(lemma_test[i])
        
target_rep = 2
np.save(f"{results_dir}/y_probs_{target_rep}.npy", y_probs2)
np.save(f"{results_dir}/y_pred_{target_rep}.npy", y_pred2)
np.save(f"{results_dir}/y_true_{target_rep}.npy", y_true2)
np.save(f"{results_dir}/cls_test_{target_rep}.npy", cls_test2)
np.save(f"{results_dir}/pos_test_{target_rep}.npy", pos_test2)
np.save(f"{results_dir}/lemma_test_{target_rep}.npy", lemma_test2)        
        






# ============================================================
# Evaluate and marginalize learned joint prior p(y | r, t)
# ============================================================

import matplotlib.pyplot as plt

prior_net.eval()

# Original (unbalanced) data
rep_vals = (reps_org.astype(np.int64) - 1)     # 0..3
pos_vals = pos_all.astype(np.int64)            # min_position..max_position

# Normalize for prior network
rep_norm = rep_vals / 3.0
pos_norm = (pos_vals - min_position) / (max_position - min_position)

X_rt = torch.tensor(
    np.stack([rep_norm, pos_norm], axis=1),
    dtype=torch.float32
).to(device)

with torch.no_grad():
    prior_logits = prior_net(X_rt)
    prior_probs = torch.softmax(prior_logits, dim=1).cpu().numpy()
    

# ============================================================
# p(y | r)
# ============================================================

p_y_given_r = np.zeros((4, 2), dtype=np.float64)

for r in range(4):
    mask = rep_vals == r
    assert mask.sum() > 0, f"No samples for repetition r={r}"
    p_y_given_r[r] = prior_probs[mask].mean(axis=0)



# ============================================================
# p(y | t)
# ============================================================

positions = np.arange(min_position, max_position + 1)
p_y_given_t = np.zeros((len(positions), 2), dtype=np.float64)

for i, t in enumerate(positions):
    mask = pos_vals == t
    # assert mask.sum() > 0, f"No samples for position t={t}"
    if mask.sum() < 1:
        continue
    p_y_given_t[i] = prior_probs[mask].mean(axis=0)





priors_r_tab = priors_r.cpu().numpy()          # shape (4, 2)
priors_t_tab = priors_t.cpu().numpy()            # shape (151, 2)




# ============================================================
# Plot p(y | r) for both classes
# ============================================================

p_y_given_r = p_y_given_r / p_y_given_r.sum(0)

r_vals = np.arange(1, 5)
width = 0.35

for y in [0, 1]:
    plt.figure(figsize=(6, 4))

    plt.bar(
        r_vals - width / 2,
        priors_r_tab[:, y],
        width=width,
        label=f"Tabular (y={y})",
        alpha=0.8
    )
    plt.bar(
        r_vals + width / 2,
        p_y_given_r[:, y],
        width=width,
        label=f"Learned (y={y})",
        alpha=0.8
    )

    plt.xlabel("Repetition (r)")
    plt.ylabel(f"p(y={y} | r)")
    plt.title(f"Repetition prior comparison (class {y})")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(priors_dir, f"prior_r_class_{y}.png")
    plt.savefig(save_path)
    plt.close()



# ============================================================
# Plot p(y | t) for both classes
# ============================================================

for y in [0, 1]:
    plt.figure(figsize=(10, 4))

    plt.plot(
        positions,
        priors_t_tab[:, y],
        label=f"Tabular (y={y})",
        linewidth=2
    )
    plt.plot(
        positions,
        p_y_given_t[:, y],
        label=f"Learned (y={y})",
        linewidth=2,
        linestyle="--"
    )

    plt.xlabel("Token position (t)")
    plt.ylabel(f"p(y={y} | t)")
    plt.title(f"Token-position prior comparison (class {y})")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(priors_dir, f"prior_t_class_{y}.png")
    plt.savefig(save_path)
    plt.close()





# ============================================================
# Difference plots (Learned − Tabular)
# ============================================================

# Repetition
for y in [0, 1]:
    plt.figure(figsize=(6, 3))
    plt.bar(
        r_vals,
        p_y_given_r[:, y] - priors_r_tab[:, y]
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Repetition (r)")
    plt.ylabel(f"Δ p(y={y} | r)")
    plt.title(f"Difference in repetition prior (class {y})")
    plt.tight_layout()

    save_path = os.path.join(priors_dir, f"diff_prior_r_class_{y}.png")
    plt.savefig(save_path)
    plt.close()

# Position
for y in [0, 1]:
    plt.figure(figsize=(10, 3))
    plt.plot(
        positions,
        p_y_given_t[:, y] - priors_t_tab[:, y]
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Token position (t)")
    plt.ylabel(f"Δ p(y={y} | t)")
    plt.title(f"Difference in token-position prior (class {y})")
    plt.tight_layout()

    save_path = os.path.join(priors_dir, f"diff_prior_t_class_{y}.png")
    plt.savefig(save_path)
    plt.close()


