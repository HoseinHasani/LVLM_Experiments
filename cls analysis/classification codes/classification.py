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
from eval_module import evaluate




#######################

attn = True
logit = True

first = True
poscond = True
repcond = True

balanced_train = True
balanced_test = False

seed = 0
#######################



target_rep = 1
train_size = 0.5
test_size = 0.5


base_save_dir = "ablation/"
if first:
    dataset_path = "../data/1"
else:
    dataset_path = "../data"    

os.makedirs(base_save_dir, exist_ok=True)



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


exp_name = f"attn_{attn}_logit_{logit}_first_{first}_poscond_{poscond}_repcond_{repcond}_btrain_{balanced_train}_btest_{balanced_test}"
save_dir = os.path.join(base_save_dir, exp_name)
results_dir = os.path.join(save_dir, "results")
os.makedirs(results_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")


np.random.seed(seed)

def upsample_class_fractional(
    indices,
    target_size,
    rng,
    max_repeat=40,
):
    """
    Upsample indices to target_size using:
    - integer repetition
    - fractional sampling without replacement
    """
    n = len(indices)
    if n == 0:
        return []

    r = target_size / n
    k = int(np.floor(r))
    k = min(k, max_repeat)

    out = indices * k
    remaining = target_size - len(out)

    if remaining > 0:
        remaining = min(remaining, n)
        sampled = rng.choice(indices, size=remaining, replace=False).tolist()
        out += sampled

    return out



def balance_samples_by_position(
    X, y, pos, cls, lmm, occ, reps,
    classes=("fp", "tp"),
    win=2,
    max_repeat=40,
    random_state=0,
):
    rng = np.random.default_rng(random_state)

    X_bal, y_bal, pos_bal, cls_bal, lmm_bal, occ_bal, reps_bal = \
        [X], [y], [pos], [cls], [lmm], [occ], [reps]

    min_pos, max_pos = int(pos.min()), int(pos.max())

    for j in range(min_pos, max_pos + 1):
        # window mask
        mask_w = (pos >= j - win) & (pos <= j + win)

        # collect indices per class
        class_indices = {
            c: np.where(mask_w & (cls == c))[0].tolist()
            for c in classes
        }

        # remove empty classes
        present = {c: idxs for c, idxs in class_indices.items() if len(idxs) > 0}
        if len(present) < 2:
            continue

        # target size
        N_star = max(len(v) for v in present.values())

        for c, idxs in present.items():
            up_idx = upsample_class_fractional(
                idxs,
                target_size=N_star,
                rng=rng,
                max_repeat=max_repeat,
            )

            X_bal.append(X[up_idx])
            y_bal.append(y[up_idx])
            pos_bal.append(pos[up_idx])
            cls_bal.append(cls[up_idx])
            lmm_bal.append(lmm[up_idx])
            occ_bal.append(occ[up_idx])
            reps_bal.append(reps[up_idx])

    return (
        np.concatenate(X_bal, axis=0),
        np.concatenate(y_bal, axis=0),
        np.concatenate(pos_bal, axis=0),
        np.concatenate(cls_bal, axis=0),
        np.concatenate(lmm_bal, axis=0),
        np.concatenate(occ_bal, axis=0),
        np.concatenate(reps_bal, axis=0),
    )




print("Loading saved dataset...")
X_all = np.load(f"{dataset_path}/x.npy")
y_all = np.load(f"{dataset_path}/y.npy")
pos_all = np.load(f"{dataset_path}/pos.npy")
cls_all = np.load(f"{dataset_path}/cls.npy")
lemma_all = np.load(f"{dataset_path}/cls.npy", allow_pickle=True)
all_occ_list_all = np.load(f"{dataset_path}/cls.npy", allow_pickle=True)
reps_all = np.load(f"{dataset_path}/repeats.npy", allow_pickle=True)

if not first:
    perm = np.random.permutation(len(y_all))
    X_all = X_all[perm].copy()
    y_all = y_all[perm].copy()
    pos_all = pos_all[perm].copy()
    cls_all = cls_all[perm].copy()
    lemma_all = lemma_all[perm].copy()
    all_occ_list_all = all_occ_list_all[perm].copy()
    reps_all = reps_all[perm].copy()

if not attn:
    X_all[:, 1:-4] = np.random.randn(X_all.shape[0], 4*1024)
    
if not logit:
    X_all[:, -4:-1] = np.random.randn(X_all.shape[0], 3)
    
if not repcond:
    X_all[:, 0:1] = np.random.randn(X_all.shape[0], 1)
    
if not poscond:
    X_all[:, -1:] = np.random.randn(X_all.shape[0], 1)

    
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
# Apply Adaptive  Balancing
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


print('sum', np.sum(y_test) / len(y_test))

print(f"Train size: {len(y_train)} | TP={np.sum(y_train==1)}, FP={np.sum(y_train==0)}")
print(f"Test size:  {len(y_test)} | TP={np.sum(y_test==1)}, FP={np.sum(y_test==0)}")


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
        if balanced_train:
            bayes_unnormalized = raw_posteriors * priors_rt
        else:
            bayes_unnormalized = raw_posteriors
            
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


print(exp_name)

print('FIRST')
evaluate(y_true, y_pred, y_probs, cls_test, pos_test, results_dir=None)

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
        


print('NON-FIRST')
evaluate(y_true2, y_pred2, y_probs2, cls_test2, pos_test2, results_dir=None)


y_true = np.concatenate([y_true, y_true2])
y_pred = np.concatenate([y_pred, y_pred2])
y_probs = np.concatenate([y_probs, y_probs2])
cls_test = np.concatenate([cls_test, cls_test2])
pos_test = np.concatenate([pos_test, pos_test2])

print('ALL')
evaluate(y_true, y_pred, y_probs, cls_test, pos_test, results_dir=results_dir)
