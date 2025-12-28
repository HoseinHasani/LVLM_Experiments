import os
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# Configuration
# -----------------------------

attn_dir = "../../data/all layers all attention tp fp rep double"
score_dir = "../../data/double_scores"  # only used for alignment
save_dir = "prior_position_rep"
os.makedirs(save_dir, exist_ok=True)

min_position = 5
max_position = 155

train_ratio = 0.7
batch_size = 256
n_epochs = 10
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# -----------------------------
# Data Extraction
# -----------------------------

def extract_position_rep_labels(attn_files_dict, score_files_dict, file_ids):
    """
    Extract (position, rep_num, label) per token.
    Label: FP = 0, TP = 1
    """
    X = []
    y = []

    for fid in tqdm(file_ids, desc="Extracting position/rep"):
        try:
            with open(attn_files_dict[fid], "rb") as f:
                attn_data = pickle.load(f)
        except Exception:
            continue

        for cls_name, label in [("fp", 0), ("tp", 1)]:
            entries = attn_data.get(cls_name, {}).get("image", [])

            for e in entries:
                if len(e["token_indices"]) == 0:
                    continue

                pos = int(e["token_indices"][0])
                rep = int(e["rep_num"])

                if pos < min_position or pos > max_position:
                    continue

                X.append([pos, rep])
                y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y


# -----------------------------
# Load file lists
# -----------------------------

attn_files = sorted(glob(os.path.join(attn_dir, "attentions_*.pkl")))
score_files = sorted(glob(os.path.join(score_dir, "scores_*.pkl")))

attn_files_dict = {
    int(os.path.basename(f).split("_")[1].split(".")[0]): f
    for f in attn_files
}
score_files_dict = {
    int(os.path.basename(f).split("_")[1].split(".")[0]): f
    for f in score_files
}

file_ids = sorted(set(attn_files_dict) & set(score_files_dict))
print("Found files:", len(file_ids))

# -----------------------------
# Extract dataset
# -----------------------------

X, y = extract_position_rep_labels(attn_files_dict, score_files_dict, file_ids)

print("Total samples:", len(X))
print("TP:", np.sum(y == 1), "FP:", np.sum(y == 0))

np.save(os.path.join(save_dir, "X_raw.npy"), X)
np.save(os.path.join(save_dir, "y.npy"), y)

# -----------------------------
# Normalize features
# -----------------------------

# Normalize position to [0, 1]
X[:, 0] = (X[:, 0] - min_position) / (max_position - min_position)

# Normalize rep (log-scale is usually better)
X[:, 1] = np.log1p(X[:, 1])

# -----------------------------
# Train / test split
# -----------------------------

perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

n_train = int(train_ratio * len(X))
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train)
    ),
    batch_size=batch_size,
    shuffle=True,
)

test_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_test), torch.tensor(y_test)
    ),
    batch_size=batch_size,
    shuffle=False,
)

# -----------------------------
# Model
# -----------------------------

class PositionRepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


model = PositionRepMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# -----------------------------
# Training
# -----------------------------

for epoch in range(n_epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    train_loss = total_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = torch.argmax(model(xb), dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    acc = correct / total

    print(
        f"Epoch {epoch+1}/{n_epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Acc: {acc:.4f}"
    )

# -----------------------------
# Save model
# -----------------------------

torch.save(
    {
        "model": model.state_dict(),
        "min_position": min_position,
        "max_position": max_position,
    },
    os.path.join(save_dir, "position_rep_prior.pt"),
)

print("Prior model saved.")
