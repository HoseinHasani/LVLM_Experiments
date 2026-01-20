import json
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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


def train_val_split(X, y, train_ratio=0.8, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))

    n_train = int(len(X) * train_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx]
    )


class PriorRTNet(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, x):
        return self.net(x)


def train_prior_net(
    X_train, y_train,
    X_val, y_val,
    device,
    n_epochs=20,
    batch_size=256,
):
    model = PriorRTNet(hidden=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train).to(device),
            torch.tensor(y_train).to(device),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val).to(device),
            torch.tensor(y_val).to(device),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    for epoch in range(n_epochs):
        # -------- TRAIN --------
        model.train()
        train_loss, train_correct, n_train = 0, 0, 0

        for xb, yb in train_loader:
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            train_correct += (logits.argmax(1) == yb).sum().item()
            n_train += xb.size(0)

        # -------- VALIDATION --------
        model.eval()
        val_loss, val_correct, n_val = 0, 0, 0

        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss += loss.item() * xb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                n_val += xb.size(0)

        print(
            f"[Epoch {epoch+1:02d}] "
            f"Train Loss={train_loss/n_train:.4f} "
            f"Acc={train_correct/n_train:.3f} | "
            f"Val Loss={val_loss/n_val:.4f} "
            f"Acc={val_correct/n_val:.3f}"
        )

    return model


if __name__ == "__main__":
    METADATA_DIR = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/rz15dire/Ensemble/experiments/new_per_sample_outputs/gpt4o_meta_data"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_prior_metadata(
        metadata_dir=METADATA_DIR,
        min_position=5,
        max_position=155,
    )

    print(f"Loaded {len(y)} samples | TP={np.sum(y==1)} FP={np.sum(y==0)}")

    X_tr, y_tr, X_va, y_va = train_val_split(X, y)

    prior_net = train_prior_net(
        X_tr, y_tr,
        X_va, y_va,
        device=device,
        n_epochs=20,
    )

    torch.save(prior_net.state_dict(), "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/rz15dire/Ensemble/experiments/eval_new/gpt40_experiments/prior_rt_from_metadata.pt")


