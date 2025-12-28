import os
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict

# -----------------------------
# Configuration
# -----------------------------

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

attn_dir = "../../data/all layers all attention tp fp rep double"
score_dir = "../../data/double_scores"
base_save_dir = "final_cl_results"
os.makedirs(base_save_dir, exist_ok=True)
dataset_path = f"cls_data_{target_rep}"

# -----------------------------
# Helper Functions
# -----------------------------

def extract_position_occurrence_features(X_all, pos_all, cls_all, target_rep_all):
    """Extract position and occurrence features."""
    X_pos_rep = []
    y_pos_rep = []
    for pos, cls, rep in zip(pos_all, cls_all, target_rep_all):
        # We take token position and the number of occurrences (target_rep) as features
        features = [pos, rep]
        
        X_pos_rep.append(features)
        y_pos_rep.append(cls)  # FP = 0, TP = 1
    return np.array(X_pos_rep), np.array(y_pos_rep)

class SimpleMLPClassifier(nn.Module):
    """A simple MLP for predicting FP or TP based on position and target_rep."""
    def __init__(self, input_dim, hidden1=64, hidden2=32, dropout_rate=0.5):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 2)  # Output: FP or TP
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Load Data or Extract Features
# -----------------------------
# Load previously extracted data (or extract features)
if dataset_path and os.path.exists(f"{dataset_path}/x.npy"):
    print("Loading saved dataset...")
    X_all = np.load(f"{dataset_path}/x.npy")
    y_all = np.load(f"{dataset_path}/y.npy")
    pos_all = np.load(f"{dataset_path}/pos.npy")
    cls_all = np.load(f"{dataset_path}/cls.npy")
else:
    # Code for extracting features (this part can be reused from your original script)
    pass

# Extract the features based on position and target_rep (no attention, no entropy)
target_rep_all = np.array([target_rep] * len(pos_all))  # Example of target_rep
X_pos_rep, y_pos_rep = extract_position_occurrence_features(X_all, pos_all, cls_all, target_rep_all)

# Convert to torch tensors
X_pos_rep_t = torch.tensor(X_pos_rep, dtype=torch.float32)
y_pos_rep_t = torch.tensor(y_pos_rep, dtype=torch.long)

# Create a train/test split
n_train = int(len(X_pos_rep) * train_size)
n_test = int(len(X_pos_rep) * test_size)

X_train_pos, X_test_pos = X_pos_rep_t[:n_train], X_pos_rep_t[-n_test:]
y_train_pos, y_test_pos = y_pos_rep_t[:n_train], y_pos_rep_t[-n_test:]

train_loader_pos = DataLoader(TensorDataset(X_train_pos, y_train_pos), batch_size=64, shuffle=True)
test_loader_pos = DataLoader(TensorDataset(X_test_pos, y_test_pos), batch_size=128, shuffle=False)

# -----------------------------
# Train MLP Classifier for Position and Occurrence
# -----------------------------

# Instantiate the model
pos_clf = SimpleMLPClassifier(input_dim=2).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pos_clf.parameters(), lr=1e-3)

train_losses_pos, test_losses_pos = [], []

print("\nTraining position-occurrence based MLP...")

for epoch in range(n_epochs):
    # ------------------------- Training -------------------------
    pos_clf.train()
    running_loss = 0.0

    for xb, yb in train_loader_pos:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = pos_clf(xb)

        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)

    train_loss = running_loss / len(train_loader_pos.dataset)
    train_losses_pos.append(train_loss)

    # ------------------------- Validation -------------------------
    pos_clf.eval()
    val_loss = 0.0
    correct_pos, total = 0, 0

    with torch.no_grad():
        for xb, yb in test_loader_pos:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = pos_clf(xb)

            loss = criterion(logits, yb)
            val_loss += loss.item() * xb.size(0)

            # Calculate accuracy
            pred = torch.argmax(logits, dim=1)
            correct_pos += (pred == yb).sum().item()
            total += yb.size(0)

    val_loss /= len(test_loader_pos.dataset)
    test_losses_pos.append(val_loss)

    print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {correct_pos/total:.3f}")

# -----------------------------
# Save the Model
# -----------------------------

# Save the trained MLP model
torch.save(pos_clf.state_dict(), os.path.join(base_save_dir, "simple_mlp_classifier.pt"))
print("Position-occurrence classifier model saved.\n")

# -----------------------------
# Evaluate Model and Get Probabilities
# -----------------------------
pos_clf.eval()
y_probs_pos, y_pred_pos, y_true_pos = [], [], []

with torch.no_grad():
    for xb, yb in test_loader_pos:
        xb = xb.to(device)

        logits = pos_clf(xb)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Get TP probabilities (index 1)
        
        y_probs_pos.extend(probs)
        y_pred_pos.extend(torch.argmax(logits, dim=1).cpu().numpy())
        y_true_pos.extend(yb.numpy())

# Save the predictions
np.save(f"{base_save_dir}/y_probs_pos.npy", y_probs_pos)
np.save(f"{base_save_dir}/y_pred_pos.npy", y_pred_pos)
np.save(f"{base_save_dir}/y_true_pos.npy", y_true_pos)

print("Model evaluation completed and results saved.\n")
