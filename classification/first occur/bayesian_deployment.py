import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from glob import glob
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


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


class BayesianMLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        prior_net,
        pos_norm_min,
        pos_norm_max,
        rep_norm_min,
        rep_norm_max,
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

        self.prior_net = prior_net

        self.pos_norm_min = pos_norm_min
        self.pos_norm_max = pos_norm_max
        self.rep_norm_min = rep_norm_min
        self.rep_norm_max = rep_norm_max

        self.min_position = min_position
        self.max_position = max_position

        self.min_repetition = 0
        self.max_repetition = 3

    def decode_positions(self, x):
        pos_norm = x[:, -1]
        pos_real = (
            (pos_norm - self.pos_norm_min)
            / (self.pos_norm_max - self.pos_norm_min)
        ) * (self.max_position - self.min_position) + self.min_position
        return torch.clamp(torch.round(pos_real), self.min_position, self.max_position).long()

    def decode_repetitions(self, x):
        rep_norm = x[:, 0]
        rep_real = (
            (rep_norm - self.rep_norm_min)
            / (self.rep_norm_max - self.rep_norm_min)
        ) * (self.max_repetition - self.min_repetition)
        return torch.clamp(torch.round(rep_real), self.min_repetition, self.max_repetition).long()

    def forward(self, x):
        raw_logits = self.net(x)
        raw_post = torch.softmax(raw_logits, dim=1)

        pos_real = self.decode_positions(x)
        rep_real = self.decode_repetitions(x)

        pos_norm = (pos_real - self.min_position) / (self.max_position - self.min_position)
        rep_norm = rep_real / self.max_repetition

        rt = torch.stack([rep_norm, pos_norm], dim=1)

        prior_logits = self.prior_net(rt)
        priors = torch.softmax(prior_logits, dim=1)

        bayes_unnorm = raw_post * priors
        bayes_post = bayes_unnorm / (bayes_unnorm.sum(dim=1, keepdim=True) + 1e-8)
        bayes_pred = torch.argmax(bayes_post, dim=1)

        return raw_post, bayes_post, bayes_pred


class BayesianMLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        prior_net,
        pos_norm_min,
        pos_norm_max,
        rep_norm_min,
        rep_norm_max,
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

        self.prior_net = prior_net

        self.pos_norm_min = pos_norm_min
        self.pos_norm_max = pos_norm_max
        self.rep_norm_min = rep_norm_min
        self.rep_norm_max = rep_norm_max

        self.min_position = min_position
        self.max_position = max_position

        self.min_repetition = 0
        self.max_repetition = 3

    def decode_positions(self, x):
        pos_norm = x[:, -1]
        pos_real = (
            (pos_norm - self.pos_norm_min)
            / (self.pos_norm_max - self.pos_norm_min)
        ) * (self.max_position - self.min_position) + self.min_position
        return torch.clamp(torch.round(pos_real), self.min_position, self.max_position).long()

    def decode_repetitions(self, x):
        rep_norm = x[:, 0]
        rep_real = (
            (rep_norm - self.rep_norm_min)
            / (self.rep_norm_max - self.rep_norm_min)
        ) * (self.max_repetition - self.min_repetition)
        return torch.clamp(torch.round(rep_real), self.min_repetition, self.max_repetition).long()

    def forward(self, x):
        raw_logits = self.net(x)
        raw_post = torch.softmax(raw_logits, dim=1)

        pos_real = self.decode_positions(x)
        rep_real = self.decode_repetitions(x)

        pos_norm = (pos_real - self.min_position) / (self.max_position - self.min_position)
        rep_norm = rep_real / self.max_repetition

        rt = torch.stack([rep_norm, pos_norm], dim=1)

        prior_logits = self.prior_net(rt)
        priors = torch.softmax(prior_logits, dim=1)

        bayes_unnorm = raw_post * priors
        bayes_post = bayes_unnorm / (bayes_unnorm.sum(dim=1, keepdim=True) + 1e-8)
        bayes_pred = torch.argmax(bayes_post, dim=1)

        return raw_post, bayes_post, bayes_pred



class BayesianInference:
    def __init__(
        self,
        model_path,
        prior_path,
        scaler_path,
        n_layers=32,
        n_heads=32,
        n_top_k=20,
        min_position=5,
        max_position=155,
        use_attns=True,
        use_logits=True,
        use_entropy=True,
        dropout_rate=0.5,
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_top_k = n_top_k
        self.min_position = min_position
        self.max_position = max_position
        self.use_attns = use_attns
        self.use_logits = use_logits
        self.use_entropy = use_entropy

        scaler = torch.load(scaler_path, map_location=self.device)
        self.mean = scaler["mean"].to(self.device)
        self.std = scaler["std"].to(self.device)

        self.input_dim = self.mean.numel()

        prior_net = PriorRTNet()
        prior_net.load_state_dict(torch.load(prior_path, map_location=self.device))
        prior_net.eval()
        for p in prior_net.parameters():
            p.requires_grad = False

        self.model = BayesianMLPClassifier(
            input_dim=self.input_dim,
            prior_net=prior_net,
            pos_norm_min=float(self.mean[0, -1]),
            pos_norm_max=float(self.mean[0, -1] + self.std[0, -1]),
            rep_norm_min=float(self.mean[0, 0]),
            rep_norm_max=float(self.mean[0, 0] + self.std[0, 0]),
            min_position=min_position,
            max_position=max_position,
            dropout_rate=dropout_rate,
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    # ---------- feature utils ----------

    @staticmethod
    def compute_entropy(vals, eps=1e-10):
        p = vals / (np.sum(vals) + eps)
        return -np.sum(p * np.log(p + eps))

    @staticmethod
    def softmax(x, eps=1e-10):
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / (np.sum(ex) + eps)

    @staticmethod
    def logit_entropy(logits, k=100, eps=1e-10):
        topk = np.partition(logits, -k)[-k:]
        p = np.exp(topk - np.max(topk))
        p /= p.sum() + eps
        return -np.sum(p * np.log(p + eps))

    # ---------- feature extraction ----------

    def extract_features(
        self,
        token_idx,
        topk_attn,
        topk_attn_next,
        logits,
        total_reps,
    ):
        feats = []

        n_repeat = min(total_reps - 1, 3)
        feats.append(n_repeat)

        if self.use_attns:
            for l in range(self.n_layers):
                for h in range(self.n_heads):
                    v1 = topk_attn[l, h, :self.n_top_k]
                    v2 = topk_attn_next[l, h, :self.n_top_k]

                    feats.append(np.mean(v1))
                    feats.append(np.mean(v2))

                    if self.use_entropy:
                        feats.append(self.compute_entropy(v1))
                        feats.append(self.compute_entropy(v2))

        if self.use_logits:
            feats.extend([
                self.logit_entropy(logits),
                float(np.max(logits)),
                float(np.max(self.softmax(logits))),
            ])

        feats.append(token_idx / self.max_position)
        return np.array(feats, dtype=np.float32)

    # ---------- inference ----------

    def predict(self, samples):
        """
        samples: list of dicts with keys:
          {
            "token_idx": int,
            "topk_attn": (L,H,K),
            "topk_attn_next": (L,H,K),
            "logits": np.ndarray,
            "total_reps": int
          }
        """
        feats, token_ids = [], []

        for s in samples:
            idx = s["token_idx"]
            if idx < self.min_position or idx > self.max_position:
                continue

            feats.append(
                self.extract_features(
                    idx,
                    s["topk_attn"],
                    s["topk_attn_next"],
                    s["logits"],
                    s["total_reps"],
                )
            )
            token_ids.append(idx)

        if not feats:
            return {}

        X = torch.tensor(np.stack(feats), device=self.device)
        X = (X - self.mean) / (self.std + 1e-6)

        with torch.no_grad():
            raw_post, bayes_post, bayes_pred = self.model(X)

        results = {}
        for i, idx in enumerate(token_ids):
            results[idx] = {
                "raw_tp_prob": float(raw_post[i, 1]),
                "bayes_tp_prob": float(bayes_post[i, 1]),
                "bayes_pred": int(bayes_pred[i]),
            }

        return results


    


def load_real_attention_samples(data_dir, n_samples=100, n_layers=32, n_heads=32, n_top_k=20):
    files = sorted(glob(os.path.join(data_dir, "attentions_*.pkl")))
    samples = []

    for fpath in files[:n_samples]:
        try:
            with open(fpath, "rb") as handle:
                data = pickle.load(handle)
        except Exception as e:
            print(f"Skipping {fpath}: {e}")
            continue

        for cls_name, label in [("fp", 1), ("tp", 0), ("other", 0)]:
            entries = data.get(cls_name, {}).get("image", [])
            for e in entries:
                if not e.get("subtoken_results"):
                    continue
                for sub in e["subtoken_results"][:1]:
                    topk_vals = np.array(sub.get("topk_values"), dtype=float)
                    if topk_vals.ndim != 3:
                        continue
                    idx = int(sub.get("idx", -1))
                    if idx < 0:
                        continue

                    topk_vals = topk_vals[..., :n_top_k]
                    attn_dict = {idx: topk_vals}
                    samples.append((attn_dict, label))
    return samples



def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()




def evaluate_on_real_data(classifier, data_dir, n_eval=100):
    samples = load_real_attention_samples(data_dir, n_samples=n_eval,
                                          n_layers=classifier.n_layers,
                                          n_heads=classifier.n_heads)

    y_true, y_pred = [], []

    for attn_dict, label in tqdm(samples):
        preds = classifier.predict(attn_dict)
        if not preds:
            continue
        prob = list(preds.values())[0]
        y_true.append(label)
        y_pred.append(prob)

    if not y_true:
        print("No valid samples found for evaluation.")
        return

    print(np.max(y_pred), np.min(y_pred), np.mean(y_pred), np.std(y_pred))

    y_bin = (np.array(y_pred) > 0.5).astype(int)
    acc = accuracy_score(y_true, y_bin)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_bin, average="binary"
    )

    print(f"\n=== Evaluation on {n_eval} Data ===")
    print(f"Samples:   {len(y_true)}")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

    plot_confusion_matrix(y_true, y_bin, classes=['Not Target', 'Target'], normalize=False)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


    
if __name__ == "__main__":
    
    classifier = FPAttentionClassifier(
        model_path="results_all_layers/pytorch_mlp_exp__ent1_gin0/model/pytorch_mlp_with_l2.pt",
        scaler_path="results_all_layers/pytorch_mlp_exp__ent1_gin0/model/scaler.pt",
        n_layers=32,
        n_heads=32,
        use_entropy=True,
        use_gini=False,
    )

    # Dummy attention data for testing
    sample = {
        0: np.random.rand(32, 32, 20),
        3: np.random.rand(32, 32, 20),
        10: np.random.rand(32, 32, 20),
        20: np.random.rand(32, 32, 20),
    }

    preds = classifier.predict(sample)
    print("\nPredictions for a dummy sample:")
    for idx, prob in preds.items():
        print(f"Token {idx:>3}: {prob:.4f}")
        
    if False:
        data_dir = "data/all layers all attention tp fp"
        print("\nEvaluating classifier on real dataset samples...")
        results = evaluate_on_real_data(classifier, data_dir, n_eval=3000)
