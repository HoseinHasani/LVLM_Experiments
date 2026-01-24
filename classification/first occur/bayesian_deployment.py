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

        scaler = torch.load(scaler_path, map_location=self.device, weights_only=True)
        self.mean = scaler["mean"].to(self.device)
        self.std = scaler["std"].to(self.device)

        self.input_dim = self.mean.numel()

        prior_net = PriorRTNet()
        prior_net.load_state_dict(torch.load(prior_path, map_location=self.device, weights_only=True))
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

        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
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
        samples: list of dicts with this structure:
            
        sample_dict = {
            "sentence_idx": int,
            "token_idx": int,
            "topk_attn": (L,H,K),
            "topk_attn_next": (L,H,K),
            "logits": np.ndarray,
            "total_reps": int,
            "rep_num": int,
            "token_id": int,
            "token": str,
            "is_first_occ": bool,
            "lemma": str,
            "all_occurrence_indices": list[int],
        }
        
        
    
        returns: list of dicts (same structure, enriched with predictions):
            
        {
            ... original fields except heavy tensors ...,
            "prior_probs": [p0, p1],
            "raw_probs": [p0, p1],
            "posterior_probs": [p0, p1],
            "final_pred": int
        }
        """
    
        if not samples:
            return []
    
        # --------------------------------------------------
        # Collect first-occurrence samples
        # --------------------------------------------------
        first_occ_samples = []
        for s in samples:
            if s.get("is_first_occ", False):
                first_occ_samples.append(s)
    
        # --------------------------------------------------
        # Feature extraction (ONLY first occurrences)
        # --------------------------------------------------
        feats = []
        keys = []  # (sentence_idx, token_idx)
    
        for s in first_occ_samples:
            idx = int(np.clip(s["token_idx"], self.min_position, self.max_position))
    
            feats.append(
                self.extract_features(
                    token_idx=idx,
                    topk_attn=s["topk_attn"],
                    topk_attn_next=s["topk_attn_next"],
                    logits=s["logits"],
                    total_reps=s["total_reps"],
                )
            )
            keys.append((s["sentence_idx"], s["token_idx"]))
    
        if feats:
            X = torch.tensor(np.stack(feats), device=self.device)
            X = (X - self.mean) / (self.std + 1e-6)
    
            with torch.no_grad():
                raw_post, bayes_post, bayes_pred = self.model(X)
        else:
            raw_post = bayes_post = bayes_pred = None
    
        # --------------------------------------------------
        # Build FULL prediction table via first occurrences
        # --------------------------------------------------
        pred_table = {}  # (sentence_idx, token_idx) -> prediction
    
        for i, s in enumerate(first_occ_samples):
            sent_idx = s["sentence_idx"]
    
            pred = {
                "raw_probs": raw_post[i].cpu().tolist(),
                "prior_probs": (bayes_post[i] / raw_post[i]).cpu().tolist(),
                "posterior_probs": bayes_post[i].cpu().tolist(),
                "final_pred": int(bayes_pred[i].item()),
            }
    
            all_occurrence_indices = s["all_occurrence_indices"]
            if len(all_occurrence_indices) < 1:
                all_occurrence_indices = [s["token_idx"]]
            # Assign to ALL occurrences listed by the first-occ token
            for tok_idx in all_occurrence_indices:
                pred_table[(sent_idx, tok_idx)] = pred
    
        # --------------------------------------------------
        # Build outputs (order preserved)
        # --------------------------------------------------
        outputs = []
    
        for s in samples:
            key = (s["sentence_idx"], s["token_idx"])
            pred = pred_table.get(key)
    
            out = dict(s)
            out.pop("topk_attn", None)
            out.pop("topk_attn_next", None)
            out.pop("logits", None)
    
            if pred is not None:
                out.update(pred)
            else:
                out.update({
                    "raw_probs": None,
                    "prior_probs": None,
                    "posterior_probs": None,
                    "final_pred": None,
                })
    
            outputs.append(out)
    
        return outputs


    


def load_real_attention_samples(
    attn_dir,
    score_dir,
    n_samples=100,
    n_layers=32,
    n_heads=32,
    n_top_k=20,
):
    attn_files = sorted(glob(os.path.join(attn_dir, "attentions_*.pkl")))
    score_files = sorted(glob(os.path.join(score_dir, "scores_*.pkl")))

    samples = []

    attn_files_dict = {
        int(os.path.splitext(os.path.basename(f))[0].split("_")[1]): f
        for f in attn_files
    }
    score_files_dict = {
        int(os.path.splitext(os.path.basename(f))[0].split("_")[1]): f
        for f in score_files
    }

    intersect_ids = sorted(set(attn_files_dict) & set(score_files_dict))

    for id_ in intersect_ids[:n_samples]:
        try:
            with open(attn_files_dict[id_], "rb") as f:
                attn_data = pickle.load(f)
            with open(score_files_dict[id_], "rb") as f:
                score_data = pickle.load(f)
        except Exception as e:
            print(f"Skipping sentence {id_}: {e}")
            continue

        for cls_name, label in [("fp", 0), ("tp", 1)]:
            attn_entries = attn_data.get(cls_name, {}).get("image", [])
            score_entries = score_data.get(cls_name, [])
            meta_entries = attn_data.get(cls_name, {}).get("meta", [])

            for k in range(min(len(attn_entries), len(score_entries), len(meta_entries))):
                e = attn_entries[k]
                meta = meta_entries[k]

                if not e.get("subtoken_results"):
                    continue

                sub = e["subtoken_results"][0]

                idx = int(sub.get("idx", -1))
                if idx < 0:
                    continue

                topk_vals = np.array(sub.get("topk_values"), dtype=float)
                topk_vals_next = np.array(sub.get("topk_values_next"), dtype=float)

                if topk_vals.ndim != 3 or topk_vals_next.ndim != 3:
                    continue

                topk_vals = topk_vals[..., :n_top_k]
                topk_vals_next = topk_vals_next[..., :n_top_k]

                logits = np.array(score_entries[k][1], dtype=float)

                total_reps = meta.get("total_reps", 1)
                rep_num = meta.get("rep_num", 1)
                all_occurrence_indices = meta.get("all_occurrence_indices", [idx])
                is_first_occ = meta.get("is_first_occ", False)

                token_id = meta.get("token_id", -1)
                lemma = meta.get("lemma", None)

                sample = {
                    "sentence_idx": id_,
                    "token_idx": idx,
                    "topk_attn": topk_vals,
                    "topk_attn_next": topk_vals_next,
                    "logits": logits,
                    "total_reps": total_reps,
                    "rep_num": rep_num,
                    "token_id": token_id,
                    "token": None,
                    "lemma": lemma,
                    "is_first_occ": is_first_occ,
                    "all_occurrence_indices": all_occurrence_indices,
                }

                samples.append((sample, label))

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




def evaluate_on_real_data(classifier, attn_dir, score_dir, n_eval=100):
    data = load_real_attention_samples(
        attn_dir,
        score_dir,
        n_samples=n_eval,
        n_layers=classifier.n_layers,
        n_heads=classifier.n_heads,
        n_top_k=classifier.n_top_k,
    )

    if not data:
        print("No samples loaded.")
        return

    samples = [s for s, _ in data]
    labels = {(s["sentence_idx"], s["token_idx"]): y for s, y in data}

    preds = classifier.predict(samples)

    y_true, y_pred = [], []

    for out in preds:
        if out['final_pred'] is None:
            continue

        key = (out["sentence_idx"], out["token_idx"])
        if key not in labels:
            continue

        y_true.append(labels[key])
        y_pred.append(out["posterior_probs"][1])

    if not y_true:
        print("No valid first-occurrence samples for evaluation.")
        return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_bin = (y_pred > 0.5).astype(int)

    acc = accuracy_score(y_true, y_bin)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_bin, average="binary"
    )

    print(f"\n=== Bayesian Evaluation ({len(y_true)} samples) ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

    plot_confusion_matrix(
        y_true,
        y_bin,
        classes=["FP", "TP"],
        normalize=False,
        title="Bayesian Confusion Matrix",
    )

    return dict(accuracy=acc, precision=precision, recall=recall, f1=f1)



    
if __name__ == "__main__":
    classifier = BayesianInference(
        model_path="final_cl_results_r/exp__bayes_ce/model/pytorch_mlp.pt",
        prior_path="priors_r/prior_rt_mlp.pt",
        scaler_path="final_cl_results_r/exp__bayes_ce/model/scaler.pt",
        n_layers=32,
        n_heads=32,
        use_entropy=True,
        use_logits=True,
    )

    dummy_samples = [
        {
            "sentence_idx": 0,
            "token_idx": 10,
            "topk_attn": np.random.rand(32, 32, 20),
            "topk_attn_next": np.random.rand(32, 32, 20),
            "logits": np.random.randn(500),
            "total_reps": 2,
            "rep_num": 1,
            "token_id": 123,
            "token": "dummy",
            "lemma": "dummy",
            "is_first_occ": True,
            "all_occurrence_indices": [10, 25],
        },
        {
            "sentence_idx": 0,
            "token_idx": 25,
            "topk_attn": None,
            "topk_attn_next": None,
            "logits": None,
            "total_reps": 2,
            "rep_num": 2,
            "token_id": 123,
            "token": "dummy",
            "lemma": "dummy",
            "is_first_occ": False,
            "all_occurrence_indices": [25],
        },
    ]

    preds = classifier.predict(dummy_samples)
    print(preds)
    print("\nDummy predictions:")
    for p in preds:
        print(
            f"[sent={p['sentence_idx']} idx={p['token_idx']}] "
            f"raw={p['raw_probs']} "
            f"post={p['posterior_probs']} "
            f"pred={p['final_pred']}"
        )

    if False:
        # Real data eval
        attn_dir = "../../data/all layers all attention tp fp rep double r"
        score_dir = "../../data/double_scores r"
    
        evaluate_on_real_data(classifier, attn_dir, score_dir, n_eval=3000)


