import torch
import torch.nn as nn
import numpy as np

import os
import json
from glob import glob
from tqdm import tqdm



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


def load_prior_model(ckpt_path, device):
    model = PriorRTNet(hidden=32).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def build_rt_features(words_meta, min_pos=5, max_pos=155, max_rep=3):
    feats = []
    positions = []

    for w in words_meta:
        t = w["position"]
        if t < min_pos or t > max_pos:
            continue

        r = min(w["total_reps"] - 1, max_rep)

        r_norm = r / max_rep
        t_norm = (t - min_pos) / (max_pos - min_pos)

        feats.append([r_norm, t_norm])
        positions.append(t)

    return np.array(feats, dtype=np.float32), positions


def predict_and_mark(
    caption,
    words_meta,
    model,
    device,
    threshold=0.5
):
    words = [w["word"] for w in words_meta]

    X, positions = build_rt_features(words_meta)
    if len(X) == 0:
        return {
            "fp_positions": [],
            "masked_caption": caption,
            "fp_marked_caption": caption
        }

    with torch.no_grad():
        logits = model(torch.tensor(X).to(device))
        probs = torch.softmax(logits, dim=-1)[:, 0]  # FP prob

    fp_positions = [
        positions[i] for i, p in enumerate(probs.cpu().numpy()) if p > threshold
    ]

    # --- Build masked & marked captions ---
    masked_words = []
    marked_words = []


    for w in words_meta:
        word = w["word"]
        pos = w["position"]

        if pos in fp_positions:
            marked_words.append(f"${word}")
            # masked_caption: remove FP word
        else:
            marked_words.append(word)
            masked_words.append(word)

    masked_caption = " ".join(masked_words)
    fp_marked_caption = " ".join(marked_words)

    return {
        "fp_positions": fp_positions,
        "masked_caption": masked_caption,
        "fp_marked_caption": fp_marked_caption
    }



def run_fp_marking(
    metadata_dir,
    output_dir,
    prior_ckpt,
    threshold=0.5
):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_prior_model(prior_ckpt, device)

    files = glob(os.path.join(metadata_dir, "image_id_*.json"))

    for f in tqdm(files, desc="FP marking"):
        with open(f, "r") as h:
            data = json.load(h)

        caption = data["caption"]
        words_meta = data["words_meta"]
        image_id = data["image_id"]

        res = predict_and_mark(
            caption=caption,
            words_meta=words_meta,
            model=model,
            device=device,
            threshold=threshold
        )

        out = {
            "image_id": image_id,
            "caption": caption,
            "fp_positions": res["fp_positions"],
            "masked_caption": res["masked_caption"],
            "fp_marked_caption": res["fp_marked_caption"]
        }

        out_path = os.path.join(output_dir, f"image_id_{image_id}.jsonl")
        with open(out_path, "w") as g:
            json.dump(out, g)
            g.write("\n")





if __name__ == "__main__":
    METADATA_DIR = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/rz15dire/Ensemble/experiments/new_per_sample_outputs/gpt4o_meta_data_greedy"
    OUTPUT_DIR   = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/rz15dire/Ensemble/experiments/new_per_sample_outputs/gpt4o_fp_marked_responses_greedy"
    PRIOR_CKPT   = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/rz15dire/Ensemble/experiments/eval_new/gpt40_experiments/prior_rt_from_metadata.pt"

    run_fp_marking(
        metadata_dir=METADATA_DIR,
        output_dir=OUTPUT_DIR,
        prior_ckpt=PRIOR_CKPT,
        threshold=0.1
    )

