import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import argparse
import os
import numpy as np


def load_item_embeddings(dataset_path):
    emb_path = os.path.join(dataset_path, "process_data", "item_embeddings.pt")
    data = torch.load(emb_path, map_location="cpu")
    embeddings = data["embeddings"]
    titles = data["item_titles"]
    title2idx = {title: idx for idx, title in enumerate(titles)}
    return embeddings, titles, title2idx


def load_popularity_data(dataset_path, alpha):
    with open(os.path.join(dataset_path, "process_data", "popularity.json"), "r") as f:
        popularity_data = json.load(f)
        p_data = {title: info["p_data"] for title, info in popularity_data.items()}

    p_target_file = f"p_target_alpha_{alpha:.2f}.json"
    with open(os.path.join(dataset_path, "process_data", p_target_file), "r") as f:
        p_target_data = json.load(f)
        p_target = p_target_data["p_target"]

    return p_data, p_target


def build_negative_pools(p_data, p_target, all_titles):
    overpopular_pool = []
    for title in all_titles:
        if title in p_data and title in p_target and p_target[title] > 0:
            ratio = p_data[title] / p_target[title]
            if ratio > 1.0:
                overpopular_pool.append(title)

    all_items_set = set(all_titles)
    return overpopular_pool, all_items_set


def find_hard_negative(chosen_title, candidate_pool, embeddings, title2idx):
    if chosen_title not in title2idx or len(candidate_pool) == 0:
        return None

    chosen_emb = embeddings[title2idx[chosen_title]]

    candidate_indices = []
    candidate_titles = []
    for title in candidate_pool:
        if title in title2idx:
            candidate_indices.append(title2idx[title])
            candidate_titles.append(title)

    if len(candidate_indices) == 0:
        return None

    candidate_embs = embeddings[candidate_indices]
    similarities = F.cosine_similarity(chosen_emb.unsqueeze(0), candidate_embs, dim=-1)
    max_idx = torch.argmax(similarities).item()
    hard_negative = candidate_titles[max_idx]
    return hard_negative


def compute_balance_weight(y_w, y_l, p_data, p_target, gamma=1.0, alpha=0.8):
    import math

    p_data_w = p_data.get(y_w, 1e-10)
    p_data_l = p_data.get(y_l, 1e-10)
    p_target_w = p_target.get(y_w, 1e-10)
    p_target_l = p_target.get(y_l, 1e-10)

    calibration_w = p_target_w / p_data_w
    calibration_l = p_target_l / p_data_l
    delta = calibration_w - calibration_l
    weight = 1.0 + alpha * math.tanh(gamma * delta)
    return weight


def generate_dpo_pairs(
    dataset_path,
    alpha_ptarget,
    gamma=1.0,
    alpha_weight=0.8,
    random_ratio=0.0,
    overpopular_ratio=0.5,
    hard_ratio=0.5,
    use_weights=True,
    device="cuda",
):
    total_ratio = random_ratio + overpopular_ratio + hard_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValueError(f"Negative ratios must sum to 1.0, got {total_ratio:.4f}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    embeddings, titles, title2idx = load_item_embeddings(dataset_path)
    embeddings = embeddings.to(device)

    p_data, p_target = load_popularity_data(dataset_path, alpha_ptarget)
    overpopular_pool, all_items_set = build_negative_pools(p_data, p_target, titles)

    with open(os.path.join(dataset_path, "process_data", "train.json"), "r") as f:
        train_data = json.load(f)

    dpo_pairs = []
    skipped = 0
    neg_type_counts = {"random": 0, "overpopular": 0, "hard": 0}
    weights = []

    neg_types = []
    neg_probs = []
    if random_ratio > 0:
        neg_types.append("random")
        neg_probs.append(random_ratio)
    if overpopular_ratio > 0:
        neg_types.append("overpopular")
        neg_probs.append(overpopular_ratio)
    if hard_ratio > 0:
        neg_types.append("hard")
        neg_probs.append(hard_ratio)

    import re

    for sample in tqdm(train_data):
        prompt = sample["instruction"] + sample["input"]
        chosen = sample["output"].strip().strip('"')

        input_text = sample["input"]
        history_titles = re.findall(r'"([^"]+)"', input_text)
        history_set = set(history_titles)

        candidate_negatives = list(all_items_set - history_set - {chosen})
        if len(candidate_negatives) == 0:
            skipped += 1
            continue

        neg_type = np.random.choice(neg_types, p=neg_probs)
        rejected = None

        if neg_type == "random":
            rejected = np.random.choice(candidate_negatives)
            neg_type_counts["random"] += 1
        elif neg_type == "overpopular":
            overpopular_candidates = list(set(overpopular_pool) & set(candidate_negatives))
            if len(overpopular_candidates) > 0:
                rejected = np.random.choice(overpopular_candidates)
                neg_type_counts["overpopular"] += 1
            else:
                rejected = np.random.choice(candidate_negatives)
                neg_type_counts["random"] += 1
        elif neg_type == "hard":
            rejected = find_hard_negative(
                chosen, candidate_negatives, embeddings, title2idx
            )
            if rejected is None:
                rejected = np.random.choice(candidate_negatives)
                neg_type_counts["random"] += 1
            else:
                neg_type_counts["hard"] += 1

        if rejected is None:
            skipped += 1
            continue

        if use_weights:
            weight = compute_balance_weight(
                chosen, rejected, p_data, p_target, gamma, alpha_weight
            )
        else:
            weight = 1.0
        weights.append(weight)

        dpo_pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "weight": float(weight),
                "neg_type": neg_type,
                "calibration_chosen": float(
                    p_target.get(chosen, 1e-10) / p_data.get(chosen, 1e-10)
                ),
                "calibration_rejected": float(
                    p_target.get(rejected, 1e-10) / p_data.get(rejected, 1e-10)
                ),
            }
        )

    _ = (skipped, neg_type_counts, weights)
    return dpo_pairs


def save_dpo_pairs(dpo_pairs, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in dpo_pairs:
            json.dump(pair, f, ensure_ascii=False)
            f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MovieLens1M")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--alpha_weight", type=float, default=0.8)

    parser.add_argument("--random_ratio", type=float, default=0.0)
    parser.add_argument("--overpopular_ratio", type=float, default=0.5)
    parser.add_argument("--hard_ratio", type=float, default=0.5)

    parser.add_argument("--use_standard_dpo", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="dpo_data")

    args = parser.parse_args()

    dataset_path = f"mydata/{args.dataset}"

    required_files = [
        os.path.join(dataset_path, "process_data", "item_embeddings.pt"),
        os.path.join(dataset_path, "process_data", "popularity.json"),
        os.path.join(
            dataset_path,
            "process_data",
            f"p_target_alpha_{args.alpha:.2f}.json",
        ),
        os.path.join(dataset_path, "process_data", "train.json"),
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        return

    dpo_pairs = generate_dpo_pairs(
        dataset_path=dataset_path,
        alpha_ptarget=args.alpha,
        gamma=args.gamma,
        alpha_weight=args.alpha_weight,
        random_ratio=args.random_ratio,
        overpopular_ratio=args.overpopular_ratio,
        hard_ratio=args.hard_ratio,
        use_weights=(not args.use_standard_dpo),
        device=args.device,
    )

    if args.random_ratio == 1.0:
        neg_suffix = "_random"
    elif args.overpopular_ratio == 1.0:
        neg_suffix = "_overpop"
    elif args.hard_ratio == 1.0:
        neg_suffix = "_hard"
    elif (
        args.random_ratio == 0.0
        and args.overpopular_ratio == 0.5
        and args.hard_ratio == 0.5
    ):
        neg_suffix = ""
    else:
        neg_suffix = (
            f"_r{int(args.random_ratio*100)}_o{int(args.overpopular_ratio*100)}_h{int(args.hard_ratio*100)}"
        )

    if args.use_standard_dpo:
        output_path = os.path.join(
            args.output_dir,
            args.dataset,
            f"dpo_pairs_alpha_{args.alpha:.2f}{neg_suffix}_standard.jsonl",
        )
    else:
        output_path = os.path.join(
            args.output_dir,
            args.dataset,
            f"dpo_pairs_alpha_{args.alpha:.2f}{neg_suffix}_aw_{args.alpha_weight:.2f}.jsonl",
        )

    save_dpo_pairs(dpo_pairs, output_path)


if __name__ == "__main__":
    main()
