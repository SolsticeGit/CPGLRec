import os
import json
import argparse
import math
import re
import pandas as pd
from tqdm import tqdm


def process_string(s):
    s_clean = re.sub(r"[^\w\s]", " ", s)
    words = s_clean.lower().split()
    words = [w for w in words if w.strip()]
    return words


def calculate_entropy(word_counts):
    if not word_counts:
        return 0.0

    total_words = sum(word_counts.values())
    if total_words == 0:
        return 0.0

    entropy = 0.0
    for count in word_counts.values():
        if count > 0:
            probability = count / total_words
            entropy += -probability * math.log2(probability)

    return entropy


def calculate_ttr(word_counts):
    if not word_counts:
        return 0.0

    num_unique_words = len(word_counts)
    total_words = sum(word_counts.values())

    if total_words == 0:
        return 0.0

    return num_unique_words / total_words


DATASET = "MovieLens1M"
ALPHA = 0.1
LAMBDA_VAL = 0.5
K_LIST = [1, 5, 10, 20]


def load_predictions(prediction_file):
    with open(prediction_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    return predictions


def load_p_target(data_dir, alpha):
    p_target_path = f"{data_dir}/p_target_alpha_{alpha:.2f}.json"

    if not os.path.exists(p_target_path):
        raise FileNotFoundError(f"P_target file not found: {p_target_path}")

    with open(p_target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["p_target"]


def compute_p_llm_from_predictions(predictions, num_beams=20):
    from collections import defaultdict

    item_prob_sum = defaultdict(float)

    for pred in tqdm(predictions):
        candidates = pred["candidates"]
        llm_scores = pred["llm_scores"]

        max_score = max(llm_scores)
        exp_scores = [math.exp(s - max_score) for s in llm_scores]
        sum_exp = sum(exp_scores)

        probs = [exp_s / sum_exp for exp_s in exp_scores]

        for cand, prob in zip(candidates, probs):
            cand_clean = cand.strip()
            item_prob_sum[cand_clean] += prob / len(predictions)

    total_prob = sum(item_prob_sum.values())
    p_llm = {item: prob / total_prob for item, prob in item_prob_sum.items()}
    return p_llm


def rerank_with_p_target(predictions, p_target, p_llm, lambda_val):
    reranked_predictions = []

    for pred in tqdm(predictions):
        candidates = pred["candidates"]
        llm_scores = pred["llm_scores"]

        final_scores = []
        for cand, log_p_llm_given_x in zip(candidates, llm_scores):
            cand_clean = cand.strip()
            log_p_target = math.log(p_target.get(cand_clean, 1e-10))
            log_p_llm = math.log(p_llm.get(cand_clean, 1e-10))
            calibration_term = lambda_val * (log_p_target - log_p_llm)
            s_final = log_p_llm_given_x + calibration_term
            final_scores.append(s_final)

        sorted_indices = sorted(
            range(len(final_scores)), key=lambda i: final_scores[i], reverse=True
        )

        reranked_candidates = [candidates[i].strip() for i in sorted_indices]
        reranked_llm_scores = [llm_scores[i] for i in sorted_indices]
        reranked_final_scores = [final_scores[i] for i in sorted_indices]

        reranked_pred = {
            "ground_truth": pred["ground_truth"],
            "candidates": reranked_candidates,
            "llm_scores": reranked_llm_scores,
            "final_scores": reranked_final_scores,
        }
        if "input" in pred:
            reranked_pred["input"] = pred["input"]
        reranked_predictions.append(reranked_pred)

    return reranked_predictions


def load_item_to_pop_group(data_dir, dataset):
    pop_csv_names = {
        "MovieLens1M": "movie_pop_count_5.csv",
        "steam": "game_pop_count_5.csv",
        "CDs_and_Vinyl": "item_pop_count_5.csv",
        "Beauty": "item_pop_count_5.csv",
        "Toys_and_Games": "item_pop_count_5.csv",
        "Video_Games": "item_pop_count_5.csv",
    }

    pop_csv_path = f"{data_dir}/{pop_csv_names.get(dataset, 'movie_pop_count_5.csv')}"

    if not os.path.exists(pop_csv_path):
        raise FileNotFoundError(f"Popularity group file not found: {pop_csv_path}")

    pop_df = pd.read_csv(pop_csv_path)
    item_to_pop = {}

    for _, row in pop_df.iterrows():
        title = row["Title"].strip()
        pop_group = str(int(row["Pop"]))
        item_to_pop[title] = pop_group

    return item_to_pop


def calculate_history_distribution(data_dir, item_to_pop):
    test_json_path = f"{data_dir}/test.json"

    if not os.path.exists(test_json_path):
        raise FileNotFoundError(f"Test JSON not found: {test_json_path}")

    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    group_set = ["0", "1", "2", "3", "4"]
    history_count = {g: 0 for g in group_set}

    for sample in tqdm(test_data):
        input_text = sample["input"]
        history_titles = re.findall(r'"([^"]+)"', input_text)

        for title in history_titles:
            title_clean = title.strip()
            if title_clean in item_to_pop:
                pop_group = item_to_pop[title_clean]
                history_count[pop_group] += 1

    return history_count


def calculate_metrics(predictions, item_to_pop, history_count, k_list=None):
    if k_list is None:
        k_list = [1, 5, 10, 20]

    group_set = ["0", "1", "2", "3", "4"]

    hr_count = {k: 0 for k in k_list}
    ndcg_sum = {k: 0.0 for k in k_list}
    topk_count = {k: {g: 0 for g in group_set} for k in k_list}

    word_set_dict = {k: {} for k in k_list}
    unique_items_dict = {k: set() for k in k_list}

    num_samples = len(predictions)
    valid_samples = 0

    for pred in tqdm(predictions):
        ground_truth = pred["ground_truth"].strip()
        candidates = [c.strip() for c in pred["candidates"]]

        if not candidates:
            continue

        valid_samples += 1

        target_position = len(candidates)
        for idx, candidate in enumerate(candidates):
            if candidate == ground_truth:
                target_position = idx
                break

        for k in k_list:
            for rank in range(min(k, len(candidates))):
                item = candidates[rank]

                if item in item_to_pop:
                    pop_group = item_to_pop[item]
                    topk_count[k][pop_group] += 1

                words = process_string(item)
                for word in words:
                    word_set_dict[k][word] = word_set_dict[k].get(word, 0) + 1

                unique_items_dict[k].add(item)

            if target_position < k:
                hr_count[k] += 1
                ndcg_sum[k] += 1.0 / math.log2(target_position + 2)

    metrics = {}

    for k in k_list:
        metrics[f"HR@{k}"] = hr_count[k] / num_samples if num_samples > 0 else 0

    for k in k_list:
        idcg = 1.0 / math.log2(2)
        metrics[f"NDCG@{k}"] = (ndcg_sum[k] / num_samples) / idcg if num_samples > 0 else 0

    total_history = sum(history_count.values())

    for k in k_list:
        total_recommend = sum(topk_count[k].values())

        gu_values = []
        for group in group_set:
            history_ratio = history_count[group] / total_history if total_history > 0 else 0
            recommend_ratio = (
                topk_count[k][group] / total_recommend if total_recommend > 0 else 0
            )
            gu_values.append(abs(recommend_ratio - history_ratio))

        metrics[f"MGU@{k}"] = sum(gu_values) / len(gu_values) if gu_values else 0
        metrics[f"DGU@{k}"] = max(gu_values) - min(gu_values) if gu_values else 0

    for k in k_list:
        metrics[f"H@{k}"] = calculate_entropy(word_set_dict[k])
        metrics[f"TTR@{k}"] = calculate_ttr(word_set_dict[k])

        num_unique_items = len(unique_items_dict[k])
        total_recommendations = valid_samples * k
        metrics[f"DivRatio@{k}"] = (
            num_unique_items / total_recommendations if total_recommendations > 0 else 0
        )

    metrics["num_samples"] = num_samples
    metrics["history_distribution"] = history_count
    metrics["recommend_distribution"] = topk_count
    metrics["word_statistics"] = {
        k: {"unique_words": len(word_set_dict[k]), "total_words": sum(word_set_dict[k].values())}
        for k in k_list
    }
    metrics["diversity_statistics"] = {
        k: {
            "unique_items": len(unique_items_dict[k]),
            "total_recommendations": valid_samples * k,
            "valid_samples": valid_samples,
        }
        for k in k_list
    }

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=[
            "MovieLens1M",
            "steam",
            "CDs_and_Vinyl",
            "Beauty",
            "Toys_and_Games",
            "Video_Games",
        ],
    )
    parser.add_argument("--stage", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--alpha_weight", type=float, default=None)
    parser.add_argument(
        "--exp_type",
        type=str,
        default=None,
        choices=["standard_dpo", "random", "overpop", "hard"],
    )
    parser.add_argument("--lambda_val", type=float, default=None)
    parser.add_argument("--k_list", type=int, nargs="+", default=None)
    parser.add_argument("--no_rerank", action="store_true")
    parser.add_argument("--hist_pred", type=str, default=None)

    args = parser.parse_args()

    dataset = args.dataset if args.dataset else DATASET
    alpha = args.alpha if args.alpha is not None else ALPHA
    lambda_val = args.lambda_val if args.lambda_val is not None else LAMBDA_VAL
    k_list = args.k_list if args.k_list else K_LIST

    data_dir = f"mydata/{dataset}/process_data"

    if args.stage == 1:
        mu = args.mu if args.mu is not None else 0.2
        results_dir = f"results_stage1/{dataset}/alpha_{alpha:.2f}_mu_{mu:.2f}"
    elif args.stage == 2:
        mu = args.mu if args.mu is not None else 0.3
        alpha_weight = args.alpha_weight if args.alpha_weight is not None else 0.6

        if args.exp_type is None:
            raise ValueError("Stage 2 requires --exp_type")

        if args.exp_type == "standard_dpo":
            exp_name = "standard_dpo"
        elif args.exp_type == "random":
            exp_name = f"random_aw_{alpha_weight:.2f}"
        elif args.exp_type == "overpop":
            exp_name = f"overpop_aw_{alpha_weight:.2f}"
        else:
            exp_name = f"hard_aw_{alpha_weight:.2f}"

        results_dir = f"results_stage2_abla/{dataset}/alpha_{alpha:.2f}_mu_{mu:.2f}_{exp_name}"
    else:
        mu = args.mu if args.mu is not None else 1.0
        alpha_weight = args.alpha_weight if args.alpha_weight is not None else 0.8
        results_dir = f"results_stage3/{dataset}/alpha_{alpha:.2f}_mu_{mu:.2f}_aw_{alpha_weight:.2f}"

    prediction_file = f"{results_dir}/predictions.json"
    if not os.path.exists(prediction_file):
        raise FileNotFoundError(f"Prediction file not found: {prediction_file}")

    predictions = load_predictions(prediction_file)
    item_to_pop = load_item_to_pop_group(data_dir, dataset)
    history_count = calculate_history_distribution(data_dir, item_to_pop)

    if not args.no_rerank:
        hist_file = args.hist_pred if args.hist_pred else f"{results_dir}/hist_predictions.json"
        if not os.path.exists(hist_file):
            raise FileNotFoundError(f"Historical prediction file not found: {hist_file}")
        hist_predictions = load_predictions(hist_file)
        p_llm = compute_p_llm_from_predictions(hist_predictions, num_beams=20)
        p_target = load_p_target(data_dir, alpha)
        predictions_to_eval = rerank_with_p_target(
            predictions, p_target, p_llm, lambda_val
        )

        reranked_file = f"{results_dir}/predictions_lambda_{lambda_val:.2f}.json"
        with open(reranked_file, "w", encoding="utf-8") as f:
            json.dump(predictions_to_eval, f, indent=2, ensure_ascii=False)

        metrics_file = f"{results_dir}/metrics_lambda_{lambda_val:.2f}.json"
    else:
        predictions_to_eval = predictions
        metrics_file = f"{results_dir}/metrics_no_rerank.json"

    metrics = calculate_metrics(
        predictions=predictions_to_eval,
        item_to_pop=item_to_pop,
        history_count=history_count,
        k_list=k_list,
    )

    metrics["config"] = {
        "dataset": dataset,
        "alpha": alpha,
        "lambda_val": lambda_val if not args.no_rerank else 0.0,
        "use_rerank": not args.no_rerank,
    }

    metrics_to_save = {
        k: v
        for k, v in metrics.items()
        if not k.startswith("history_") and not k.startswith("recommend_")
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)

    full_metrics_file = metrics_file.replace(".json", "_full.json")
    with open(full_metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
