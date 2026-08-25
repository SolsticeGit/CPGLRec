import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import os


DATASET = "MovieLens1M"
BASE_MODEL = "/home/power/MyLLMs/Qwen2.5-1.5B-Instruct"
BATCH_SIZE = 16
MAX_LENGTH = 128


def precompute_item_embeddings(
    base_model_path: str,
    items_dat_path: str,
    output_path: str,
    batch_size: int = 16,
    max_length: int = 64,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    item_titles = []
    item_ids = []

    with open(items_dat_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split("::")
            if len(parts) >= 2:
                item_id = parts[0]
                title = parts[1]
                item_ids.append(item_id)
                item_titles.append(title)

    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(item_titles), batch_size)):
            batch_titles = item_titles[i : i + batch_size]

            inputs = tokenizer(
                batch_titles,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            hidden_states = outputs.hidden_states[-1]
            attention_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            )

            sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask

            all_embeddings.append(mean_embeddings.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(
        {
            "item_ids": item_ids,
            "item_titles": item_titles,
            "embeddings": all_embeddings,
            "model_name": base_model_path,
            "hidden_dim": all_embeddings.shape[1],
        },
        output_path,
    )


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
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)

    args = parser.parse_args()

    dataset = args.dataset if args.dataset else DATASET
    base_model = args.base_model if args.base_model else BASE_MODEL
    batch_size = args.batch_size if args.batch_size else BATCH_SIZE
    max_length = args.max_length if args.max_length else MAX_LENGTH

    dataset_configs = {
        "MovieLens1M": {
            "items_dat": "mydata/MovieLens1M/process_data/movies.dat",
            "output": "mydata/MovieLens1M/process_data/item_embeddings.pt",
        },
        "steam": {
            "items_dat": "mydata/steam/process_data/games.dat",
            "output": "mydata/steam/process_data/item_embeddings.pt",
        },
        "CDs_and_Vinyl": {
            "items_dat": "mydata/CDs_and_Vinyl/process_data/items.dat",
            "output": "mydata/CDs_and_Vinyl/process_data/item_embeddings.pt",
        },
        "Beauty": {
            "items_dat": "mydata/Beauty/process_data/items.dat",
            "output": "mydata/Beauty/process_data/item_embeddings.pt",
        },
        "Toys_and_Games": {
            "items_dat": "mydata/Toys_and_Games/process_data/items.dat",
            "output": "mydata/Toys_and_Games/process_data/item_embeddings.pt",
        },
        "Video_Games": {
            "items_dat": "mydata/Video_Games/process_data/items.dat",
            "output": "mydata/Video_Games/process_data/item_embeddings.pt",
        },
    }

    config = dataset_configs[dataset]
    if not os.path.exists(config["items_dat"]):
        return

    precompute_item_embeddings(
        base_model_path=base_model,
        items_dat_path=config["items_dat"],
        output_path=config["output"],
        batch_size=batch_size,
        max_length=max_length,
    )


if __name__ == "__main__":
    main()
