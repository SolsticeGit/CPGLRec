import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse
import json
import os
from typing import Dict
from dataset_concept import ConceptAlignmentDataset, collate_fn


class ConceptAlignmentTrainer(Trainer):
    def __init__(
        self,
        item_embeddings: torch.Tensor,
        item_titles: list,
        p_target: Dict[str, float],
        mu: float = 0.2,
        temperature: float = 1.0,
        num_bins: int = 5,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.item_embeddings = item_embeddings.to(self.args.device)
        self.item_titles = item_titles
        self.p_target = p_target
        self.mu = mu
        self.temperature = temperature
        self.num_bins = num_bins

        self.title_to_idx = {title: idx for idx, title in enumerate(item_titles)}

        p_target_list = [p_target.get(title, 1e-10) for title in item_titles]
        self.p_target_tensor = torch.tensor(p_target_list, dtype=torch.float32).to(self.args.device)
        self.p_target_tensor = self.p_target_tensor / self.p_target_tensor.sum()

        self._build_bins()

    def _build_bins(self):
        p_target_sorted, sorted_indices = torch.sort(self.p_target_tensor)

        n = len(p_target_sorted)
        bin_size = n // self.num_bins

        self.item_to_bin = torch.zeros(n, dtype=torch.long, device=self.args.device)

        for b in range(self.num_bins):
            start_idx = b * bin_size
            end_idx = (b + 1) * bin_size if b < self.num_bins - 1 else n
            items_in_bin = sorted_indices[start_idx:end_idx]
            self.item_to_bin[items_in_bin] = b

        self.bin_target_quotas = torch.zeros(self.num_bins, device=self.args.device)
        for b in range(self.num_bins):
            mask = self.item_to_bin == b
            self.bin_target_quotas[b] = self.p_target_tensor[mask].sum()

        self.bin_weights = 1.0 / torch.sqrt(self.bin_target_quotas.clamp(min=1e-10))
        self.bin_weights = self.bin_weights / self.bin_weights.mean()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = dict(inputs)
        inputs.pop("item_title", None)

        outputs = model(**inputs)
        sft_loss = outputs.loss

        hidden_states = outputs.hidden_states[-1] if getattr(outputs, "hidden_states", None) else None
        if hidden_states is None:
            outputs_with_hidden = model(**inputs, output_hidden_states=True)
            hidden_states = outputs_with_hidden.hidden_states[-1]

        attention_mask = inputs["attention_mask"]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]

        last_hidden_states = hidden_states[torch.arange(batch_size), sequence_lengths, :]

        last_hidden_norm = F.normalize(last_hidden_states, p=2, dim=1)
        item_embeddings_norm = F.normalize(self.item_embeddings, p=2, dim=1)

        similarities = torch.matmul(last_hidden_norm, item_embeddings_norm.T)
        similarities = similarities / self.temperature
        p_model = F.softmax(similarities, dim=-1)

        bin_model_quotas = torch.zeros(batch_size, self.num_bins, device=self.args.device)
        for b in range(self.num_bins):
            mask = self.item_to_bin == b
            bin_model_quotas[:, b] = p_model[:, mask].sum(dim=1)

        target_quotas = self.bin_target_quotas.unsqueeze(0)
        bin_weights = self.bin_weights.unsqueeze(0)

        quota_diff = (bin_model_quotas - target_quotas) ** 2
        bin_loss = (quota_diff * bin_weights).sum(dim=1).mean()

        total_loss = sft_loss + self.mu * bin_loss

        if self.state is not None:
            log_dict = {
                "train/sft_loss": float(sft_loss.detach().cpu().item()),
                "train/bin_loss": float(bin_loss.detach().cpu().item()),
                "train/total_loss": float(total_loss.detach().cpu().item()),
            }
            for b in range(self.num_bins):
                quota_diff_b = (bin_model_quotas[:, b] - target_quotas[0, b]).abs().mean()
                log_dict[f"train/bin_{b+1}_diff"] = float(quota_diff_b.detach().cpu().item())
            self.log(log_dict)

        return (total_loss, outputs) if return_outputs else total_loss


def load_item_embeddings(embedding_path: str, device: str):
    data = torch.load(embedding_path, map_location=device)
    return data["embeddings"], data["item_titles"]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["MovieLens1M", "steam", "CDs_and_Vinyl", "Beauty", "Toys_and_Games", "Video_Games"],
    )
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--template", type=str, default="qwen", choices=["qwen", "llama"])

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_4bit", action="store_true")

    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--mu", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"output_stage1/{args.dataset}/alpha_{args.alpha:.2f}_mu_{args.mu:.2f}"
    os.makedirs(args.output_dir, exist_ok=True)

    with open(f"{args.output_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    data_dir = f"mydata/{args.dataset}/process_data"
    train_path = f"{data_dir}/train.json"
    p_target_path = f"{data_dir}/p_target_alpha_{args.alpha:.2f}.json"
    embedding_path = f"{data_dir}/item_embeddings.pt"

    for path in [train_path, p_target_path, embedding_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = ConceptAlignmentDataset(
        data_path=train_path,
        tokenizer=tokenizer,
        p_target_path=p_target_path,
        max_length=args.max_length,
        template=args.template,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    item_embeddings, item_titles = load_item_embeddings(embedding_path, device)

    with open(p_target_path, "r", encoding="utf-8") as f:
        p_target = json.load(f)["p_target"]

    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config if args.use_4bit else None,
        torch_dtype=None if args.use_4bit else torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.config.output_hidden_states = True

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    trainer = ConceptAlignmentTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        item_embeddings=item_embeddings,
        item_titles=item_titles,
        p_target=p_target,
        mu=args.mu,
        temperature=args.temperature,
    )

    trainer.train()

    final_output_dir = f"{args.output_dir}/final"
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)


if __name__ == "__main__":
    main()
