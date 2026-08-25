import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
import argparse
import json
import os


class WeightedDPOTrainer(DPOTrainer):
    def dpo_loss(self, *args, **kwargs):
        losses, chosen_rewards, rejected_rewards = super().dpo_loss(*args, **kwargs)
        self._unmeaned_losses = losses
        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        weights = batch.pop("weight", None)
        loss, metrics = super().get_batch_loss_metrics(model, batch, train_eval)
        if weights is not None:
            losses = self._unmeaned_losses
            weights = torch.as_tensor(weights, device=losses.device, dtype=losses.dtype).reshape(-1)
            loss = (losses * weights).mean()
            metrics[f"{train_eval}_weight_mean"] = weights.mean().item()
            metrics[f"{train_eval}_weight_std"] = weights.std().item()
            metrics[f"{train_eval}_weight_min"] = weights.min().item()
            metrics[f"{train_eval}_weight_max"] = weights.max().item()
        return loss, metrics

    def concatenated_forward(self, model, batch, is_ref_model=False):
        weights = batch.pop("weight", None)
        result = super().concatenated_forward(model, batch, is_ref_model=is_ref_model)
        if weights is not None:
            batch["weight"] = weights
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["MovieLens1M", "steam", "CDs_and_Vinyl", "Beauty", "Toys_and_Games", "Video_Games"])
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--alpha_weight", type=float, default=0.8)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--stage1_checkpoint", type=str, default=None)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--use_standard_dpo", action="store_true")
    parser.add_argument("--exp_type", type=str, default=None, choices=["standard_dpo", "random", "overpop", "hard"])
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)

    args = parser.parse_args()

    if args.stage1_checkpoint is None:
        args.stage1_checkpoint = f"output_stage1/{args.dataset}/alpha_{args.alpha:.2f}_mu_{args.mu:.2f}/final"

    if not os.path.exists(args.stage1_checkpoint):
        raise FileNotFoundError(args.stage1_checkpoint)

    if args.output_dir is None:
        if args.use_standard_dpo or args.exp_type == "standard_dpo":
            args.output_dir = f"output_stage2_abla/{args.dataset}/alpha_{args.alpha:.2f}_mu_{args.mu:.2f}_standard_dpo"
        elif args.exp_type in ["random", "overpop", "hard"]:
            args.output_dir = f"output_stage2_abla/{args.dataset}/alpha_{args.alpha:.2f}_mu_{args.mu:.2f}_{args.exp_type}_aw_{args.alpha_weight:.2f}"
        else:
            args.output_dir = f"output_stage2/{args.dataset}/alpha_{args.alpha:.2f}_mu_{args.mu:.2f}_aw_{args.alpha_weight:.2f}"

    os.makedirs(args.output_dir, exist_ok=True)

    with open(f"{args.output_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.use_standard_dpo or args.exp_type == "standard_dpo":
        dpo_data_path = f"dpo_data_abla/{args.dataset}/dpo_pairs_alpha_{args.alpha:.2f}_standard.jsonl"
    elif args.exp_type == "random":
        dpo_data_path = f"dpo_data_abla/{args.dataset}/dpo_pairs_alpha_{args.alpha:.2f}_random_aw_{args.alpha_weight:.2f}.jsonl"
    elif args.exp_type == "overpop":
        dpo_data_path = f"dpo_data_abla/{args.dataset}/dpo_pairs_alpha_{args.alpha:.2f}_overpop_aw_{args.alpha_weight:.2f}.jsonl"
    elif args.exp_type == "hard":
        dpo_data_path = f"dpo_data_abla/{args.dataset}/dpo_pairs_alpha_{args.alpha:.2f}_hard_aw_{args.alpha_weight:.2f}.jsonl"
    else:
        dpo_data_path = f"dpo_data/{args.dataset}/dpo_pairs_alpha_{args.alpha:.2f}_aw_{args.alpha_weight:.2f}.jsonl"

    if not os.path.exists(dpo_data_path):
        raise FileNotFoundError(dpo_data_path)

    train_dataset = load_dataset("json", data_files=dpo_data_path, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not args.use_4bit else None,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    model = PeftModel.from_pretrained(model, args.stage1_checkpoint, is_trainable=True)

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not args.use_4bit else None,
        device_map="auto",
        trust_remote_code=True,
    )

    ref_model = PeftModel.from_pretrained(ref_model, args.stage1_checkpoint)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        beta=args.beta,
    )

    if args.use_standard_dpo:
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
    else:
        dpo_trainer = WeightedDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )

    dpo_trainer.train()

    final_output_dir = f"{args.output_dir}/final"
    dpo_trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)


if __name__ == "__main__":
    main()
