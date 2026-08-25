import os
import json
import random
import torch
import argparse
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)
from peft import PeftModel


def load_model_and_tokenizer(base_model_path, stage2_checkpoint, use_4bit=False):
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config if use_4bit else None,
    )

    model = PeftModel.from_pretrained(model, stage2_checkpoint)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def generate_candidates(
    model,
    tokenizer,
    test_data,
    num_beams=20,
    batch_size=128,
    micro_batch_size=4,
    max_new_tokens=64,
    temperature=1.0,
):
    device = model.device
    results = []

    num_batches = (len(test_data) + micro_batch_size - 1) // micro_batch_size

    for batch_idx in tqdm(range(num_batches)):
        batch_start = batch_idx * micro_batch_size
        batch_end = min(batch_start + micro_batch_size, len(test_data))
        batch_samples = test_data[batch_start:batch_end]

        input_texts = []
        for sample in batch_samples:
            instruction = sample.get("instruction", "Recommend the next item.")
            prompt = (
                f"<|imim_start|>system\n{instruction}<|im_end|>\n"
                f"<|im_start|>user\n{sample['input']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            input_texts.append(prompt)

        encodings = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        generation_config = GenerationConfig(
            num_beams=num_beams,
            num_return_sequences=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            early_stopping=True,
        )

        with torch.no_grad():
            generation_output = model.generate(
                **encodings,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_sequences = generation_output.sequences
        beam_scores = generation_output.sequences_scores.cpu().tolist()

        input_length = encodings.input_ids.shape[1]
        generated_tokens = generated_sequences[:, input_length:]

        decoded_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        cleaned_texts = []
        for text in decoded_texts:
            text = text.strip()
            if text.startswith('"') and '"' in text[1:]:
                end_quote = text.index('"', 1)
                text = text[1:end_quote]
            else:
                text = text.strip('"').strip()
            cleaned_texts.append(text)

        for i, sample in enumerate(batch_samples):
            start_idx = i * num_beams
            end_idx = start_idx + num_beams

            candidates = [c.strip() for c in cleaned_texts[start_idx:end_idx]]
            llm_scores = beam_scores[start_idx:end_idx]

            result = {
                "input": sample["input"],
                "ground_truth": sample["output"].strip().strip('"'),
                "candidates": candidates,
                "llm_scores": llm_scores,
            }
            results.append(result)

    _ = batch_size
    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "MovieLens1M",
            "steam",
            "CDs_and_Vinyl",
            "Beauty",
            "Toys_and_Games",
            "Video_Games",
        ],
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--alpha_weight", type=float, default=0.8)

    parser.add_argument(
        "--base_model",
        type=str,
        default="/home/power/MyLLMs/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument("--stage2_checkpoint", type=str, default=None)
    parser.add_argument("--use_4bit", action="store_true")

    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["test", "hist"])
    parser.add_argument("--hist_sample_size", type=int, default=5000)

    args = parser.parse_args()

    if args.stage2_checkpoint is None:
        args.stage2_checkpoint = (
            f"output_stage2/{args.dataset}/alpha_{args.alpha:.2f}_mu_{args.mu:.2f}_aw_{args.alpha_weight:.2f}/final"
        )

    if not os.path.exists(args.stage2_checkpoint):
        raise FileNotFoundError(f"Stage 2 checkpoint not found: {args.stage2_checkpoint}")

    data_dir = f"mydata/{args.dataset}/process_data"
    if args.split == "hist":
        data_file = f"{data_dir}/train.json"
    else:
        data_file = f"{data_dir}/test.json"

    if args.output_dir is None:
        output_dir = (
            f"results_stage3/{args.dataset}/alpha_{args.alpha:.2f}_mu_{args.mu:.2f}_aw_{args.alpha_weight:.2f}"
        )
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(
        args.base_model, args.stage2_checkpoint, args.use_4bit
    )

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.split == "hist" and len(data) > args.hist_sample_size:
        random.seed(42)
        data = random.sample(data, args.hist_sample_size)

    results = generate_candidates(
        model=model,
        tokenizer=tokenizer,
        test_data=data,
        num_beams=args.num_beams,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    if args.split == "hist":
        output_file = f"{output_dir}/hist_predictions.json"
    else:
        output_file = f"{output_dir}/predictions.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
