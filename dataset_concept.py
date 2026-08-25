import torch
from torch.utils.data import Dataset
import json
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer


class ConceptAlignmentDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        p_target_path: Optional[str] = None,
        max_length: int = 512,
        template: str = "qwen",
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template

        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.p_target = None
        if p_target_path:
            with open(p_target_path, "r", encoding="utf-8") as f:
                p_target_data = json.load(f)
                self.p_target = p_target_data.get("p_target", None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output_text = sample.get("output", "")

        item_title = output_text.strip('"')

        if self.template == "qwen":
            conversation = (
                f"<|im_start|>system\n{instruction}<|im_end|>\n"
                f"<|im_start|>user\n{input_text}<|im_end|>\n"
                f"<|im_start|>assistant\n{output_text}<|im_end|>"
            )
        elif self.template == "llama":
            conversation = f"[INST] {instruction}\n{input_text} [/INST] {output_text}"
        else:
            conversation = f"{instruction}\n{input_text}\n{output_text}"

        encodings = self.tokenizer(
            conversation,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        labels = input_ids.copy()

        if self.template == "qwen":
            assistant_start = conversation.find("<|im_start|>assistant\n") + len(
                "<|im_start|>assistant\n"
            )
        elif self.template == "llama":
            assistant_start = conversation.find("[/INST]") + len("[/INST] ")
        else:
            assistant_start = len(f"{instruction}\n{input_text}\n")

        prefix = conversation[:assistant_start]
        prefix_tokens = self.tokenizer(
            prefix,
            truncation=False,
            padding=False,
            return_tensors=None,
        )["input_ids"]
        prefix_len = len(prefix_tokens)

        labels[:prefix_len] = [-100] * prefix_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "item_title": item_title,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    item_titles = [item["item_title"] for item in batch]

    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    max_len = max(len(ids) for ids in input_ids)

    input_ids_padded = torch.full((len(batch), max_len), 0, dtype=torch.long)
    attention_mask_padded = torch.full((len(batch), max_len), 0, dtype=torch.long)
    labels_padded = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, (ids, mask, labs) in enumerate(zip(input_ids, attention_mask, labels)):
        length = len(ids)
        input_ids_padded[i, :length] = ids
        attention_mask_padded[i, :length] = mask
        labels_padded[i, :length] = labs

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded,
        "item_title": item_titles,
    }


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True
    )

    dataset = ConceptAlignmentDataset(
        data_path="mydata/MovieLens1M/process_data/train.json",
        tokenizer=tokenizer,
        p_target_path="mydata/MovieLens1M/process_data/p_target_alpha_0.5.json",
        max_length=512,
        template="qwen",
    )

    sample = dataset[0]

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
    )

    batch = next(iter(dataloader))
    _ = (sample, batch)
