import json
from torch.utils.data import Dataset
from typing import Dict


class DPODataset(Dataset):
    def __init__(self, data_path: str):
        super().__init__()

        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        if len(self.data) > 0:
            sample = self.data[0]
            assert "prompt" in sample
            assert "chosen" in sample
            assert "rejected" in sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        sample = self.data[idx]

        result = {
            "prompt": sample["prompt"],
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
        }

        if "weight" in sample:
            result["weight"] = float(sample["weight"])

        if "neg_type" in sample:
            result["neg_type"] = sample["neg_type"]

        return result

    def get_statistics(self):
        import numpy as np

        stats = {
            "total_samples": len(self.data),
            "avg_prompt_length": 0,
            "avg_chosen_length": 0,
            "avg_rejected_length": 0,
        }

        weights = []
        neg_types = {"random": 0, "overpopular": 0, "hard": 0}

        for sample in self.data:
            stats["avg_prompt_length"] += len(sample["prompt"])
            stats["avg_chosen_length"] += len(sample["chosen"])
            stats["avg_rejected_length"] += len(sample["rejected"])

            if "weight" in sample:
                weights.append(sample["weight"])

            if "neg_type" in sample:
                neg_type = sample["neg_type"]
                if neg_type in neg_types:
                    neg_types[neg_type] += 1

        if len(self.data) > 0:
            stats["avg_prompt_length"] /= len(self.data)
            stats["avg_chosen_length"] /= len(self.data)
            stats["avg_rejected_length"] /= len(self.data)

        if weights:
            stats["has_weights"] = True
            stats["weight_mean"] = float(np.mean(weights))
            stats["weight_std"] = float(np.std(weights))
            stats["weight_min"] = float(np.min(weights))
            stats["weight_max"] = float(np.max(weights))
        else:
            stats["has_weights"] = False

        if sum(neg_types.values()) > 0:
            stats["neg_type_distribution"] = neg_types

        return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit(1)

    data_path = sys.argv[1]
    dataset = DPODataset(data_path)
    stats = dataset.get_statistics()
    _ = (dataset, stats)
