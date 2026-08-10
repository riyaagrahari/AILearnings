import math

class ToyModel:
    """Tiny deterministic model surrogate: produces reproducible loss from tokens."""
    def loss(self, batch):
        values = []
        for sample in batch["samples"]:
            for x, y, m in zip(sample["input_ids"], sample["labels"], sample["loss_mask"]):
                if m:
                    values.append(1.0 + ((x + y) % 17) / 10.0)
        return (sum(values) / len(values)) if values else 0.0, len(values)
