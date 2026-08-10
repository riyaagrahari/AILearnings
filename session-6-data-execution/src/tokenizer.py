from dataclasses import dataclass
from .utils import sha256_obj

@dataclass(frozen=True)
class FrozenTokenizer:
    version: str = "demo-tokenizer-v1"
    vocab_size: int = 256
    special_tokens: tuple = (("<PAD>", 0), ("<BOS>", 1), ("<EOS>", 2), ("<UNK>", 3))

    @property
    def tokenizer_hash(self):
        return sha256_obj({
            "version": self.version,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "algorithm": "utf8-byte-mod-252",
        })

    def encode(self, text):
        ids = [1]
        for b in text.encode("utf-8"):
            ids.append(4 + (b % (self.vocab_size - 4)))
        ids.append(2)
        return ids
