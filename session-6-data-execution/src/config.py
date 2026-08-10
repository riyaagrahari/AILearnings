from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "submission_artifacts"

@dataclass(frozen=True)
class Config:
    sequence_length: int = 16
    batch_size: int = 2
    seed: int = 20260809
    crash_step: int = 4
    replay_start: int = 1
    replay_end: int = 4
    vocab_size: int = 256
    lanes: tuple = ("general", "coding", "reasoning", "agentic", "indic", "math", "science")

DEFAULT_CONFIG = Config()
