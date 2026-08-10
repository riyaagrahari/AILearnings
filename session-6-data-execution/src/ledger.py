from pathlib import Path
from .utils import canonical, sha256_text

class Ledger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.path.write_text("", encoding="utf-8")
        self.prev_hash = "GENESIS"

    def append(self, record):
        record = dict(record)
        record["prev_hash"] = self.prev_hash
        record_hash = sha256_text(canonical(record))
        record["record_hash"] = record_hash
        self.prev_hash = record_hash
        self.records.append(record)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(canonical(record) + "\n")
        return len(self.records) - 1

    @property
    def offset(self):
        return len(self.records)

    def verify_chain(self):
        prev = "GENESIS"
        for r in self.records:
            stored = r["record_hash"]
            body = dict(r)
            body.pop("record_hash")
            assert body["prev_hash"] == prev
            assert sha256_text(canonical(body)) == stored
            prev = stored
        return True
