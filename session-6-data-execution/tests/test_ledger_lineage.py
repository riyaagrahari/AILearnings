from pathlib import Path
from src.ledger import Ledger

def test_ledger_offsets_are_append_only(tmp_path):
    l = Ledger(tmp_path / "ledger.jsonl")
    assert l.offset == 0
    l.append({"event":"A","step":0})
    assert l.offset == 1
    l.append({"event":"B","step":1})
    assert l.offset == 2
    lines = (tmp_path / "ledger.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert lines[0] != lines[1]
