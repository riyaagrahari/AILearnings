import json
from pathlib import Path

def test_evidence_and_performance_exist_after_demo():
    artifacts = Path("submission_artifacts")
    assert (artifacts / "evidence.json").exists()
    assert (artifacts / "evidence.md").exists()
    assert (artifacts / "performance.json").exists()
    evidence = json.loads((artifacts / "evidence.json").read_text())
    assert all(v["result"] == "PASS" for v in evidence.values())
