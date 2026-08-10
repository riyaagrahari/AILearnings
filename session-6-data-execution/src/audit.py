import json
from .utils import write_json

def generate_evidence(path, checks):
    evidence = {}
    for name, data in checks.items():
        evidence[name] = {
            "result": "PASS" if data["passed"] else "FAIL",
            "evidence": data["evidence"],
        }
    write_json(path, evidence)
    return evidence

def generate_evidence_md(path, evidence):
    labels = {
        "tokenizer_integrity": "Tokenizer integrity",
        "evaluation_firewall": "Evaluation firewall",
        "packing_correctness": "Packing correctness",
        "mixture_compliance": "Mixture compliance",
        "opus_audit": "OPUS audit trail",
        "crash_recovery": "Crash recovery",
        "replay": "Replay",
        "learning_trace": "Learning trace",
        "throughput": "Throughput",
    }
    lines = [
        "# V5 Training Data Execution Evidence",
        "",
        "| Requirement | Result | Evidence |",
        "|---|---|---|",
    ]
    for k, v in evidence.items():
        lines.append(f"| {labels.get(k,k)} | {v['result']} | {v['evidence']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
