# V5 Training Data Execution Evidence

| Requirement | Result | Evidence |
|---|---|---|
| Tokenizer integrity | PASS | manifests/all.json |
| Evaluation firewall | PASS | run.log + eval manifest |
| OPUS audit trail | PASS | manifests/opus_decisions.json |
| Packing correctness | PASS | manifests/batches.json |
| Crash recovery | PASS | checkpoints/checkpoint-0004.json |
| Replay | PASS | manifests/replay.json |
| Mixture compliance | PASS | manifests/mixture.json + mixture_actual.json |
| Learning trace | PASS | ledgers/consumption.jsonl + learning.jsonl |
| Throughput | PASS | performance.json |
| ledger_integrity | PASS | hash-chained ledgers |
