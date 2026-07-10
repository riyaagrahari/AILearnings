"""Core, from-scratch BPE tokenizer package.

Pipeline stages, in order:

    preprocess.py   -- raw source text -> clean plain text
    pretokenizer.py -- clean text -> word-level pretokens (shared by
                       training and inference)
    corpus.py       -- data/raw/<lang>/*.txt -> weighted, combined
                       word-frequency table (the only module that reads
                       the user-provided corpus)
    trainer.py      -- word-frequency table -> learned merges
                       (vocab.json / merges.json)
    tokenizer.py    -- vocab.json + merges.json -> encode() / decode()
    evaluation.py   -- trained tokenizer + held-out text -> fertility,
                       UNK-rate, compression, roundtrip, vocab-utilization
                       metrics (eval_report.json)

All six stages are implemented; see docs/ARCHITECTURE.md for the full
data flow and docs/BPE_ALGORITHM.md for the design rationale behind each
one. ``scripts/train_tokenizer.py`` and ``scripts/evaluate_tokenizer.py``
are thin CLI wrappers around this package -- no algorithmic logic lives
in the scripts themselves.
"""
