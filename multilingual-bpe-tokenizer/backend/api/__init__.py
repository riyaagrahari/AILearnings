"""FastAPI backend exposing the from-scratch BPE tokenizer package (bpe/).

This layer is intentionally thin: every number returned by
``GET /api/statistics`` and ``POST /api/tokenize`` comes from actually
running the trained tokenizer (``bpe.tokenizer.BPETokenizer``) against
real corpus files -- nothing here computes, caches, or hardcodes a
statistic independently of the tokenizer itself. See ``service.py`` for
where the trained artifacts and evaluation corpus are loaded from.
"""
