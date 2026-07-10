"""Pydantic request/response models for the tokenizer API.

These models are a pure presentation-layer mirror of the dataclasses in
``bpe.evaluation`` (``AssignmentLanguageRatio``/``AssignmentRatioReport``,
where ``ratio = Xi = total_tokens / total_words``) and of
``bpe.tokenizer.BPETokenizer``'s encode/decode outputs -- no statistic is
computed here, only shaped for JSON.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LanguageStatistics(BaseModel):
    """One language's row in the statistics table."""

    language: str = Field(..., description="Human-readable language name, e.g. 'English'")
    total_tokens: int = Field(..., description="Total BPE tokens produced for this language's words")
    total_words: int = Field(..., description="Total words (word/digit pretokens) in this language's text")
    ratio: float = Field(..., description="Xi = total_tokens / total_words (fertility, tokens per word)")


class StatisticsResponse(BaseModel):
    vocab_size: int
    languages: list[LanguageStatistics]
    largest_ratio: float
    smallest_ratio: float
    difference: float
    # `Any`, not `float | str`: Pydantic's float coercion happily accepts
    # the *string* "Infinity" (Python's float() does too) and would
    # silently turn it back into a real inf, which the standard JSON
    # encoder then refuses to serialize. `Any` passes the value through
    # untouched, so the edge case stays the JSON string "Infinity" and
    # the normal case stays a plain numeric literal.
    assignment_score: Any = Field(
        ..., description="1000 / difference (number), or the string 'Infinity' if difference == 0"
    )


class TokenizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)


class TokenizeResponse(BaseModel):
    pretokens: list[str] = Field(..., description="Output of the shared pretokenizer, before merges")
    tokens: list[str] = Field(..., description="Final BPE subword tokens after merges")
    ids: list[int] = Field(..., description="Vocabulary ids for `tokens`")
    decoded_text: str = Field(..., description="decode(ids) -- reconstructed text")


class ErrorResponse(BaseModel):
    detail: str
