"""FastAPI app for the multilingual BPE tokenizer.

Run with:

    cd backend && uvicorn api.main:app --reload --port 8000

See README.md for the full train -> evaluate -> serve -> deploy flow.
Every response is computed live from the trained tokenizer (see
service.py) -- this file only wires HTTP routes to it and translates the
service layer's exceptions into honest HTTP error responses (503 "not
trained/no data yet" rather than a fabricated 200).
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from api import service
from api.schemas import (
    ErrorResponse,
    LanguageStatistics,
    StatisticsResponse,
    TokenizeRequest,
    TokenizeResponse,
)

app = FastAPI(
    title="Multilingual BPE Tokenizer API",
    description=(
        "Serves statistics, a tokenize/decode playground, and download "
        "endpoints for a from-scratch BPE tokenizer trained on "
        "English/Hindi/Telugu/Tamil. All numbers are computed live from "
        "the trained tokenizer and real evaluation corpus -- nothing is "
        "hardcoded."
    ),
    version="1.0.0",
)

_allowed_origins = [
    origin.strip()
    for origin in os.environ.get("BPE_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    """Cheap liveness check the frontend can use for connection status,
    plus whether a trained tokenizer is currently available.
    """
    return {
        "status": "ok",
        "tokenizer_trained": service.VOCAB_PATH.exists() and service.MERGES_PATH.exists(),
    }


@app.get(
    "/api/statistics",
    response_model=StatisticsResponse,
    responses={503: {"model": ErrorResponse}},
)
def get_statistics() -> StatisticsResponse:
    try:
        report = service.compute_statistics()
    except (service.ArtifactsNotFoundError, service.NoEvaluationDataError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return StatisticsResponse(
        vocab_size=report.vocab_size,
        languages=[
            LanguageStatistics(
                language=service.display_name(entry.language),
                total_tokens=entry.total_tokens,
                total_words=entry.total_words,
                ratio=entry.ratio,
            )
            for entry in report.languages
        ],
        largest_ratio=report.largest_ratio,
        smallest_ratio=report.smallest_ratio,
        difference=report.difference,
        assignment_score=report.assignment_score,
    )


@app.post(
    "/api/tokenize",
    response_model=TokenizeResponse,
    responses={503: {"model": ErrorResponse}},
)
def tokenize(request: TokenizeRequest) -> TokenizeResponse:
    try:
        result = service.tokenize_text(request.text)
    except service.ArtifactsNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return TokenizeResponse(**result)


def _download_response(path, filename: str, media_type: str = "application/json") -> Response:
    try:
        service._require_artifacts()
    except service.ArtifactsNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not found")
    return Response(
        content=path.read_bytes(),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/tokenizer/vocab.json", responses={503: {"model": ErrorResponse}})
def download_vocab() -> Response:
    return _download_response(service.VOCAB_PATH, "vocab.json")


@app.get("/tokenizer/merges.json", responses={503: {"model": ErrorResponse}})
def download_merges() -> Response:
    return _download_response(service.MERGES_PATH, "merges.json")


@app.get("/tokenizer/tokenizer.json", responses={503: {"model": ErrorResponse}})
def download_tokenizer_json() -> Response:
    try:
        payload = service.build_combined_tokenizer_json()
    except service.ArtifactsNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="tokenizer.json"'},
    )
