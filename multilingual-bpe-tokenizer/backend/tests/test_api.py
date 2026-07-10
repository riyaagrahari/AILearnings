"""Integration tests for the FastAPI app (api/main.py).

Trains a tiny real tokenizer into a temp "artifacts" directory and points
the service layer at it via environment variables, so these tests
exercise the actual HTTP layer end-to-end against a real (if small)
tokenizer -- no mocking of bpe.tokenizer/bpe.evaluation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import importlib  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from bpe.trainer import save_merges, save_vocab, train_bpe_from_texts  # noqa: E402


def _reimport_api_app():
    """Force a fully fresh import of the whole ``api`` package tree.

    Popping just "api.service"/"api.main" from ``sys.modules`` is *not*
    enough: ``api/main.py`` does ``from api import service``, and
    CPython's `from X import Y` resolves via ``getattr(sys.modules["X"],
    "Y")`` before it consults ``sys.modules["X.Y"]`` -- so if the parent
    package module "api" is still cached (which it is; we never popped
    it), it still holds a stale ``.service`` attribute pointing at the
    old module object, and the "fresh" reimport silently reuses old,
    already-baked-in environment-variable-derived paths. Popping every
    "api"-prefixed module avoids that trap.
    """
    for mod_name in list(sys.modules):
        if mod_name == "api" or mod_name.startswith("api."):
            del sys.modules[mod_name]
    return importlib.import_module("api.main")


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    """Train a tiny tokenizer + write a tiny eval corpus into tmp_path,
    point the service layer's module-level paths at them, then import a
    fresh FastAPI app bound to those paths.
    """
    artifacts_dir = tmp_path / "artifacts"
    eval_dir = tmp_path / "eval"
    artifacts_dir.mkdir()
    for lang in ("en", "hi"):
        (eval_dir / lang).mkdir(parents=True)

    texts = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog are friends",
    ]
    result = train_bpe_from_texts(texts, vocab_size=150)
    save_vocab(result, artifacts_dir / "vocab.json")
    save_merges(result, artifacts_dir / "merges.json")
    (eval_dir / "en" / "sample.txt").write_text("the cat sat on the log", encoding="utf-8")
    (eval_dir / "hi" / "sample.txt").write_text("the dog and the cat", encoding="utf-8")

    monkeypatch.setenv("BPE_ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("BPE_EVAL_DIR", str(eval_dir))
    monkeypatch.setenv("BPE_LANGUAGES", "en,hi")

    # api.service reads these env vars at import time -- reimport fresh
    # modules so the fixture's paths actually take effect.
    main = _reimport_api_app()

    with TestClient(main.app) as client:
        yield client, artifacts_dir, eval_dir


@pytest.fixture()
def api_client_no_artifacts(tmp_path, monkeypatch):
    """Same as api_client, but the artifacts directory is empty -- for
    testing the "not trained yet" error path.
    """
    monkeypatch.setenv("BPE_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("BPE_EVAL_DIR", str(tmp_path / "eval"))
    main = _reimport_api_app()
    with TestClient(main.app) as client:
        yield client


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------


def test_health_reports_trained_when_artifacts_exist(api_client):
    client, _, _ = api_client
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "tokenizer_trained": True}


def test_health_reports_not_trained_when_artifacts_missing(api_client_no_artifacts):
    response = api_client_no_artifacts.get("/api/health")
    assert response.status_code == 200
    assert response.json()["tokenizer_trained"] is False


# ---------------------------------------------------------------------------
# /api/statistics
# ---------------------------------------------------------------------------


def test_statistics_returns_real_computed_values(api_client):
    client, _, _ = api_client
    response = client.get("/api/statistics")
    assert response.status_code == 200
    data = response.json()

    assert data["vocab_size"] == 150 or data["vocab_size"] <= 150
    assert {entry["language"] for entry in data["languages"]} == {"English", "Hindi"}
    for entry in data["languages"]:
        assert entry["total_tokens"] > 0
        assert entry["total_words"] > 0
        assert entry["ratio"] == pytest.approx(
            entry["total_tokens"] / entry["total_words"]
        )
    assert data["largest_ratio"] >= data["smallest_ratio"]
    assert data["difference"] == pytest.approx(data["largest_ratio"] - data["smallest_ratio"])


def test_statistics_503_when_not_trained(api_client_no_artifacts):
    response = api_client_no_artifacts.get("/api/statistics")
    assert response.status_code == 503
    assert "train_tokenizer.py" in response.json()["detail"]


def test_statistics_matches_direct_tokenizer_computation(api_client):
    """The API must not diverge from calling the tokenizer directly --
    this is the core "tokenizer remains source of truth" guarantee.

    The API's Xi is fertility (word-like tokens / words), so this
    replicates evaluate_language on the same eval text and checks the API
    agrees.
    """
    client, artifacts_dir, _ = api_client
    from bpe.evaluation import evaluate_language
    from bpe.tokenizer import BPETokenizer

    tok = BPETokenizer.from_files(artifacts_dir / "vocab.json", artifacts_dir / "merges.json")
    # Same eval text the api_client fixture writes to data/eval/en/.
    direct = evaluate_language(tok, ["the cat sat on the log"], "en")

    response = client.get("/api/statistics")
    data = response.json()
    en_entry = next(e for e in data["languages"] if e["language"] == "English")
    assert en_entry["total_tokens"] == direct.num_tokens
    assert en_entry["total_words"] == direct.num_words
    assert en_entry["ratio"] == pytest.approx(direct.fertility)


# ---------------------------------------------------------------------------
# /api/tokenize
# ---------------------------------------------------------------------------


def test_tokenize_returns_consistent_pretokens_tokens_ids_decoded(api_client):
    client, _, _ = api_client
    response = client.post("/api/tokenize", json={"text": "the cat sat"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["tokens"]) == len(data["ids"])
    assert data["decoded_text"] == "the cat sat"
    assert data["pretokens"]  # non-empty


def test_tokenize_empty_text_is_rejected(api_client):
    client, _, _ = api_client
    response = client.post("/api/tokenize", json={"text": ""})
    assert response.status_code == 422


def test_tokenize_503_when_not_trained(api_client_no_artifacts):
    response = api_client_no_artifacts.post("/api/tokenize", json={"text": "hello"})
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


def test_download_vocab_json_has_correct_filename_and_content(api_client):
    client, artifacts_dir, _ = api_client
    response = client.get("/tokenizer/vocab.json")
    assert response.status_code == 200
    assert 'filename="vocab.json"' in response.headers["content-disposition"]
    assert response.content == (artifacts_dir / "vocab.json").read_bytes()


def test_download_merges_json_has_correct_filename_and_content(api_client):
    client, artifacts_dir, _ = api_client
    response = client.get("/tokenizer/merges.json")
    assert response.status_code == 200
    assert 'filename="merges.json"' in response.headers["content-disposition"]
    assert response.content == (artifacts_dir / "merges.json").read_bytes()


def test_download_tokenizer_json_combines_vocab_and_merges(api_client):
    client, _, _ = api_client
    response = client.get("/tokenizer/tokenizer.json")
    assert response.status_code == 200
    assert 'filename="tokenizer.json"' in response.headers["content-disposition"]
    payload = response.json()
    assert "vocab" in payload
    assert "merges" in payload


def test_downloads_503_when_not_trained(api_client_no_artifacts):
    for path in ("/tokenizer/vocab.json", "/tokenizer/merges.json", "/tokenizer/tokenizer.json"):
        response = api_client_no_artifacts.get(path)
        assert response.status_code == 503
