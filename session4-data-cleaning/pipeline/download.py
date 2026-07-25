"""Download real Wikipedia parquet shards for the Session 4 cleaning pipeline.

We pull an *India-first, multilingual* corpus straight from the open
``wikimedia/wikipedia`` dataset on the Hugging Face Hub (CC BY-SA 4.0) -- the
same class of source Session 3 lists under "Wikipedia" and "High-quality Web".

Only the first shard of each language is fetched; the pipeline itself reads a
bounded character budget out of these shards, so we never need every shard.

Note: this repo's networks sit behind a TLS-intercepting corporate proxy, so
certificate verification is disabled for these downloads (the same reason the
other sessions document ``--strict-ssl=false``). The files are public and their
integrity is re-checked by content hashing inside the pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

RAW_DIR = Path(__file__).parent / "data" / "raw"

# language code -> Hugging Face config (dump date . wiki code)
LANGUAGES: dict[str, str] = {
    "hi": "20231101.hi",  # Hindi   (Devanagari)
    "te": "20231101.te",  # Telugu  (Telugu script)
    "mr": "20231101.mr",  # Marathi (Devanagari -- shares script with Hindi)
}

API = "https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/{cfg}/train"


def shard_urls(cfg: str) -> list[str]:
    r = requests.get(API.format(cfg=cfg), verify=False, timeout=120)
    r.raise_for_status()
    return r.json()


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already have {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return
    print(f"  downloading {dest.name} ...", flush=True)
    with requests.get(url, verify=False, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        got = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                got += len(chunk)
                if total:
                    pct = 100 * got / total
                    print(f"\r    {got/1e6:6.1f}/{total/1e6:.1f} MB ({pct:4.1f}%)",
                          end="", flush=True)
        print()


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for lang, cfg in LANGUAGES.items():
        print(f"[{lang}] {cfg}")
        urls = shard_urls(cfg)
        # first shard is plenty -- the pipeline reads a bounded char budget.
        download(urls[0], RAW_DIR / f"{lang}.parquet")
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
