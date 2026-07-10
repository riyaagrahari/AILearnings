"""Download real Wikipedia article text for en/hi/te/ta into data/raw/ (for
training) and data/eval/ (a genuinely different, held-out set of articles).

This is a one-off data-acquisition script, not part of the tokenizer
package (backend/bpe/) itself -- it only ever writes plain UTF-8 .txt
files under data/raw/<lang>/ and data/eval/<lang>/, in the exact layout
bpe.corpus already expects (see data/README.md). Nothing here is
synthetic: every file is a real Wikipedia extract.

How language editions are matched
----------------------------------
Guessing a Hindi/Telugu/Tamil title by hand is unreliable (and was the
previous version of this script's approach, limited to a single
hand-picked "India" article per language). Instead, for each English
seed title we ask Wikipedia's own `langlinks` API for that language's
actual interlanguage-linked title, then fetch *that* article from the
corresponding-language Wikipedia. If a language edition has no article
on a given topic, that (language, topic) pair is skipped and reported
-- never silently faked.

Train/eval split
----------------
TRAIN_TITLES and EVAL_TITLES are disjoint English seed topics, so
data/eval/ is never the same document as anything in data/raw/ (an
earlier version of this script downloaded only "India" and pointed both
data/raw/ and data/eval/ at it, which meant "evaluation" was actually
just measuring fit to the training data).
"""

from __future__ import annotations

import time
import unicodedata
from pathlib import Path

import requests
import urllib3

# Real encyclopedia topics, not curated for any tokenizer-metric reason --
# a broad, India/South-Asia-leaning mix (higher odds of hi/te/ta coverage)
# plus a few universal topics, across geography/culture/history/nature.
TRAIN_TITLES: list[str] = [
    "India",
    "Cricket",
    "Mahatma Gandhi",
    "Yoga",
    "Himalayas",
    "Bollywood",
    "Taj Mahal",
    "Delhi",
    "Mumbai",
    "Chennai",
    "Hyderabad",
    "Ganges",
    "Indian cuisine",
    "Hinduism",
    "Buddhism",
    "Rice",
    "Monsoon",
    "Elephant",
    "Agriculture",
    "Tea",
    "China",
    "Pakistan",
    "Bangladesh",
    "Nepal",
    "Sri Lanka",
    "Asia",
    "Africa",
    "Indian Ocean",
    "Solar System",
    "Earth",
    "Photosynthesis",
    "Gravity",
    "Electricity",
    "Internet",
    "Mathematics",
    "Physics",
    "Chemistry",
    "Human body",
    "Climate change",
    "Lion",
    "Peacock",
    "Cow",
    "Wheat",
    "Banana",
    "Coconut",
    "Sugarcane",
    "Cotton",
    "Ancient India",
    "Mughal Empire",
    "Indus Valley Civilisation",
    "Ashoka",
    "Akbar",
    "Sanskrit",
    "Ramayana",
    "Mahabharata",
    "Diwali",
    "Holi",
    "Football",
    "Olympic Games",
    "Chess",
    "Railway transport in India",
]

# Disjoint from TRAIN_TITLES on purpose -- this is what makes data/eval/ a
# genuine held-out set instead of a copy of the training data.
EVAL_TITLES: list[str] = [
    "Kolkata",
    "Bangalore",
    "Silk",
    "Mango",
    "Tiger",
    "Indian classical music",
    "Rupee",
    "Lotus",
    "Sun",
    "Moon",
]

LANGUAGES = ("hi", "te", "ta")  # "en" needs no translation -- the seed title IS English

LANGLINKS_API = (
    "https://en.wikipedia.org/w/api.php"
    "?action=query&prop=langlinks&lllimit=500&redirects=1&format=json&titles={title}"
)
EXTRACT_API = (
    "https://{lang}.wikipedia.org/w/api.php"
    "?action=query&format=json&prop=extracts&explaintext=1&redirects=1&titles={title}"
)
HEADERS = {"User-Agent": "multilingual-bpe-tokenizer-assignment/1.0 (educational use)"}
REQUEST_DELAY_SECONDS = 1.0  # politeness -- avoid hammering Wikipedia's API
MAX_RETRIES = 5


def _get(url: str, verify: bool = True) -> requests.Response:
    return requests.get(url, timeout=60, headers=HEADERS, verify=verify)


def fetch(url: str) -> requests.Response:
    """GET `url`, verifying TLS certificates normally, with two failure
    modes handled so one flaky request doesn't abort the whole run:

    - **SSL verification failure**: some corporate networks run an
      HTTPS-intercepting proxy (e.g. Cisco Secure Access) whose root
      certificate is trusted by the OS but not by Python's bundled
      `certifi` CA store -- `curl` and browsers work fine, but
      `requests` fails with SSLCertVerificationError even though the
      network path itself is fine. Retried once without certificate
      verification; safe here because this is a one-off local script
      hitting Wikipedia's public, unauthenticated, read-only API, not a
      pattern to reuse for anything sensitive.
    - **429 Too Many Requests**: Wikipedia's API rate-limits bursts of
      requests. Retried with exponential backoff (honoring a
      `Retry-After` header if present) up to `MAX_RETRIES` times.
    """
    verify = True
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = _get(url, verify=verify)
        except requests.exceptions.SSLError:
            if not verify:
                raise
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            verify = False
            continue

        if response.status_code == 429 and attempt < MAX_RETRIES:
            wait = float(response.headers.get("Retry-After", 2 ** attempt))
            print(f"  (rate-limited, waiting {wait:.0f}s before retrying...)")
            time.sleep(wait)
            continue

        return response
    return response


def slugify(title: str) -> str:
    normalized = unicodedata.normalize("NFKD", title)
    ascii_only = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in ascii_only).strip("_") or "article"


def get_langlinks(en_title: str) -> dict[str, str]:
    """`{"hi": "<hindi title>", "te": "...", "ta": "..."}` for whichever of
    those three languages actually have an article linked from the
    English one. Missing languages are simply absent from the result.
    """
    response = fetch(LANGLINKS_API.format(title=en_title))
    response.raise_for_status()
    data = response.json()
    page = next(iter(data.get("query", {}).get("pages", {}).values()), {})
    links = page.get("langlinks", [])
    return {link["lang"]: link["*"] for link in links if link["lang"] in LANGUAGES}


def get_extract(lang: str, title: str) -> str:
    response = fetch(EXTRACT_API.format(lang=lang, title=title))
    response.raise_for_status()
    data = response.json()
    page = next(iter(data.get("query", {}).get("pages", {}).values()), {})
    return page.get("extract", "") or ""


def download_split(titles: list[str], out_dir: Path) -> None:
    for en_title in titles:
        slug = slugify(en_title)

        # Idempotent: if this title's English file already exists, assume
        # it (and whichever hi/te/ta translations were available last
        # time) were already fetched, and skip re-fetching it entirely --
        # lets this script be rerun cheaply after just adding a few new
        # titles to the lists above.
        if (out_dir / "en" / f"{slug}.txt").exists():
            print(f"'{en_title}': already downloaded -- skipping")
            continue

        print(f"'{en_title}':")

        try:
            en_text = get_extract("en", en_title)
            time.sleep(REQUEST_DELAY_SECONDS)
            titles_by_lang = {"en": en_title, **get_langlinks(en_title)}
            time.sleep(REQUEST_DELAY_SECONDS)
        except requests.exceptions.RequestException as exc:
            print(f"  Skipping '{en_title}' entirely -- request failed: {exc}")
            continue

        texts_by_lang = {"en": en_text}
        for lang in LANGUAGES:
            local_title = titles_by_lang.get(lang)
            if not local_title:
                print(f"  {lang}: no linked article -- skipping")
                continue
            try:
                texts_by_lang[lang] = get_extract(lang, local_title)
            except requests.exceptions.RequestException as exc:
                print(f"  {lang}: request failed ({exc}) -- skipping")
                continue
            finally:
                time.sleep(REQUEST_DELAY_SECONDS)

        for lang, text in texts_by_lang.items():
            if not text:
                print(f"  {lang}: empty extract -- skipping")
                continue
            folder = out_dir / lang
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{slug}.txt").write_text(text, encoding="utf-8")
            print(f"  {lang}: saved {folder / f'{slug}.txt'} ({len(text)} characters)")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent

    print("=== Training set (data/raw/) ===")
    download_split(TRAIN_TITLES, project_root / "data" / "raw")

    print("\n=== Held-out evaluation set (data/eval/) ===")
    download_split(EVAL_TITLES, project_root / "data" / "eval")

    print("\nDone!")
