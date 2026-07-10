"""Corpus-cleaning utilities for the multilingual BPE tokenizer project.

Pipeline position
------------------
This module is the very first stage of the training pipeline:

    raw HTML / MediaWiki source text  --[this module]-->  clean plain text

It intentionally does **not** do anything BPE-specific (no pretokenization,
no byte/codepoint encoding, no merge logic). That belongs to later stages
(``pretokenizer.py``, ``trainer.py``, ``tokenizer.py``).

Design contract
----------------
1. **NFC normalization** is applied to every string that passes through
   :func:`clean_text`, so that visually-identical input always yields an
   identical output regardless of how it was originally encoded.
2. **All Unicode characters are preserved** unless they are structurally
   identified as markup (an HTML tag, a MediaWiki template/table/link
   delimiter, etc.). We never filter, whitelist, or rewrite based on
   *script* -- English, Hindi, Telugu and Tamil text all flow through the
   exact same code path.
3. **ZWJ (U+200D) and ZWNJ (U+200C) are preserved.** These invisible
   characters control conjunct/ligature formation in Devanagari, Telugu
   and Tamil and are linguistically significant, not "junk whitespace" --
   none of the regexes below match them.
4. **Punctuation and numbers (any script) are preserved as content.**
   Only the *markup syntax* is removed; the human-readable text that the
   markup was wrapping/decorating is kept.
5. A few MediaWiki constructs are removed *together with their content*
   because they are not article prose at all (HTML comments, ``<ref>``
   citations, ``{{templates}}``, ``{|tables|}``, file/image/category
   links). This is a deliberate editorial choice, not accidental content
   loss -- see the docstring on each helper for the reasoning.

Everything here is pure stdlib (``re``, ``html``, ``unicodedata``) -- no
tokenizer or NLP libraries are used, per the assignment constraints.
"""

from __future__ import annotations

import html
import re
import unicodedata

__all__ = [
    "normalize_unicode",
    "unescape_html_entities",
    "remove_comments",
    "remove_ref_tags",
    "remove_templates",
    "remove_tables",
    "process_wiki_links",
    "process_external_links",
    "strip_wiki_formatting",
    "strip_html_tags",
    "clean_whitespace",
    "clean_text",
    "preprocess_corpus",
]

# Invisible joiners that MUST survive every transformation in this module.
ZWNJ = "\u200c"
ZWJ = "\u200d"

# Wiki-link prefixes that mark media/metadata references rather than
# in-line prose links -- these are dropped entirely (see process_wiki_links).
_MEDIA_LINK_PREFIXES = (
    "file:",
    "image:",
    "category:",
    "media:",
)


# ---------------------------------------------------------------------------
# 1. Unicode normalization
# ---------------------------------------------------------------------------


def normalize_unicode(text: str) -> str:
    """Apply NFC (canonical composition) normalization.

    NFC is used (rather than NFD/NFKC/NFKD) because it composes decomposed
    character sequences into their canonical precomposed form without the
    lossy *compatibility* rewriting that NFKC/NFKD perform -- important for
    preserving exact script fidelity across English, Hindi, Telugu and
    Tamil text. This must be applied identically during training and at
    inference time.
    """
    return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------------------
# 2. HTML entity decoding
# ---------------------------------------------------------------------------


def unescape_html_entities(text: str) -> str:
    """Decode HTML entities (``&amp;``, ``&nbsp;``, ``&#2350;``, ...).

    This turns the *markup representation* of a character back into the
    character itself, e.g. ``&amp;`` -> ``&``. The resulting character is
    then treated as ordinary content by every later step (it is not
    special-cased or stripped).
    """
    return html.unescape(text)


# ---------------------------------------------------------------------------
# 3. Structures removed together with their content (not prose)
# ---------------------------------------------------------------------------

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_REF_PAIR_RE = re.compile(r"<ref\b[^>]*>.*?</ref\s*>", re.DOTALL | re.IGNORECASE)
_REF_SELF_CLOSING_RE = re.compile(r"<ref\b[^>]*/>", re.IGNORECASE)


def remove_comments(text: str) -> str:
    """Remove HTML comments (``<!-- ... -->``), including their content.

    Comments are, by definition, not rendered/readable article text.
    """
    return _COMMENT_RE.sub("", text)


def remove_ref_tags(text: str) -> str:
    """Remove ``<ref>...</ref>`` citations, including their content.

    Reference/citation markup (footnote text, URLs, bibliographic
    metadata) is not article prose, so -- unlike generic HTML formatting
    tags -- both the tag *and* its content are dropped.
    """
    text = _REF_PAIR_RE.sub("", text)
    text = _REF_SELF_CLOSING_RE.sub("", text)
    return text


def _remove_balanced(text: str, open_delim: str, close_delim: str) -> str:
    """Remove every (possibly nested) ``open_delim ... close_delim`` span.

    MediaWiki templates (``{{...}}``) and tables (``{|...|}``) can nest
    arbitrarily deep (e.g. a citation template embedding another
    template), which a regular expression cannot match correctly in
    general. This is a small hand-written balanced-delimiter scanner
    instead -- linear time, no external parsing library.
    """
    result: list[str] = []
    depth = 0
    i = 0
    n = len(text)
    open_len = len(open_delim)
    close_len = len(close_delim)
    while i < n:
        if text.startswith(open_delim, i):
            depth += 1
            i += open_len
        elif depth > 0 and text.startswith(close_delim, i):
            depth -= 1
            i += close_len
        elif depth == 0:
            result.append(text[i])
            i += 1
        else:
            i += 1
    return "".join(result)


def remove_templates(text: str) -> str:
    """Remove ``{{template}}`` invocations, including their content.

    Templates render infoboxes, citation metadata, and other structured
    data -- not free-form prose -- so they are dropped wholesale.
    """
    return _remove_balanced(text, "{{", "}}")


def remove_tables(text: str) -> str:
    """Remove ``{|...|}`` wikitable markup, including its content.

    Tables are structured/tabular data rather than sentences, so they are
    unsuitable as BPE training text and are dropped wholesale.
    """
    return _remove_balanced(text, "{|", "|}")


# ---------------------------------------------------------------------------
# 4. Wiki links -- transformed (kept as prose), not simply deleted
# ---------------------------------------------------------------------------


def _render_wiki_link(inner: str) -> str:
    """Decide what an ``[[...]]`` link's inner text renders to.

    - ``[[File:...]]`` / ``[[Image:...]]`` / ``[[Category:...]]`` /
      ``[[Media:...]]`` are metadata references (not prose) and are
      dropped entirely, including any caption text they contain.
    - ``[[Target|Display]]`` renders as ``Display`` (mirrors MediaWiki's
      own rendering rule of using the last pipe-segment as visible text).
    - ``[[Target]]`` renders as ``Target``.
    """
    stripped = inner.strip()
    if stripped.lower().startswith(_MEDIA_LINK_PREFIXES):
        return ""
    parts = stripped.split("|")
    return parts[-1].strip()


def process_wiki_links(text: str) -> str:
    """Replace ``[[...]]`` wiki links with their rendered display text.

    A depth-aware scanner is used (rather than a regex) so that a
    caption containing a nested link, e.g.
    ``[[File:x.png|thumb|see [[Real Link|here]]]]``, is parsed as one
    outer span instead of breaking on the first ``]]``.
    """
    result: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("[[", i):
            depth = 1
            j = i + 2
            while j < n and depth > 0:
                if text.startswith("[[", j):
                    depth += 1
                    j += 2
                elif text.startswith("]]", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            inner_end = j - 2 if depth == 0 else j
            inner = text[i + 2 : inner_end]
            result.append(_render_wiki_link(inner))
            i = j
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


_EXTERNAL_LINK_RE = re.compile(r"\[https?://[^\s\]]+(?:\s+([^\]]*))?\]")


def process_external_links(text: str) -> str:
    """Replace ``[url display text]`` with ``display text``.

    A bare external link with no display text, e.g. ``[http://x.com]``,
    renders to nothing -- a raw URL is not prose content.
    """

    def _replace(match: re.Match[str]) -> str:
        display = match.group(1)
        return display.strip() if display else ""

    return _EXTERNAL_LINK_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# 5. Inline wiki formatting -- markup stripped, text content kept
# ---------------------------------------------------------------------------

_BOLD_ITALIC_RE = re.compile(r"'{5}(.*?)'{5}")
_BOLD_RE = re.compile(r"'{3}(.*?)'{3}")
_ITALIC_RE = re.compile(r"'{2}(.*?)'{2}")
_HEADING_RE = re.compile(r"^={2,6}\s*(.*?)\s*={2,6}[ \t]*$", re.MULTILINE)
_HORIZONTAL_RULE_RE = re.compile(r"^-{4,}[ \t]*$", re.MULTILINE)
_LIST_MARKER_RE = re.compile(r"^[ \t]*[*#;:]+[ \t]*", re.MULTILINE)


def strip_wiki_formatting(text: str) -> str:
    """Strip bold/italic quotes, heading ``=`` signs, horizontal rules and
    leading list/definition markers (``*``, ``#``, ``;``, ``:``), while
    keeping the human-readable text they decorate.

    Longest quote-runs are matched first (bold-italic before bold before
    italic) so that e.g. ``'''''x'''''`` is not mis-parsed as italic
    markup with leftover quote characters.
    """
    text = _BOLD_ITALIC_RE.sub(r"\1", text)
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)
    text = _HEADING_RE.sub(r"\1", text)
    text = _HORIZONTAL_RULE_RE.sub("", text)
    text = _LIST_MARKER_RE.sub("", text)
    return text


# ---------------------------------------------------------------------------
# 6. Generic HTML tags -- tag stripped, inner text kept
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def strip_html_tags(text: str) -> str:
    """Strip any remaining generic HTML tags (``<b>``, ``<i>``, ``<br/>``,
    ``<div>``, ``<span>``, ...), keeping the text between/around them.

    This must run *after* :func:`remove_comments` and :func:`remove_ref_tags`,
    which need to delete their tag's *content* too -- by the time this
    function runs, only harmless inline formatting tags should remain.
    """
    return _HTML_TAG_RE.sub("", text)


# ---------------------------------------------------------------------------
# 7. Whitespace cleanup (only whitespace *introduced by markup removal*)
# ---------------------------------------------------------------------------


def clean_whitespace(text: str) -> str:
    """Collapse the extra blank lines/runs of spaces left behind by markup
    removal, without touching any non-whitespace character.

    Note: ``\\u200c``/``\\u200d`` (ZWNJ/ZWJ) are Unicode category *Cf*
    (format), not whitespace, so ``[ \\t]+`` and Python's line-splitting
    on ``\\n`` never match or remove them.
    """
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(" \t\n")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def clean_text(raw_text: str) -> str:
    """Turn raw HTML/MediaWiki source text into clean plain text.

    Stage order matters:
      1. Normalize unicode and decode HTML entities first, so every later
         regex/scanner operates on canonical, already-decoded characters.
      2. Remove non-prose structures that must take their content with
         them (comments, refs, templates, tables).
      3. Transform links into their rendered display text (or drop
         media/category links).
      4. Strip remaining inline wiki formatting and generic HTML tags,
         keeping their inner text.
      5. Clean up whitespace left behind by the removals above.
    """
    text = normalize_unicode(raw_text)
    text = unescape_html_entities(text)
    text = remove_comments(text)
    text = remove_ref_tags(text)
    text = remove_templates(text)
    text = remove_tables(text)
    text = process_wiki_links(text)
    text = process_external_links(text)
    text = strip_wiki_formatting(text)
    text = strip_html_tags(text)
    text = clean_whitespace(text)
    return text


def preprocess_corpus(raw_documents: list[str]) -> list[str]:
    """Apply :func:`clean_text` to a batch of documents, dropping any that
    clean down to nothing (e.g. a document that was pure markup/metadata).
    """
    cleaned = (clean_text(doc) for doc in raw_documents)
    return [doc for doc in cleaned if doc]


if __name__ == "__main__":
    import sys

    raw = sys.stdin.read()
    sys.stdout.write(clean_text(raw))
