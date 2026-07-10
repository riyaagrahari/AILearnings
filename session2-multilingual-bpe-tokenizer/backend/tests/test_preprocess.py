"""Unit tests for bpe.preprocess.

Each test targets one contract from the preprocess.py module docstring:
NFC normalization, ZWJ/ZWNJ preservation, markup-only removal, and
punctuation/number preservation -- across English, Hindi, Telugu and
Tamil sample text.
"""

import os
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bpe.preprocess import (  # noqa: E402
    ZWJ,
    ZWNJ,
    clean_text,
    clean_whitespace,
    process_external_links,
    process_wiki_links,
    remove_comments,
    remove_ref_tags,
    remove_tables,
    remove_templates,
    strip_html_tags,
    strip_wiki_formatting,
)


# ---------------------------------------------------------------------------
# 1. NFC normalization
# ---------------------------------------------------------------------------


def test_nfc_normalizes_decomposed_latin():
    decomposed = "e\u0301cole"  # "e" + combining acute accent
    result = clean_text(decomposed)
    assert result == unicodedata.normalize("NFC", decomposed)
    assert result == "\u00e9cole"  # precomposed "é"


def test_nfc_is_idempotent_on_already_composed_text():
    composed = "café"
    assert clean_text(composed) == composed


# ---------------------------------------------------------------------------
# 2. ZWJ / ZWNJ preservation (Devanagari / Telugu / Tamil conjuncts)
# ---------------------------------------------------------------------------


def test_zwnj_preserved_in_hindi_text():
    # Devanagari conjunct commonly written with an explicit ZWNJ.
    text = f"क्{ZWNJ}ष"
    result = clean_text(text)
    assert ZWNJ in result
    assert result == text


def test_zwj_preserved_in_telugu_text():
    text = f"త్{ZWJ}ర"
    result = clean_text(text)
    assert ZWJ in result
    assert result == text


def test_zwnj_survives_whitespace_cleanup():
    text = f"a  {ZWNJ}  b"
    result = clean_whitespace(text)
    assert ZWNJ in result


# ---------------------------------------------------------------------------
# 3. Comments / ref tags / templates / tables removed with their content
# ---------------------------------------------------------------------------


def test_html_comment_removed():
    text = "Before<!-- a hidden note -->After"
    assert remove_comments(text) == "BeforeAfter"


def test_ref_tag_pair_removed_with_content():
    text = "Water is wet.<ref>Some Journal, 2020, pp. 1-2</ref> Fire is hot."
    assert remove_ref_tags(text) == "Water is wet. Fire is hot."


def test_ref_self_closing_removed():
    text = "See note.<ref name=\"x\"/> Continue."
    assert remove_ref_tags(text) == "See note. Continue."


def test_template_removed_including_nested_templates():
    text = "Intro {{cite web|title={{small|Nested}}|url=http://x}} outro."
    assert remove_templates(text) == "Intro  outro."


def test_table_removed():
    text = "Before\n{|\n|Cell1||Cell2\n|}\nAfter"
    result = remove_tables(text)
    assert "Cell1" not in result
    assert "Before" in result and "After" in result


# ---------------------------------------------------------------------------
# 4. Wiki links -- transformed, not blindly deleted
# ---------------------------------------------------------------------------


def test_simple_wiki_link_keeps_target_text():
    assert process_wiki_links("Visit [[Hyderabad]] today.") == "Visit Hyderabad today."


def test_piped_wiki_link_keeps_display_text():
    text = "Visit [[Hyderabad|the city]] today."
    assert process_wiki_links(text) == "Visit the city today."


def test_category_link_removed_entirely():
    assert process_wiki_links("Text.[[Category:Cities]]") == "Text."


def test_file_link_with_nested_link_removed_entirely():
    text = "Photo [[File:x.png|thumb|see [[Real Link|here]]]] caption."
    assert process_wiki_links(text) == "Photo  caption."


def test_external_link_with_display_text():
    text = "Source [http://example.com official site] confirms it."
    assert process_external_links(text) == "Source official site confirms it."


def test_bare_external_link_removed():
    text = "See [http://example.com] for more."
    assert process_external_links(text) == "See  for more."


# ---------------------------------------------------------------------------
# 5. Inline wiki formatting stripped, text kept
# ---------------------------------------------------------------------------


def test_bold_and_italic_markup_stripped():
    assert strip_wiki_formatting("This is '''bold''' and ''italic''.") == (
        "This is bold and italic."
    )


def test_heading_markup_stripped():
    assert strip_wiki_formatting("== History ==\nSome text.") == "History\nSome text."


def test_horizontal_rule_removed():
    result = strip_wiki_formatting("Para one.\n----\nPara two.")
    assert "----" not in result
    assert "Para one." in result and "Para two." in result


def test_list_marker_stripped_but_text_kept():
    assert strip_wiki_formatting("* First item\n# Second item") == (
        "First item\nSecond item"
    )


# ---------------------------------------------------------------------------
# 6. Generic HTML tags stripped, inner text kept
# ---------------------------------------------------------------------------


def test_generic_html_tag_stripped_content_kept():
    assert strip_html_tags("This is <b>bold</b> and <i>italic</i>.") == (
        "This is bold and italic."
    )


def test_self_closing_html_tag_removed():
    assert strip_html_tags("Line one.<br/>Line two.") == "Line one.Line two."


# ---------------------------------------------------------------------------
# 7. HTML entity decoding
# ---------------------------------------------------------------------------


def test_html_entities_decoded():
    text = "Tom &amp; Jerry &lt;3"
    assert clean_text(text) == "Tom & Jerry <3"


# ---------------------------------------------------------------------------
# 8. Punctuation and numbers are preserved as content
# ---------------------------------------------------------------------------


def test_punctuation_preserved():
    text = "Cost: $12.50 (approx.), see note!"
    assert clean_text(text) == text


def test_arabic_digits_preserved():
    assert clean_text("There are 42 apples.") == "There are 42 apples."


def test_devanagari_digits_preserved():
    text = "मूल्य \u0967\u0968\u0969 रुपये है।"  # Devanagari digits + danda
    assert clean_text(text) == text


def test_telugu_digits_and_punctuation_preserved():
    text = "ధర \u0c67\u0c68 రూపాయలు."
    assert clean_text(text) == text


def test_tamil_digits_and_punctuation_preserved():
    text = "விலை \u0be7\u0be8 ரூபாய்."
    assert clean_text(text) == text


# ---------------------------------------------------------------------------
# 9. Whitespace cleanup
# ---------------------------------------------------------------------------


def test_repeated_blank_lines_collapsed():
    text = "Para one.\n\n\n\n\nPara two."
    assert clean_whitespace(text) == "Para one.\n\nPara two."


def test_trailing_and_repeated_spaces_collapsed():
    text = "Hello    world.   \nSecond   line.   "
    assert clean_whitespace(text) == "Hello world.\nSecond   line.".replace(
        "Second   line.", "Second line."
    )


# ---------------------------------------------------------------------------
# 10. Full-pipeline integration tests, one per target language
# ---------------------------------------------------------------------------


def test_full_pipeline_english_document():
    raw = (
        "<!-- draft -->{{infobox|x=1}}== Overview ==\n"
        "'''New Delhi''' is the capital of India.<ref>Census 2021</ref> "
        "See [[India|the country]] and [http://example.com official site].\n"
        "* Population: 32 million\n"
        "----\n"
    )
    result = clean_text(raw)
    assert "infobox" not in result
    assert "Census 2021" not in result
    assert "New Delhi is the capital of India." in result
    assert "the country" in result
    assert "official site" in result
    assert "Population: 32 million" in result
    assert "----" not in result


def test_full_pipeline_hindi_document():
    raw = (
        f"'''दिल्ली'''<ref>जनगणना</ref> भारत की राजधानी है। "
        f"क्{ZWNJ}ष जैसे संयुक्ताक्षर सही रहते हैं। मूल्य \u0967\u0968 है।"
    )
    result = clean_text(raw)
    assert "जनगणना" not in result
    assert "दिल्ली भारत की राजधानी है।" in result
    assert ZWNJ in result
    assert "\u0967\u0968" in result


def test_full_pipeline_telugu_document():
    raw = f"'''హైదరాబాద్'''<ref>సెన్సస్</ref> తెలంగాణ రాజధాని. త్{ZWJ}ర సరిగ్గా ఉంది."
    result = clean_text(raw)
    assert "సెన్సస్" not in result
    assert "హైదరాబాద్ తెలంగాణ రాజధాని." in result
    assert ZWJ in result


def test_full_pipeline_tamil_document():
    raw = "'''சென்னை'''<ref>கணக்கெடுப்பு</ref> தமிழ்நாட்டின் தலைநகரம். விலை \u0be7\u0be8 ரூபாய்."
    result = clean_text(raw)
    assert "கணக்கெடுப்பு" not in result
    assert "சென்னை தமிழ்நாட்டின் தலைநகரம்." in result
    assert "\u0be7\u0be8" in result
