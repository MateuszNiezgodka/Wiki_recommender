"""Remove the wiki residue the crawler's HTML cleaning cannot see, and split out the article's parts."""

import html
import re

CATEGORY_TOKEN_PREFIX = "cat:"

_BOILERPLATE_LINE = re.compile(
    r"^IPA help$|"
    r"^Musical scores are temporarily disabled|"
    r"%[0-9A-Fa-f]{2}%[0-9A-Fa-f]{2}|"  # a percent-encoded template name rendered as text
    r"\{\{\{"  # an unfilled template parameter
)
_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?/?>")
_WIKI_BRACKETS = re.compile(r"\[\[|\]\]|\{\{|\}\}")
_SPACES = re.compile(r"[ \t ]+")

_MIN_DEFINITION_WORDS = 15

_ADMINISTRATIVE_CATEGORY = re.compile(
    r"^(?:Living people|Websites which mirror Wikipedia)$|"
    r"\b[Aa]rticles\b(?! of )"  # "Main topic articles", but not "Articles of Confederation"
)


def clean_wiki_text(text: str) -> str:
    """Drop boilerplate lines and inline markup, keeping one paragraph per line."""
    lines = []
    for line in html.unescape(text).splitlines():
        if _BOILERPLATE_LINE.search(line):
            continue
        line = _WIKI_BRACKETS.sub(" ", _HTML_TAG.sub(" ", _URL.sub(" ", line)))
        line = _SPACES.sub(" ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def first_paragraph(lead: str) -> str:
    """Return the article's definition: the first lead paragraph long enough to be a sentence.

    Some leads open with a fragment left by a removed formula ("For even numbers, divide by 2;").
    """
    for line in lead.splitlines():
        if len(line.split()) >= _MIN_DEFINITION_WORDS:
            return line
    return lead


def split_lead_and_body(raw_text: str, summary: str) -> tuple[str, str]:
    """Separate the lead from the rest. The crawler stores `summary` as a prefix of `raw_text`."""
    if raw_text.startswith(summary):
        return summary, raw_text[len(summary) :].lstrip()
    return summary, raw_text


def informative_categories(categories: list[str]) -> list[str]:
    """Drop categories that describe Wikipedia's own maintenance, not the subject."""
    return [category for category in categories if not _ADMINISTRATIVE_CATEGORY.search(category)]


def clean_section_label(label: str) -> str:
    """Remove the HTML anchors some Vital articles list headings carry."""
    return _SPACES.sub(" ", _HTML_TAG.sub(" ", label)).strip()
