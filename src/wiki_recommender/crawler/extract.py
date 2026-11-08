"""Turn the HTML returned by ``action=parse`` into clean article text and metadata."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from bs4 import BeautifulSoup
from bs4.element import Tag

from wiki_recommender.crawler.titles import is_article_title, url_to_title

# Everything that is not running prose. Tables go entirely: on Wikipedia they hold infoboxes,
# navigation and statistics, whose tokens would only add noise to a text-similarity model.
_BOILERPLATE_SELECTOR = ", ".join(
    [
        "style", "script", "link", "meta", "table", "figure", "img", "audio", "video", "math",
        "sup.reference", "sup.noprint", ".mw-editsection", ".mw-empty-elt", ".shortdescription",
        ".hatnote", ".dablink", ".rellink", ".navbox", ".vertical-navbox", ".navbox-styles",
        ".sidebar", ".infobox", ".ambox", ".mbox-small", ".side-box", ".sistersitebox",
        ".portalbox", ".portal", ".toc", ".tocright", ".thumb", ".tmulti", ".gallery",
        ".reflist", ".refbegin", ".references", ".mw-references-wrap", ".mw-cite-backlink",
        ".noprint", ".metadata", ".mwe-math-element", ".IPA", ".ext-phonos", ".catlinks",
        ".timeline-wrapper", ".stub", ".error", ".mw-ext-cite-error",
    ]
)  # fmt: skip

_EXCLUDED_SECTIONS = frozenset(
    {
        "see also", "references", "citations", "footnotes", "explanatory notes",
        "notes and references", "references and notes",
        "footnotes and references", "citations and notes", "notes and citations",
        "references and sources", "sources and references", "cited sources", "works cited",
        "general references", "general bibliography", "further reading", "external links",
        "external link", "bibliography", "selected bibliography", "select bibliography",
        "selected works", "list of works", "selected publications",
        "discography", "selected discography", "filmography", "selected filmography",
        "gallery", "image gallery",
        "general and cited references", "general and cited sources", "general sources",
        "cited references", "sources cited", "sources and further reading",
        "references and further reading", "notes, references and sources", "notes and sources",
        "explanatory footnotes", "print sources", "additional sources", "photo gallery",
    }
)  # fmt: skip

# These hold a reference list in some articles and real prose in others: "Literature" is a list
# of books in one article and a country's literary history in another. Only lists are dropped.
_PROSE_ONLY_SECTIONS = frozenset(
    {
        "literature",
        "sources",
        "primary sources",
        "secondary sources",
        "publications",
        "notes",
        "note",
        "reference",
    }
)

_HEADING_TAGS = ["h2", "h3", "h4", "h5", "h6"]
_LIST_TAGS = ["ul", "ol", "dl"]
_BLOCK_TAGS = ["p", "blockquote", *_LIST_TAGS]
_CITATION_LEFTOVER = re.compile(
    r"\[(?:\d+|[a-z]|note \d+|nb \d+|citation needed|clarification needed|when\?|who\?|"
    r"better source needed|dubious[^\]]*|failed verification)\]",
    re.IGNORECASE,
)

_MAX_BIBLIOGRAPHY_ENTRY_WORDS = 60

# Bibliography entries also appear outside reference sections, for example in lists of an
# author's books. These identifiers and archive notices rarely occur in prose.
_NON_PROSE_LINE = re.compile(
    r"\bISBN[ :\-]*\d|\bISSN[ :]*\d|\bOCLC[ :]*\d|\bdoi:\s*10\.|\bPMID[ :]*\d|\bJSTOR[ :]*\d|"
    r"\bS2CID[ :]*\d|\bRetrieved (?:on )?(?:\d{1,2} [A-Z][a-z]+|[A-Z][a-z]+ \d{1,2},?) \d{4}|"
    r"\bArchived from the original|^(?:See also|Main articles?|Further information):"
)


@dataclass(frozen=True)
class ExtractedArticle:
    text: str
    summary: str
    sections: list[str]
    outlinks: list[str]


def clean_text(text: str) -> str:
    """Normalise a fragment of article text to single-spaced NFC prose."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\u00ad", "")
    text = _CITATION_LEFTOVER.sub("", text)
    text = re.sub(r"\s+", " ", text)
    # Removing pronunciations and formulas can leave "( ; born 1950)" or "word ,".
    text = re.sub(r"\(\s*[;,:]\s*", "(", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s+([,.;:!?)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return re.sub(r" {2,}", " ", text).strip()


def is_non_prose_line(line: str) -> bool:
    """Tell whether a line of text is a bibliography entry or a leftover navigation note.

    Long lines are kept even when they match: they are paragraphs with a stray citation, and
    losing the paragraph would cost far more than keeping the citation.
    """
    return len(line.split()) <= _MAX_BIBLIOGRAPHY_ENTRY_WORDS and _NON_PROSE_LINE.search(line) is not None


def _block_items(block: Tag) -> list[Tag]:
    """Split a block into its text units: list items for a list, otherwise the block itself."""
    if block.name in _LIST_TAGS:
        return block.find_all(["li", "dt", "dd"], recursive=False)
    return [block]


def _item_links(item: Tag) -> list[str]:
    titles = (
        url_to_title(a["href"]) for a in item.find_all("a", href=True) if a["href"].startswith("/wiki/")
    )
    return [title for title in titles if title and is_article_title(title)]


def extract_article(html: str) -> ExtractedArticle:
    """Extract prose, lead, kept section headings and in-text links from article HTML.

    Reference-like trailing sections are dropped with all their subsections, and links are
    read only from the text that is kept, so navigation templates never count as outlinks.
    """
    soup = BeautifulSoup(html, "lxml")
    root = soup.select_one(".mw-parser-output") or soup
    for element in root.select(_BOILERPLATE_SELECTOR):
        if not element.decomposed:
            element.decompose()

    lines: list[str] = []
    lead: list[str] = []
    sections: list[str] = []
    links: dict[str, None] = {}
    skip_level: int | None = None
    prose_only = False
    pending_heading: str | None = None
    in_lead = True

    for element in root.find_all([*_HEADING_TAGS, *_BLOCK_TAGS]):
        if element.name in _HEADING_TAGS:
            in_lead = False
            level = int(element.name[1])
            if skip_level is not None and level > skip_level:
                continue
            heading = clean_text(element.get_text())
            normalized = heading.lower().rstrip(":")
            prose_only = normalized in _PROSE_ONLY_SECTIONS
            pending_heading = heading if prose_only else None
            if prose_only or normalized in _EXCLUDED_SECTIONS:
                skip_level = level
            else:
                skip_level = None
                sections.append(heading)
        elif (skip_level is None or (prose_only and element.name == "p")) and (
            element.find_parent(_BLOCK_TAGS) is None
        ):
            for item in _block_items(element):
                line = clean_text(item.get_text())
                if line and not is_non_prose_line(line):
                    if pending_heading is not None:
                        sections.append(pending_heading)
                        pending_heading = None
                    lines.append(line)
                    if in_lead:
                        lead.append(line)
                    links.update(dict.fromkeys(_item_links(item)))

    return ExtractedArticle(
        text="\n".join(lines),
        summary="\n".join(lead),
        sections=sections,
        outlinks=list(links),
    )
