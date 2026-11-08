"""Title and URL normalisation shared by the crawler and the recommender input resolver."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, quote, unquote, urlparse

WIKI_BASE_URL = "https://en.wikipedia.org/wiki/"
_WIKI_HOSTS = {"en.wikipedia.org", "en.m.wikipedia.org"}

# A colon alone does not mean "non-article" ("2001: A Space Odyssey" is an article),
# so only the prefixes MediaWiki actually reserves are rejected.
_RESERVED_PREFIXES = frozenset(
    {
        # Namespaces and their aliases.
        "media", "special", "talk", "user", "user talk", "wikipedia", "wp", "project",
        "wikipedia talk", "wt", "project talk", "file", "image", "file talk", "image talk",
        "mediawiki", "mediawiki talk", "template", "t", "template talk", "help", "h",
        "help talk", "category", "cat", "category talk", "portal", "p", "portal talk",
        "draft", "draft talk", "mos", "mos talk", "timedtext", "timedtext talk",
        "module", "module talk", "event", "event talk",
        # Interwiki prefixes, which also cannot begin a real article title.
        "wikt", "wiktionary", "commons", "c", "m", "meta", "metawikipedia", "s", "wikisource",
        "d", "wikidata", "n", "wikinews", "q", "wikiquote", "v", "wikiversity", "voy",
        "wikivoyage", "b", "wikibooks", "species", "mw", "foundation", "wmf", "phab",
    }
)  # fmt: skip

_NON_PROSE_TITLE = re.compile(
    r"^(?:lists? of|outline of|index of|glossary of|timeline of|bibliography of|deaths in)\b",
    re.IGNORECASE,
)


def normalize_title(title: str) -> str:
    """Return the MediaWiki form of a title: spaces, no fragment, capitalised first letter."""
    title = title.split("#", 1)[0].replace("_", " ")
    title = re.sub(r"\s+", " ", title).strip().lstrip(":").strip()
    return title[:1].upper() + title[1:]


def is_article_title(title: str) -> bool:
    """Tell whether a title lives in the main (article) namespace."""
    if not title:
        return False
    prefix, colon, _ = title.partition(":")
    return not (colon and prefix.strip().lower() in _RESERVED_PREFIXES)


def is_prose_article_title(title: str) -> bool:
    """Reject main-namespace pages that are link collections rather than prose."""
    return (
        is_article_title(title)
        and not title.endswith("(disambiguation)")
        and _NON_PROSE_TITLE.match(title) is None
    )


def title_to_url(title: str) -> str:
    """Build the canonical article URL for a title."""
    return WIKI_BASE_URL + quote(normalize_title(title).replace(" ", "_"), safe="/:@!$*,;()-._~")


def url_to_title(url: str) -> str | None:
    """Extract the normalised title from an English Wikipedia URL, or None if it is not one.

    Accepts desktop and mobile hosts, missing schemes, percent-encoding, fragments, query
    strings and the ``/w/index.php?title=`` form. Also accepts site-relative ``/wiki/`` links.
    """
    url = url.strip()
    if "://" not in url and not url.startswith("/"):
        url = "https://" + url
    parsed = urlparse(url)
    if parsed.netloc and parsed.netloc.lower().removeprefix("www.") not in _WIKI_HOSTS:
        return None
    if parsed.path.startswith("/wiki/"):
        raw_title = parsed.path.removeprefix("/wiki/")
    elif parsed.path == "/w/index.php":
        raw_title = parse_qs(parsed.query).get("title", [""])[0]
    else:
        return None
    return normalize_title(unquote(raw_title)) or None


def canonicalize_url(url: str) -> str | None:
    """Map any variant of an article URL to its canonical form, or None if it is not one."""
    title = url_to_title(url)
    return title_to_url(title) if title else None
