"""Match what a user typed, a title or a URL, to an article of the corpus.

Titles are compared in MediaWiki form first ("information_retrieval" -> "Information retrieval"), then
loosely: case, accents and punctuation are ignored, so "SAO PAULO" finds "São Paulo". Anything still
unmatched gets close-title suggestions.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from wiki_recommender.crawler.titles import normalize_title, url_to_title

_NON_WORD = re.compile(r"[\W_]+")
_SUGGESTION_CUTOFF = 0.6


def loose_key(title: str) -> str:
    """Casefolded, accent-free title with punctuation reduced to single spaces."""
    decomposed = unicodedata.normalize("NFKD", title.casefold())
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return _NON_WORD.sub(" ", stripped).strip()


def looks_like_url(text: str) -> bool:
    text = text.strip().lower()
    return text.startswith(("http://", "https://", "www.", "/wiki/")) or "wikipedia.org/" in text


@dataclass(frozen=True)
class Query:
    """One user input, read as a title. `title` is None if the input is a URL that is not an article."""

    text: str
    title: str | None
    is_url: bool


def parse_query(text: str) -> Query:
    if looks_like_url(text):
        return Query(text, url_to_title(text), True)
    title = normalize_title(text)
    return Query(text, title or None, False)


class TitleResolver:
    def __init__(self, titles: Sequence[str]) -> None:
        self.titles = list(titles)
        self._exact = {title: index for index, title in enumerate(self.titles)}
        self._loose: dict[str, int] = {}
        for index, title in enumerate(self.titles):
            self._loose.setdefault(loose_key(title), index)
        self._loose_keys = list(self._loose)

    def find(self, title: str) -> int | None:
        """Index of the article with this title, or None. Accepts any spelling `loose_key` forgives."""
        title = normalize_title(title)
        if title in self._exact:
            return self._exact[title]
        return self._loose.get(loose_key(title))

    def suggest(self, text: str, limit: int = 3) -> list[str]:
        """Titles that look like the input: those starting with it as a whole word, then close spellings.

        "Mercury" thus offers "Mercury (planet)" before "Perjury".
        """
        query = parse_query(text)
        key = loose_key(query.title or text)
        if not key:
            return []
        prefixed = sorted((c for c in self._loose_keys if c.startswith(key + " ")), key=len)
        close = difflib.get_close_matches(key, self._loose_keys, n=limit, cutoff=_SUGGESTION_CUTOFF)
        matches = list(dict.fromkeys(prefixed[:limit] + close))[:limit]
        return [self.titles[self._loose[match]] for match in matches]
