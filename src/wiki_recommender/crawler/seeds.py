"""Choose which articles to crawl, from Wikipedia's "Vital articles" lists.

These editor-maintained lists rank articles by importance (level 1 = most vital) and sort them
into 11 topics and over a thousand subsections, so they give broad coverage with no reliance
on link-graph crawling, which tends to drift towards whatever the seed pages link to most.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any, Protocol

from wiki_recommender.crawler.titles import is_prose_article_title, normalize_title

LIST_ROOT = "Vital articles/Level {level}"
LIST_LEVELS = (4, 5)
CORE_MAX_LEVEL = 3

_NON_TOPIC_SUBPAGES = {"Article alerts", "Removed", "Nav bar"}
_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")
_LINK = re.compile(r"\[\[([^\[\]|#]+)(?:#[^\[\]|]*)?(?:\|[^\[\]]*)?\]\]")
_LEVEL_MARKER = re.compile(r"\[\[Wikipedia:Vital articles/Level[ /](\d)")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


class ApiGetter(Protocol):
    def get(self, **params: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Candidate:
    title: str
    topic: str
    section: str
    vital_level: int


def parse_vital_list(wikitext: str, *, topic: str, list_level: int, subpage: str = "") -> Iterator[Candidate]:
    """Yield one candidate per article linked from a list item of a vital-articles page.

    An item without a level marker belongs to the level of the list that contains it.
    """
    headings: dict[int, str] = {}
    for line in _COMMENT.sub("", wikitext).splitlines():
        if match := _HEADING.match(line):
            depth = len(match[1])
            headings = {d: text for d, text in headings.items() if d < depth}
            headings[depth] = match[2]
        elif line.startswith(("#", "*")):
            marker = _LEVEL_MARKER.search(line)
            level = int(marker[1]) if marker else list_level
            path = [subpage] if subpage else []
            path += [headings[depth] for depth in sorted(headings)]
            for raw_title in _LINK.findall(line):
                title = normalize_title(raw_title)
                if is_prose_article_title(title):
                    yield Candidate(title, topic, " > ".join(path), level)


def list_pages(client: ApiGetter, level: int) -> list[tuple[str, str, str]]:
    """Return (page title, topic, subpage path) for every topic list of a level."""
    root = LIST_ROOT.format(level=level)
    response = client.get(
        action="query",
        list="allpages",
        apnamespace=4,
        apprefix=f"{root}/",
        apfilterredir="nonredirects",
        aplimit="max",
    )
    pages = []
    for entry in response["query"]["allpages"]:
        topic, _, subpage = entry["title"].removeprefix(f"Wikipedia:{root}/").partition("/")
        if topic not in _NON_TOPIC_SUBPAGES:
            pages.append((entry["title"], topic, subpage))
    return pages


def discover_candidates(client: ApiGetter, levels: Iterable[int] = LIST_LEVELS) -> list[Candidate]:
    """Download the lists of every level and merge them into one deduplicated pool."""
    found: list[Candidate] = []
    for level in levels:
        for page, topic, subpage in list_pages(client, level):
            parsed = client.get(action="parse", page=page, prop="wikitext")["parse"]
            found.extend(parse_vital_list(parsed["wikitext"], topic=topic, list_level=level, subpage=subpage))
    return merge_candidates(found)


def merge_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Keep one entry per title: the listing that ranks it highest, decided deterministically."""
    best: dict[str, Candidate] = {}
    for candidate in sorted(candidates, key=lambda c: (c.vital_level, c.topic, c.section, c.title)):
        best.setdefault(candidate.title, candidate)
    return list(best.values())


def _round_robin_by_section(candidates: Sequence[Candidate], rng: random.Random) -> list[Candidate]:
    """Order candidates so that every section contributes before any contributes twice.

    Lists run alphabetically or by era, so sections and the articles within each section are
    shuffled with a fixed seed to avoid biasing a cut-off towards their start.
    """
    by_section: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in sorted(candidates, key=lambda c: (c.section, c.title)):
        by_section[candidate.section].append(candidate)
    groups = list(by_section.values())
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(group)
    return [c for round_ in zip_longest(*groups) for c in round_ if c is not None]


def _interleave(first: list[Candidate], second: list[Candidate], first_share: float) -> list[Candidate]:
    """Merge two ordered lists so that ``first_share`` of the items come from ``first``."""
    merged: list[Candidate] = []
    i = j = 0
    while i < len(first) and j < len(second):
        if i <= first_share * (i + j):
            merged.append(first[i])
            i += 1
        else:
            merged.append(second[j])
            j += 1
    return merged + first[i:] + second[j:]


def plan_crawl_order(
    candidates: Iterable[Candidate],
    *,
    level4_share: float = 0.6,
    seed: int = 42,
) -> dict[str, list[Candidate]]:
    """Build each topic's priority queue.

    Core articles (levels 1-3) come first, since they are the ones most likely to be asked
    about. After them, well-known level 4 articles alternate with narrower level 5 ones, which
    add the niche subjects and the dense neighbourhoods that make recommendations specific.
    """
    by_topic: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_topic[candidate.topic].append(candidate)

    queues: dict[str, list[Candidate]] = {}
    for topic in sorted(by_topic):
        rng = random.Random(f"{seed}:{topic}")
        tiers: dict[str, list[Candidate]] = {"core": [], "4": [], "5": []}
        for candidate in by_topic[topic]:
            level = candidate.vital_level
            tiers["core" if level <= CORE_MAX_LEVEL else str(min(level, 5))].append(candidate)
        core, level4, level5 = (_round_robin_by_section(tiers[name], rng) for name in ("core", "4", "5"))
        queues[topic] = core + _interleave(level4, level5, level4_share)
    return queues
