from wiki_recommender.crawler.seeds import (
    Candidate,
    merge_candidates,
    parse_vital_list,
    plan_crawl_order,
)

WIKITEXT = """
'''Vital articles''' intro with a [[Wikipedia:Vital articles|link]].
=Technology=
==Agriculture ==
{{Vital article count|2}}
# {{Icon|B}} '''''[[Farming]]''''' ([[Wikipedia:Vital articles/Level 3|Level 3]])
## {{Icon|C}} [[Plough|The plough]]<!-- [[Commented out]] -->
===Tools===
# [[Hoe (tool)#Types|Hoe]]
# [[List of tools]]
# [[File:Hoe.jpg]]
==Computing==
# [[2001: A Space Odyssey]]
[[Not a list item]]
"""


def test_parse_vital_list_reads_items_levels_and_sections():
    parsed = list(parse_vital_list(WIKITEXT, topic="Technology", list_level=4, subpage="Sub"))
    assert parsed == [
        Candidate("Farming", "Technology", "Sub > Agriculture", 3),
        Candidate("Plough", "Technology", "Sub > Agriculture", 4),
        Candidate("Hoe (tool)", "Technology", "Sub > Agriculture > Tools", 4),
        Candidate("2001: A Space Odyssey", "Technology", "Sub > Computing", 4),
    ]


def test_merge_keeps_the_highest_ranked_listing():
    merged = merge_candidates(
        [
            Candidate("Dog", "Zoology", "A", 5),
            Candidate("Dog", "Pets", "B", 4),
            Candidate("Cat", "Pets", "B", 5),
        ]
    )
    assert sorted(merged, key=lambda c: c.title) == [
        Candidate("Cat", "Pets", "B", 5),
        Candidate("Dog", "Pets", "B", 4),
    ]


def _pool():
    return [
        Candidate(f"{section}{level}{i}", "Topic", section, level)
        for section, count in (("Big", 40), ("Small", 2), ("Tiny", 1))
        for level in (4, 5)
        for i in range(count)
    ] + [Candidate(f"Core{i}", "Topic", "Big", 2) for i in range(3)]


def test_core_articles_come_first():
    queue = plan_crawl_order(_pool())["Topic"]
    assert {c.title for c in queue[:3]} == {"Core0", "Core1", "Core2"}
    assert len(queue) == len(_pool())


def test_small_sections_are_not_crowded_out():
    queue = plan_crawl_order(_pool())["Topic"][3:]
    assert {c.section for c in queue[:12]} == {"Big", "Small", "Tiny"}


def test_level_share_is_respected_while_both_tiers_last():
    queue = plan_crawl_order(_pool(), level4_share=0.6)["Topic"][3:]
    first_fifty = [c.vital_level for c in queue[:50]]
    assert first_fifty.count(4) == 30


def test_order_is_reproducible_and_depends_on_the_seed():
    pool = _pool()
    assert plan_crawl_order(pool, seed=1) == plan_crawl_order(list(reversed(pool)), seed=1)
    assert plan_crawl_order(pool, seed=1) != plan_crawl_order(pool, seed=2)
