import json

import pyarrow.parquet as pq
import pytest

from wiki_recommender.crawler.client import ApiRequestError, ApiResponseError
from wiki_recommender.crawler.pipeline import ARTICLE_SCHEMA, CrawlAborted, CrawlConfig, crawl

LONG_BODY = "<p>" + "word " * 150 + "</p>"
ARTS_LIST = "Wikipedia:Vital articles/Level 4/Arts"
HISTORY_LIST = "Wikipedia:Vital articles/Level 4/History"
DEFAULT_LISTS = {
    ARTS_LIST: "==Music==\n# [[Song]]\n# [[Stub]]\n# [[Dab]]\n# [[Gone]]\n# [[Tune]]\n# [[Melody]]",
    HISTORY_LIST: "==Eras==\n# [[Rome]]\n# [[Greece]]\n# [[Egypt]]",
}


class FakeWiki:
    """Stands in for the API. Tune redirects to Song; every page has a stable id."""

    def __init__(self, fail=(), lists=None, views=None):
        self.views = views or {}
        self.pageview_requests = 0
        self.fail = set(fail)
        self.lists = DEFAULT_LISTS if lists is None else lists
        self.article_requests = []

    def get(self, **params):
        if params.get("prop") == "pageviews":
            self.pageview_requests += 1
            titles = params["titles"].split("|")
            pages = [{"title": t, "pageviews": {"d1": self.views.get(t, 0)}} for t in titles]
            return {"query": {"pages": pages}}
        if params["action"] == "query":
            prefix = params["apprefix"]
            pages = [{"title": title} for title in self.lists if f":{prefix}" in title]
            return {"query": {"allpages": [*pages, {"title": f"Wikipedia:{prefix}Article alerts"}]}}
        page = params["page"]
        if page in self.lists:
            return {"parse": {"wikitext": self.lists[page]}}
        self.article_requests.append(page)
        if page in self.fail:
            raise ApiRequestError("network down")
        if page == "Gone":
            raise ApiResponseError("missingtitle")
        title = "Song" if page == "Tune" else page
        return {
            "parse": {
                "title": title,
                "pageid": sum(map(ord, title)) * 100 + len(title),
                "revid": 7,
                "text": "<p>too short</p>" if page == "Stub" else LONG_BODY,
                "properties": {"disambiguation": ""} if page == "Dab" else {},
                "categories": [
                    {"category": "Visible_category"},
                    {"category": "All_stub_articles", "hidden": True},
                ],
            }
        }


@pytest.fixture
def config(tmp_path):
    return CrawlConfig(output=tmp_path / "articles_raw.parquet", per_topic=3, workers=2)


def test_collects_the_quota_of_each_topic_and_filters_the_rest(config):
    records = crawl(FakeWiki(), config)
    by_topic = {}
    for record in records:
        by_topic.setdefault(record["topic"], set()).add(record["title"])
    assert by_topic["History"] == {"Rome", "Greece", "Egypt"}
    assert by_topic["Arts"] == {"Song", "Melody"}  # Every other Arts candidate is filtered out.
    assert len({record["article_id"] for record in records}) == len(records)


def test_stops_at_the_quota(tmp_path):
    lists = {HISTORY_LIST: "\n".join(f"# [[Page {i}]]" for i in range(30))}
    wiki = FakeWiki(lists=lists)
    records = crawl(wiki, CrawlConfig(output=tmp_path / "out.parquet", per_topic=5, workers=2))
    assert len(records) == 5
    assert len(wiki.article_requests) == 5


def test_writes_the_shared_parquet_contract(config):
    crawl(FakeWiki(), config)
    table = pq.read_table(config.output)
    assert table.column_names == ARTICLE_SCHEMA.names
    row = next(r for r in table.to_pylist() if r["title"] == "Rome")
    assert row["url"] == "https://en.wikipedia.org/wiki/Rome"
    assert row["categories"] == ["Visible category"]
    assert row["raw_text"].startswith("word word")
    assert row["word_count"] == 150
    assert row["vital_level"] == 4


def test_stats_account_for_every_rejection(config):
    crawl(FakeWiki(), config)
    stats = json.loads(config.stats_path.read_text(encoding="utf-8"))
    assert stats["articles"] == 5
    assert stats["per_topic"] == {"Arts": 2, "History": 3}
    for reason in ("too_short", "disambiguation", "api_missingtitle", "duplicate"):
        assert stats["statuses"][reason] == 1


def test_resume_does_not_refetch_finished_articles(config):
    crawl(FakeWiki(), config)
    second_run = FakeWiki()
    records = crawl(second_run, config)
    assert second_run.article_requests == []
    assert len(records) == 5


def test_resume_tolerates_a_truncated_checkpoint(config):
    crawl(FakeWiki(), config)
    with config.checkpoint.open("a", encoding="utf-8") as checkpoint:
        checkpoint.write('{"candidate": "Half wri')
    assert len(crawl(FakeWiki(), config)) == 5


def test_failed_requests_are_retried_on_the_next_run(config):
    assert "Rome" not in {r["title"] for r in crawl(FakeWiki(fail=["Rome"]), config)}
    assert "Rome" in {r["title"] for r in crawl(FakeWiki(), config)}


def test_most_read_articles_win_when_the_quota_is_tight(tmp_path):
    lists = {HISTORY_LIST: "\n".join(f"# [[Page {i}]]" for i in range(10))}
    wiki = FakeWiki(lists=lists, views={"Page 7": 900, "Page 2": 500})
    records = crawl(wiki, CrawlConfig(output=tmp_path / "out.parquet", per_topic=2, workers=1))
    assert {record["title"] for record in records} == {"Page 7", "Page 2"}


def test_resume_reuses_cached_pageviews(config):
    crawl(FakeWiki(), config)
    second_run = FakeWiki()
    crawl(second_run, config)
    assert second_run.pageview_requests == 0


def test_no_resume_starts_over(config):
    crawl(FakeWiki(), config)
    fresh = FakeWiki()
    crawl(fresh, CrawlConfig(output=config.output, per_topic=3, workers=2, resume=False))
    assert fresh.article_requests


def test_aborts_when_everything_is_failing(tmp_path):
    titles = [f"Page {i}" for i in range(40)]
    lists = {HISTORY_LIST: "\n".join(f"# [[{title}]]" for title in titles)}
    wiki = FakeWiki(fail=titles, lists=lists)
    with pytest.raises(CrawlAborted):
        crawl(wiki, CrawlConfig(output=tmp_path / "out.parquet", per_topic=40, workers=2))
