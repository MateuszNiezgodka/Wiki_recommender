import pytest

from wiki_recommender.crawler.titles import (
    canonicalize_url,
    is_article_title,
    is_prose_article_title,
    normalize_title,
    title_to_url,
    url_to_title,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("information_retrieval", "Information retrieval"),
        ("  Dog#Etymology ", "Dog"),
        (":Category:Dogs", "Category:Dogs"),
        ("iPhone", "IPhone"),
    ],
)
def test_normalize_title(raw, expected):
    assert normalize_title(raw) == expected


@pytest.mark.parametrize("title", ["Dog", "2001: A Space Odyssey", "Star Trek: Voyager", "C++"])
def test_article_titles_are_accepted(title):
    assert is_article_title(title)


@pytest.mark.parametrize(
    "title",
    [
        "Special:Random",
        "Category:Dogs",
        "File:Dog.jpg",
        "Talk:Dog",
        "wikt:dog",
        "Wikipedia:Vital articles",
        "",
    ],
)
def test_non_article_titles_are_rejected(title):
    assert not is_article_title(title)


@pytest.mark.parametrize(
    "title",
    ["List of dogs", "Lists of animals", "Outline of biology", "Mercury (disambiguation)", "Talk:Dog"],
)
def test_non_prose_titles_are_rejected(title):
    assert not is_prose_article_title(title)


def test_a_title_that_merely_starts_like_a_list_is_kept():
    assert is_prose_article_title("Listeria")


@pytest.mark.parametrize(
    "url",
    [
        "https://en.wikipedia.org/wiki/Schr%C3%B6dinger%27s_cat",
        "http://en.wikipedia.org/wiki/Schrödinger's_cat#Origin",
        "https://en.m.wikipedia.org/wiki/Schr%C3%B6dinger%27s_cat?action=view",
        "en.wikipedia.org/wiki/Schrödinger's cat",
        "https://en.wikipedia.org/w/index.php?title=Schr%C3%B6dinger%27s_cat&oldid=1",
        "/wiki/Schr%C3%B6dinger%27s_cat",
    ],
)
def test_url_variants_share_one_canonical_form(url):
    assert url_to_title(url) == "Schrödinger's cat"
    assert canonicalize_url(url) == "https://en.wikipedia.org/wiki/Schr%C3%B6dinger%27s_cat"


@pytest.mark.parametrize(
    "url", ["https://pl.wikipedia.org/wiki/Pies", "https://example.com/wiki/Dog", "https://en.wikipedia.org/"]
)
def test_foreign_urls_are_rejected(url):
    assert url_to_title(url) is None
    assert canonicalize_url(url) is None


def test_title_to_url_keeps_readable_punctuation():
    assert title_to_url("AC/DC") == "https://en.wikipedia.org/wiki/AC/DC"
    assert title_to_url("Who? (film)") == "https://en.wikipedia.org/wiki/Who%3F_(film)"
