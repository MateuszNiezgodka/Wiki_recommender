import numpy as np
import pytest

from wiki_recommender.crawler.client import ApiRequestError, ApiResponseError
from wiki_recommender.preprocessing.vectors import ArticleParts, DocumentVectorizer
from wiki_recommender.recommender import model as model_module
from wiki_recommender.recommender.engine import Recommender
from wiki_recommender.recommender.evaluate import nearest, weight_grid
from wiki_recommender.recommender.fallback import FetchedArticle, FetchFailed, fetch_unseen
from wiki_recommender.recommender.links import LinkVectorizer, link_features
from wiki_recommender.recommender.model import (
    ArticleIndex,
    ArticleVectors,
    RecommenderModel,
    Settings,
)
from wiki_recommender.recommender.resolve import TitleResolver, parse_query
from wiki_recommender.recommender.scoring import group_interests, interest_ranking, power_mean, top_indices

# title, topic, text, outlinks, categories
CORPUS = [
    ("Dog", "Biology", "dog canine pet bark wolf loyal breed animal", ["Wolf", "Cat"], ["canine pet"]),
    ("Wolf", "Biology", "wolf canine pack predator dog wild animal", ["Dog", "Lion"], ["canine"]),
    ("Cat", "Biology", "cat feline pet whisker hunt animal breed", ["Dog", "Lion"], ["feline pet"]),
    ("Lion", "Biology", "lion feline predator savanna pack hunt animal", ["Cat", "Wolf"], ["feline"]),
    (
        "Quantum mechanics",
        "Physics",
        "quantum particle wave physics energy theory",
        ["Electron"],
        ["physics"],
    ),
    ("Electron", "Physics", "electron particle charge quantum atom energy", ["Photon"], ["physics particle"]),
    ("Photon", "Physics", "photon light particle quantum energy wave", ["Electron"], ["physics particle"]),
    ("Atom", "Physics", "atom electron nucleus particle charge theory", ["Electron"], ["physics"]),
    ("São Paulo", "Geography", "city brazil coast wave light energy", ["Lion"], ["city"]),
    ("Rio de Janeiro", "Geography", "city brazil coast beach carnival", ["São Paulo"], ["city"]),
]
TOPICS = ["Biology", "Physics", "Geography"]


def _topic_embedding(topic: str) -> np.ndarray:
    vector = np.full(len(TOPICS), 0.1, dtype=np.float32)
    vector[TOPICS.index(topic)] = 1.0
    return vector / np.linalg.norm(vector)


def _parts(rows) -> ArticleParts:
    return ArticleParts(
        tokens=[text.split() for _, _, text, _, _ in rows],
        definitions=[text.split()[:3] for _, _, text, _, _ in rows],
        categories=[[token for c in cats for token in c.split()] for *_, cats in rows],
    )


@pytest.fixture(scope="module")
def model() -> RecommenderModel:
    text_vectorizer = DocumentVectorizer()
    link_vectorizer = LinkVectorizer()
    vectors = ArticleVectors(
        text=text_vectorizer.fit_transform(_parts(CORPUS)),
        links=link_vectorizer.fit_transform(
            [link_features(title, links) for title, _, _, links, _ in CORPUS]
        ),
        embedding=np.stack([_topic_embedding(topic) for _, topic, *_ in CORPUS]),
    )
    titles = [title for title, *_ in CORPUS]
    index = ArticleIndex(
        article_ids=list(range(len(CORPUS))),
        titles=titles,
        urls=[f"https://en.wikipedia.org/wiki/{t.replace(' ', '_')}" for t in titles],
        topics=[topic for _, topic, *_ in CORPUS],
        sections=[topic for _, topic, *_ in CORPUS],
        summaries=[text for _, _, text, _, _ in CORPUS],
    )
    return RecommenderModel(index, vectors, text_vectorizer, link_vectorizer, Settings())


@pytest.fixture
def offline(model) -> Recommender:
    return Recommender(model, fetch_unseen=False)


# --- Scoring ---------------------------------------------------------------------------------------------


def test_power_mean_spans_mean_to_max():
    similarity = np.array([[0.9, 0.1], [0.1, 0.5]])
    assert power_mean(similarity, 1.0) == pytest.approx([0.5, 0.3])
    assert power_mean(similarity, np.inf) == pytest.approx([0.9, 0.5])
    p2 = power_mean(similarity, 2.0)
    assert np.all((p2 >= [0.5, 0.3]) & (p2 <= [0.9, 0.5]))


def test_top_indices_skips_excluded_and_orders_by_score():
    scores = np.array([0.2, 0.9, 0.5, 0.7])
    assert list(top_indices(scores, 2, excluded=[1])) == [3, 2]
    assert list(top_indices(scores, 10, excluded=[0, 1, 2, 3])) == []


def test_related_history_articles_form_one_interest():
    pairwise = np.array([[1.0, 0.6, 0.05], [0.6, 1.0, 0.1], [0.05, 0.1, 1.0]])
    assert group_interests(pairwise, threshold=0.3) == [[0, 1], [2]]


def test_interests_share_the_list_by_size():
    # Rows 0-1 are one interest preferring candidates 3-6; row 2 prefers candidates 7-9.
    similarity = np.zeros((3, 10))
    similarity[0:2, 3:7] = [0.9, 0.8, 0.7, 0.6]
    similarity[2, 7:10] = [0.5, 0.4, 0.3]
    pairwise = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 1.0]])
    chosen, scores, groups = interest_ranking(similarity, pairwise, 3, [0, 1, 2], threshold=0.3, p=1.0)
    assert groups == [[0, 1], [2]]
    assert set(chosen) == {3, 4, 7}  # two slots for the larger interest, one for the smaller
    assert np.all(np.diff(scores[chosen]) <= 0)


# --- Resolving input -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("dog", "Dog"),
        ("quantum_mechanics", "Quantum mechanics"),
        ("QUANTUM MECHANICS", "Quantum mechanics"),
        ("sao paulo", "São Paulo"),
        ("https://en.wikipedia.org/wiki/Quantum_mechanics", "Quantum mechanics"),
        ("en.m.wikipedia.org/wiki/Cat#Behaviour", "Cat"),
        ("https://en.wikipedia.org/wiki/S%C3%A3o_Paulo", "São Paulo"),
    ],
)
def test_titles_and_urls_find_their_article(text, expected):
    resolver = TitleResolver([title for title, *_ in CORPUS])
    query = parse_query(text)
    assert resolver.titles[resolver.find(query.title)] == expected


def test_a_foreign_url_is_not_read_as_a_title():
    assert parse_query("https://example.com/wiki/Dog").title is None


def test_typos_get_suggestions():
    resolver = TitleResolver([title for title, *_ in CORPUS])
    assert resolver.find("Elektron") is None
    assert resolver.suggest("Elektron")[0] == "Electron"


def test_a_word_suggests_the_titles_it_begins_first():
    resolver = TitleResolver(["Perjury", "Mercury (planet)", "Mercury (element)", "Mercurial"])
    assert resolver.suggest("mercury")[:2] == ["Mercury (planet)", "Mercury (element)"]


def test_link_features_hold_the_own_title_once():
    assert link_features("dog", ["Wolf", "wolf", "Cat#Behaviour", ""]) == ["Cat", "Dog", "Wolf"]


# --- Recommending ----------------------------------------------------------------------------------------


def test_the_closest_articles_come_first_and_visited_ones_never_appear(offline):
    result = offline.recommend(["Dog"], top_k=5)
    titles = [r.title for r in result.recommendations]
    assert "Dog" not in titles
    assert set(titles[:2]) <= {"Wolf", "Cat"}
    scores = [r.score for r in result.recommendations]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_a_history_mixes_titles_and_urls(offline):
    result = offline.recommend(["electron", "https://en.wikipedia.org/wiki/Photon"], top_k=3)
    assert [v.title for v in result.visited] == ["Electron", "Photon"]
    assert {r.title for r in result.recommendations} >= {"Atom", "Quantum mechanics"}
    assert set(result.recommendations[0].by_visited) == {"Electron", "Photon"}


def test_unknown_inputs_are_reported_with_suggestions(offline):
    result = offline.recommend(["Dgo", "https://example.com/x", "Cat"], top_k=3)
    reasons = {item.query: item for item in result.unresolved}
    assert "Dog" in reasons["Dgo"].suggestions
    assert "not an English Wikipedia" in reasons["https://example.com/x"].reason
    assert [v.title for v in result.visited] == ["Cat"]
    assert "Skipped 'Dgo'" in str(result)


def test_nothing_resolvable_gives_an_empty_list(offline):
    result = offline.recommend(["Xyzzy"])
    assert result.recommendations == []
    assert "No recommendations" in str(result)


def test_a_repeated_article_counts_once(offline):
    result = offline.recommend(["Dog", "dog", "https://en.wikipedia.org/wiki/Dog"])
    assert len(result.visited) == 1


def test_two_interests_both_get_recommendations(offline):
    result = offline.recommend(["Dog", "Wolf", "Electron"], top_k=4, aggregation="interests")
    topics = {r.topic for r in result.recommendations}
    assert {"Biology", "Physics"} <= topics
    assert len(result.interests) == 2


def test_explanations_name_shared_terms(offline):
    best = offline.recommend(["Electron"], top_k=1).recommendations[0]
    assert best.shared_terms
    assert set(best.by_signal) == {"text", "links", "embedding"}


@pytest.mark.parametrize("history", [["Electron"], ["Electron", "Photon"], ["Dog", "Wolf", "Electron"]])
def test_signal_contributions_add_up_to_the_score(offline, history):
    for item in offline.recommend(history, top_k=4).recommendations:
        assert sum(item.signal_contributions.values()) == pytest.approx(item.score, abs=1e-3)


def test_term_contributions_add_up_to_the_text_part(offline):
    # The toy articles share fewer terms than are listed, so the listed terms are all of them.
    for item in offline.recommend(["Electron", "Photon"], top_k=3).recommendations:
        total = sum(value for _, value in item.term_contributions)
        assert total == pytest.approx(item.signal_contributions["text"], abs=1e-3)
        assert [term for term, _ in item.term_contributions][: len(item.shared_terms)] == item.shared_terms


def test_each_recommendation_names_the_interest_behind_its_score(offline):
    result = offline.recommend(["Dog", "Wolf", "Electron"], top_k=4, aggregation="interests")
    for item in result.recommendations:
        assert item.interest in (["Dog", "Wolf"], ["Electron"])
        expected = np.mean([item.by_visited[title] for title in item.interest])
        assert item.score == pytest.approx(expected, abs=1e-3)


def test_the_model_survives_saving(model, tmp_path):
    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = Recommender(RecommenderModel.load(path), fetch_unseen=False)
    assert [r.title for r in loaded.recommend(["Dog"]).recommendations] == [
        r.title for r in Recommender(model, fetch_unseen=False).recommend(["Dog"]).recommendations
    ]


# --- Pages outside the corpus ----------------------------------------------------------------------------

COYOTE_HTML = (
    "<p>The coyote is a canine animal, a wild dog related to the wolf. "
    + "It hunts in a pack. " * 20
    + "</p>"
)


class FakeWiki:
    def __init__(self):
        self.requests = []

    def get(self, **params):
        page = params["page"]
        self.requests.append(page)
        if page == "Offline":
            raise ApiRequestError("network down")
        if page == "Nowhere":
            raise ApiResponseError("missingtitle")
        title = {"Doggo": "Dog"}.get(page, page)
        return {
            "parse": {
                "title": title,
                "pageid": 99,
                "revid": 1,
                "text": COYOTE_HTML,
                "properties": {"disambiguation": ""} if page == "Mercury" else {},
                "categories": [{"category": "Canis"}],
            }
        }

    def __exit__(self, *_):
        pass


@pytest.fixture
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(
        model_module, "embed_articles", lambda passages, name: _topic_embedding("Biology")[None]
    )


def test_a_page_outside_the_corpus_is_fetched_and_vectorized(model, fake_embeddings, tmp_path):
    resolver = TitleResolver(model.index.titles)
    fetched = fetch_unseen("Coyote", model, resolver.find, FakeWiki(), tmp_path / "processed.parquet")
    assert isinstance(fetched, FetchedArticle)
    assert fetched.title == "Coyote"
    assert len(fetched.vectors) == 1 and fetched.vectors.text.nnz > 0


def test_a_redirect_into_the_corpus_uses_the_corpus_article(model, tmp_path):
    resolver = TitleResolver(model.index.titles)
    assert fetch_unseen("Doggo", model, resolver.find, FakeWiki(), tmp_path / "p.parquet") == 0


@pytest.mark.parametrize(
    ("page", "reason"),
    [
        ("Mercury", "disambiguation"),
        ("Nowhere", "no English Wikipedia article"),
        ("Offline", "could not be reached"),
    ],
)
def test_unusable_pages_explain_why(model, tmp_path, page, reason):
    resolver = TitleResolver(model.index.titles)
    with pytest.raises(FetchFailed, match=reason):
        fetch_unseen(page, model, resolver.find, FakeWiki(), tmp_path / "p.parquet")


def test_a_fetched_page_drives_recommendations(model, fake_embeddings, monkeypatch):
    monkeypatch.setattr("wiki_recommender.recommender.fallback.query_client", FakeWiki)
    result = Recommender(model, fetch_unseen=True).recommend(["Coyote"], top_k=3)
    assert result.visited[0].title == "Coyote" and result.visited[0].index is None
    assert {r.title for r in result.recommendations} & {"Dog", "Wolf"}


# --- Evaluation helpers ----------------------------------------------------------------------------------


def test_the_weight_grid_covers_the_simplex():
    grid = weight_grid(["a", "b", "c"], step=0.5)
    assert len(grid) == 6
    assert all(sum(w.values()) == pytest.approx(1.0) for w in grid)


def test_nearest_never_returns_the_article_itself():
    similarity = np.array([[1.0, 0.2, 0.9], [0.2, 1.0, 0.3], [0.9, 0.3, 1.0]])
    rows = np.arange(3)
    assert nearest(similarity, rows, 3, k=1).ravel().tolist() == [2, 2, 0]
