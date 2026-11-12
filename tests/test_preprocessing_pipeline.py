import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp

from wiki_recommender.crawler.pipeline import ARTICLE_SCHEMA
from wiki_recommender.preprocessing.evaluate import (
    Labels,
    bootstrap_difference,
    build_labels,
    nearest_neighbours,
    score_queries,
    summarize_scores,
)
from wiki_recommender.preprocessing.pipeline import (
    PROCESSED_SCHEMA,
    PreprocessConfig,
    TokenizedArticle,
    article_fields,
    build_record,
    load_repairs,
    process_article,
    process_corpus,
    save_repairs,
)
from wiki_recommender.preprocessing.tokens import LemmaRepairs, load_nlp
from wiki_recommender.preprocessing.vectors import ArticleParts, DocumentVectorizer, load_parts


def _raw_article(article_id, title, lead, body, categories):
    text = f"{lead}\n{body}" if body else lead
    return {
        "article_id": article_id,
        "title": title,
        "url": f"https://en.wikipedia.org/wiki/{title}",
        "raw_text": text,
        "categories": categories,
        "outlinks": ["Wolf"],
        "summary": lead,
        "sections": ["Biology"],
        "topic": "Biology",
        "seed_section": '<span id="A"></span>Animals',
        "vital_level": 4,
        "word_count": len(text.split()),
        "revision_id": 1,
        "retrieved_at": "2026-11-07T00:00:00+00:00",
    }


RAW = [
    _raw_article(
        1,
        "Dog",
        "The dog is a domesticated descendant of wolves that hunters selectively bred over many years.\n"
        "Dogs live with people.",
        "Dogs eat meat.\nMusical scores are temporarily disabled due to a security issue. See Help:Score.",
        ["Domesticated animals", "Main topic articles"],
    ),
    _raw_article(
        2,
        "Wolf",
        "The wolf hunts in large family packs across northern forests and open tundra regions.",
        "",
        [],
    ),
]


def test_fields_separate_the_definition_from_the_lead_and_clean_the_body():
    fields = article_fields(RAW[0])
    assert fields["definition"] == [RAW[0]["summary"].splitlines()[0]]
    assert len(fields["lead"]) == 2
    assert fields["body"] == ["Dogs eat meat."]
    assert fields["categories"] == ["Domesticated animals"]


def test_record_joins_lead_and_body_and_records_the_boundary():
    article = TokenizedArticle(
        index=0,
        lemmas={
            "definition": "dog wolf",
            "lead": "dog wolf\nhuman",
            "body": "meat",
            "categories": "animal pet\ncanine",
        },
        stems={
            "definition": "dog wolf",
            "lead": "dog wolf\nhuman",
            "body": "meat",
            "categories": "anim pet\ncanin",
        },
    )
    record = build_record(RAW[0], article)
    assert record["tokens_lemmatized"] == ["dog", "wolf", "human", "meat"]
    assert record["tokens_lemmatized"][: record["lead_token_count"]] == ["dog", "wolf", "human"]
    assert record["categories_lemmatized"] == ["animal pet", "canine"]
    assert record["processed_text"] == "dog wolf dog wolf human meat cat:animal cat:pet cat:canine"
    assert record["seed_section"] == "Animals"
    assert record["categories"] == ["Domesticated animals"]


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    folder = tmp_path_factory.mktemp("corpus")
    source = folder / "raw.parquet"
    pq.write_table(pa.Table.from_pylist(RAW, schema=ARTICLE_SCHEMA), source)
    config = PreprocessConfig(input=source, output=folder / "processed.parquet", workers=1)
    return config, process_corpus(config)


def test_whole_stage_writes_the_contract_columns(processed):
    config, _ = processed
    table = pq.read_table(config.output)
    assert table.schema.equals(PROCESSED_SCHEMA)
    assert table["title"].to_pylist() == ["Dog", "Wolf"]
    tokens = table["tokens_lemmatized"].to_pylist()
    assert "dog" in tokens[0]
    assert "musical" not in tokens[0]
    assert [len(t) for t in tokens] == [len(s) for s in table["tokens_stemmed"].to_pylist()]
    assert table["token_count"].to_pylist() == [len(t) for t in tokens]


def test_whole_stage_leaves_no_temporary_file_and_writes_stats(processed):
    config, stats = processed
    assert not config.output.with_suffix(".tmp").exists()
    assert json.loads(config.stats_path.read_text(encoding="utf-8"))["vocabulary"] == stats["vocabulary"]
    assert stats["articles"] == 2
    assert stats["vocabulary"]["raw_casefolded"] > stats["vocabulary"]["lemmas"] > 0


def test_a_single_article_is_processed_like_the_corpus(processed):
    config, _ = processed
    repairs = load_repairs(config.repairs_path)
    corpus_row = pq.read_table(config.output).to_pylist()[0]
    alone = process_article(load_nlp(), RAW[0], repairs)
    assert alone["tokens_lemmatized"] == corpus_row["tokens_lemmatized"]
    assert alone["processed_text"] == corpus_row["processed_text"]


def test_repairs_round_trip_and_tolerate_a_missing_file(tmp_path):
    path = tmp_path / "x.repairs.json"
    save_repairs(LemmaRepairs(plurals={"cryptocurrencies": "cryptocurrency"}, names={"dogs": "dog"}), path)
    loaded = load_repairs(path)
    assert loaded.apply("~cryptocurrencies dogs") == "cryptocurrency dog"
    assert len(load_repairs(tmp_path / "missing.json")) == 0


def test_stored_parts_turn_into_vectors(processed):
    config, _ = processed
    parts = load_parts(config.output)
    assert parts.categories[0] == ["domesticate", "animal"] and parts.categories[1] == []
    assert len(load_parts(config.output, stemmed=True).tokens) == 2


PARTS = ArticleParts(
    tokens=[["dog", "wolf", "pack", "meat"], ["dog", "wolf", "tundra"], ["piano", "music"], ["piano", "dog"]],
    definitions=[["dog", "wolf"], ["wolf"], ["piano"], ["piano"]],
    categories=[["animal"], ["animal"], ["music"], []],
)


def test_vectors_are_unit_length_and_group_similar_articles():
    vectors = DocumentVectorizer().fit_transform(PARTS)
    assert np.allclose(sp.linalg.norm(vectors, axis=1), 1.0)
    similarities = (vectors @ vectors.T).toarray()
    assert similarities[0, 1] > similarities[0, 2]


def test_definition_weight_pulls_a_vector_towards_the_definition():
    plain = DocumentVectorizer(definition_weight=0.0, category_weight=0.0)
    boosted = DocumentVectorizer(definition_weight=1.0, category_weight=0.0)
    plain_vectors, boosted_vectors = plain.fit_transform(PARTS), boosted.fit_transform(PARTS)
    wolf = plain.text.vocabulary_["wolf"]
    assert boosted_vectors[1, wolf] > plain_vectors[1, wolf]


def test_categories_use_their_own_features_with_named_columns():
    vectorizer = DocumentVectorizer()
    vectors = vectorizer.fit_transform(PARTS)
    names = list(vectorizer.feature_names)
    assert len(names) == vectors.shape[1]
    assert "cat:animal" in names and "animal" not in names
    assert vectors[0, names.index("cat:animal")] > 0


def test_an_article_outside_the_corpus_can_be_vectorized():
    vectorizer = DocumentVectorizer().fit(PARTS)
    vector = vectorizer.transform(ArticleParts([["wolf", "unknownword"]], [["wolf"]], [["animal"]]))
    assert vector.shape[0] == 1 and vector.nnz > 0


def _section_labels(sections):
    size = {s: sections.count(s) for s in sections}
    n = len(sections)
    empty = sp.csr_matrix((n, 1))
    return Labels(np.array(sections), np.array([size[s] for s in sections]), empty, sp.csr_matrix((n, n)))


def test_neighbours_exclude_the_article_itself_and_are_ranked():
    vectors = sp.csr_matrix(np.array([[1.0, 0.0], [0.9, 0.1], [0.5, 0.5], [0.0, 1.0]]))
    assert nearest_neighbours(vectors, k=2)[0].tolist() == [1, 2]


def test_section_score_is_normalized_by_what_the_section_can_give():
    labels = _section_labels([0, 0, 0, 0, 0, 1])
    neighbours = np.array([[1, 2], [0, 2], [0, 1], [0, 1], [0, 5], [0, 1]])
    scores = score_queries(labels, neighbours)["section"]
    assert scores[0] == 1.0
    assert scores[4] == 0.5
    assert np.isnan(scores[5])


def test_labels_hide_categories_and_detect_only_mutual_links():
    labels = build_labels(
        ["A", "B", "C"],
        ["t", "t", "t"],
        ["s", "s", "z"],
        [[], [], []],
        [["B", "C"], ["A"], []],
    )
    assert labels.mutual_link[0, 1] == 1 and labels.mutual_link[1, 0] == 1
    assert labels.mutual_link[0, 2] == 0
    assert labels.section[0] == labels.section[1] != labels.section[2]


def test_primary_score_averages_section_and_category():
    scores = {
        "section": np.array([1.0, np.nan]),
        "category": np.array([0.5, 0.5]),
        "mutual_link": np.array([0.0, 0.0]),
    }
    assert summarize_scores(scores, np.array([True, True]))["primary"] == 0.75


def test_bootstrap_interval_is_above_zero_for_a_clear_winner():
    n = 200
    good = {"section": np.full(n, 0.8), "category": np.full(n, 0.8)}
    bad = {"section": np.full(n, 0.2), "category": np.full(n, 0.2)}
    result = bootstrap_difference(good, bad, np.ones(n, dtype=bool))
    assert result["low"] > 0.5
    assert result["difference"] == pytest.approx(0.6)
