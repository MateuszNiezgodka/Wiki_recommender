from collections import Counter

import pytest

from wiki_recommender.preprocessing.tokens import (
    PLURAL_MARK,
    LemmaEvidence,
    LemmaRepairs,
    fold,
    is_valid_token,
    load_nlp,
    tokenize_texts,
)


@pytest.fixture(scope="module")
def nlp():
    return load_nlp()


def _lemmas(nlp, *texts):
    """Lemmas of the last text, repaired with evidence from all of them, as the pipeline does."""
    evidence = LemmaEvidence()
    tokenized = tokenize_texts(nlp, texts, evidence)
    repairs = evidence.repairs()
    return [repairs.resolve(token) for token in tokenized[-1].lemmas]


@pytest.mark.parametrize(
    ("word", "expected"),
    [("Müller", "muller"), ("São", "sao"), ("Łódź", "lodz"), ("O'Brien", "obrien"), ("U.S.", "us")],
)
def test_fold_merges_spelling_variants(word, expected):
    assert fold(word) == expected


@pytest.mark.parametrize("token", ["dog", "covid-19", "c++", "mp3", "h2o", "f-16", "wwii"])
def test_meaningful_tokens_are_valid(token):
    assert is_valid_token(token)


@pytest.mark.parametrize("token", ["1945", "19th", "1990s", "1914-1918", "x", "", "中国", "a" * 41])
def test_noise_tokens_are_invalid(token):
    assert not is_valid_token(token)


def test_lemmatization_uses_the_part_of_speech(nlp):
    assert _lemmas(nlp, "The mice were running.") == ["mouse", "run"]


def test_stopwords_punctuation_and_numbers_are_removed(nlp):
    assert _lemmas(nlp, "This is the 3rd of 12, (and) it was in 1945!") == []


def test_acronyms_are_not_mistaken_for_stopwords(nlp):
    lemmas = _lemmas(nlp, "the US and the WHO told us about it")
    assert "us" in lemmas and "who" in lemmas
    assert lemmas.count("us") == 1


def test_lemmas_and_stems_stay_aligned(nlp):
    result = tokenize_texts(nlp, ["Dogs chased the running mice across beautiful gardens."], LemmaEvidence())[
        0
    ]
    assert len(result.lemmas) == len(result.stems) > 0
    assert "garden" in result.lemmas and "beauti" in result.stems


def test_a_mostly_common_noun_is_repaired():
    evidence = LemmaEvidence(noun_lemmas=Counter({("dogs", "dog"): 90}), proper_nouns=Counter({"dogs": 5}))
    assert evidence.repairs().names == {"dogs": "dog"}


def test_a_mostly_proper_noun_is_left_alone():
    evidence = LemmaEvidence(
        noun_lemmas=Counter({("paris", "pari"): 2}), proper_nouns=Counter({"paris": 900})
    )
    assert len(evidence.repairs()) == 0


def test_a_word_only_ever_seen_as_a_noun_needs_no_repair():
    assert len(LemmaEvidence(noun_lemmas=Counter({("mice", "mouse"): 9})).repairs()) == 0


def test_evidence_from_workers_adds_up():
    total = LemmaEvidence()
    total.update(LemmaEvidence(noun_lemmas=Counter({("dogs", "dog"): 2}), proper_nouns=Counter({"dogs": 1})))
    total.update(LemmaEvidence(noun_lemmas=Counter({("dogs", "dog"): 3}), proper_nouns=Counter({"dogs": 1})))
    assert total.noun_lemmas["dogs", "dog"] == 5
    assert total.proper_nouns["dogs"] == 2


def test_a_plural_outside_the_dictionary_gets_its_real_singular():
    evidence = LemmaEvidence(
        surfaces=Counter({"cryptocurrency": 200, "cryptocurrencies": 100, "goddess": 50, "goddesses": 9}),
        guessed_plurals=Counter({"cryptocurrencies": 100, "goddesses": 9}),
    )
    assert evidence.repairs().plurals == {"cryptocurrencies": "cryptocurrency", "goddesses": "goddess"}


def test_a_real_word_ending_in_ie_keeps_its_form():
    evidence = LemmaEvidence(
        surfaces=Counter({"movie": 500, "movies": 200, "ties": 90, "tie": 80, "ty": 3}),
        guessed_plurals=Counter({"movies": 200, "ties": 90}),
    )
    assert evidence.repairs().plurals == {"movies": "movie", "ties": "tie"}


@pytest.mark.parametrize(
    "surfaces",
    [
        {"series": 22000, "serie": 129},  # "Serie A" is not the singular of "series"
        {"ramesses": 424},
        {"selfies": 4},
    ],
)
def test_a_word_without_a_common_singular_stays_whole(surfaces):
    surface = max(surfaces, key=surfaces.get)
    evidence = LemmaEvidence(surfaces=Counter(surfaces), guessed_plurals=Counter({surface: 3}))
    assert evidence.repairs().resolve(PLURAL_MARK + surface) == surface


def test_a_repair_touches_only_the_occurrences_that_need_it():
    repairs = LemmaRepairs(plurals={"indies": "indie"})
    assert repairs.apply("~indies west indies\nindies") == "indie west indies\nindies"


def test_a_noun_left_unlemmatized_joins_its_usual_lemma():
    """spaCy gives `datum` for most uses of `data`, but leaves it as is in "data interpretation"."""
    evidence = LemmaEvidence(
        noun_lemmas=Counter({("data", "datum"): 900}), proper_nouns=Counter({"data": 20})
    )
    assert evidence.repairs().apply("datum data interpretation") == "datum datum interpretation"


def test_a_name_is_not_touched_by_the_plural_of_a_lookalike(nlp):
    evidence = LemmaEvidence()
    tokenize_texts(nlp, ["Marie Curie met Mary in Paris."], evidence)
    assert "marie" not in evidence.repairs().names


def test_a_plural_tagged_as_a_name_can_take_both_repairs():
    evidence = LemmaEvidence(
        noun_lemmas=Counter({("cryptocurrencies", "cryptocurrencie"): 50}),
        proper_nouns=Counter({"cryptocurrencies": 5}),
        surfaces=Counter({"cryptocurrency": 200, "cryptocurrencies": 55}),
    )
    assert evidence.repairs().names == {"cryptocurrencies": "cryptocurrency"}


def test_a_suffix_rule_lemma_of_a_name_follows_the_corpus():
    """spaCy turns the noun `dies` into `dy`. The corpus says `die`."""
    evidence = LemmaEvidence(
        noun_lemmas=Counter({("dies", "dy"): 20}),
        proper_nouns=Counter({"dies": 5}),
        surfaces=Counter({"die": 2500, "dies": 760, "dy": 70}),
    )
    assert evidence.repairs().names == {"dies": "die"}


def test_a_correctly_lemmatized_series_survives_a_bad_occurrence(nlp):
    lemmas = _lemmas(
        nlp, "Serie A is an Italian league.", "Series of numbers.", "The television series was popular."
    )
    assert lemmas == ["television", "series", "popular"]


def test_no_marker_leaks_into_resolved_lemmas(nlp):
    lemmas = _lemmas(nlp, "Dogs chased cryptocurrencies past the West Indies and Ramesses.")
    assert not any(token.startswith(PLURAL_MARK) for token in lemmas)


def test_a_dictionary_plural_keeps_its_lemma_even_if_the_singular_is_rare(nlp):
    """`atrocity` is 12 times rarer than `atrocities` in the corpus, but spaCy knows the word."""
    assert _lemmas(nlp, "The atrocities were documented.") == ["atrocity", "document"]
