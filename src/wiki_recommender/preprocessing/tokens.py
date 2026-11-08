"""Turn text into normalized lemma and stem tokens with one shared filter.

Lemmas come from spaCy's part-of-speech aware lemmatizer. Stems come from the Snowball stemmer, applied
to the same surviving words, so both lists have equal length and compare like with like.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache

import spacy
from nltk.stem.snowball import SnowballStemmer
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Token

MODEL = "en_core_web_sm"
_MAX_TOKEN_LENGTH = 40

_STEMMER = SnowballStemmer("english")
_NOT_DECOMPOSED = str.maketrans(
    {"ß": "ss", "ø": "o", "æ": "ae", "œ": "oe", "ł": "l", "đ": "d", "ð": "d", "þ": "th", "ı": "i"}
)
_REMOVED_MARKS = str.maketrans("", "", "'’‘`.·")
_VALID_TOKEN = re.compile(r"[a-z0-9+#-]+")
_HAS_LETTER = re.compile(r"[a-z]")
_NUMERIC_NOISE = re.compile(r"\d+(?:st|nd|rd|th|s)|\d+(?:-\d+)+")
_ES_PLURAL_ENDINGS = ("sses", "xes", "zes", "ches", "shes")
# A singular rarer than this share of its plural is taken as a coincidence (`serie` in "Serie A").
_MIN_SINGULAR_SHARE = 0.1
# Marks a lemma decided after the whole corpus is read. It cannot occur in a valid token.
PLURAL_MARK = "~"


def load_nlp():
    """Load the tagger and lemmatizer. The parser and NER add cost and nothing the tokens use."""
    return spacy.load(MODEL, disable=["parser", "ner"])


@lru_cache(maxsize=1_000_000)
def fold(word: str) -> str:
    """Lowercase and strip accents and inner punctuation, so spelling variants of a name meet."""
    folded = unicodedata.normalize("NFKD", word.casefold().translate(_NOT_DECOMPOSED))
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return folded.translate(_REMOVED_MARKS).strip("-")


@lru_cache(maxsize=1_000_000)
def _stem(word: str) -> str:
    return _STEMMER.stem(word)


def _is_acronym(surface: str) -> bool:
    letters = surface.replace(".", "")
    return len(letters) >= 2 and letters.isupper()


def is_valid_token(token: str) -> bool:
    """Keep Latin-script tokens with a letter, like `covid-19` or `c++`, and reject numeric noise."""
    return (
        2 <= len(token) <= _MAX_TOKEN_LENGTH
        and _VALID_TOKEN.fullmatch(token) is not None
        and _HAS_LETTER.search(token) is not None
        and _NUMERIC_NOISE.fullmatch(token) is None
    )


def normalize(token: Token) -> tuple[str, str] | None:
    """Return (folded surface, folded lemma), or None if the word carries no retrieval value.

    An all-caps word is an acronym rather than a stopword, so `US` and `WHO` survive casefolding.
    """
    if token.is_space or token.is_punct:
        return None
    surface = fold(token.text)
    lemma = fold(token.lemma_)
    if not is_valid_token(surface) or not is_valid_token(lemma):
        return None
    if not _is_acronym(token.text) and (surface in STOP_WORDS or lemma in STOP_WORDS):
        return None
    return surface, lemma


def _regular_singular(plural: str) -> str | None:
    """Return the singular an English plural ending implies, or None if the ending implies nothing."""
    if plural.endswith("ies"):
        return plural[:-3] + "y"
    if plural.endswith(_ES_PLURAL_ENDINGS):
        return plural[:-2]
    return None


def _is_stripped_s(surface: str, lemma: str) -> bool:
    """True if the lemma only drops the final `s` of an `-ies`/`-es` word, spaCy's guess for unknown words.

    Such a lemma can be right (`movies`), half right (`cryptocurrencie`) or wrong (`series` -> `serie`).
    """
    return lemma == surface[:-1] and _regular_singular(surface) is not None


def _is_suffix_guess(surface: str, lemma: str) -> bool:
    """True if the lemma is either singular a plural-looking ending suggests, as with `dies` -> `dy`."""
    singular = _regular_singular(surface)
    return singular is not None and lemma in (surface[:-1], singular)


@dataclass
class LemmaRepairs:
    """Final lemmas, decided once the whole corpus has been read.

    `plurals` resolves the marked surfaces `tokenize_texts` emits for stripped-`s` guesses; an unlisted one
    keeps its form. `names` maps a word spaCy left unlemmatized to its usual noun lemma, wherever it occurs.
    """

    plurals: dict[str, str] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.plurals) + len(self.names)

    def resolve(self, token: str) -> str:
        if token.startswith(PLURAL_MARK):
            return self.plurals.get(token[1:], token[1:])
        return self.names.get(token, token)

    def apply(self, text: str) -> str:
        """Resolve every token of space-joined, newline-separated text."""
        return "\n".join(" ".join(map(self.resolve, line.split())) for line in text.split("\n"))


@dataclass
class LemmaEvidence:
    """Corpus-wide counts used to repair two systematic lemmatizer errors after all text is read."""

    noun_lemmas: Counter = field(default_factory=Counter)
    proper_nouns: Counter = field(default_factory=Counter)
    surfaces: Counter = field(default_factory=Counter)
    guessed_plurals: Counter = field(default_factory=Counter)

    def update(self, other: LemmaEvidence) -> None:
        self.noun_lemmas.update(other.noun_lemmas)
        self.proper_nouns.update(other.proper_nouns)
        self.surfaces.update(other.surfaces)
        self.guessed_plurals.update(other.guessed_plurals)

    def _singular(self, surface: str) -> str:
        """Choose the singular the corpus itself uses, or keep the word if it has none.

        `cryptocurrencies` becomes `cryptocurrency`, `dies` becomes `die`. A word that only looks plural,
        such as `series` or `Ramesses`, has no singular nearly as common as itself, and stays whole.
        """
        candidates = (surface[:-1], _regular_singular(surface))
        best = max(candidates, key=lambda word: self.surfaces[word])
        if self.surfaces[best] > 0 and self.surfaces[best] >= _MIN_SINGULAR_SHARE * self.surfaces[surface]:
            return best
        return surface

    def repairs(self) -> LemmaRepairs:
        """Decide the final lemmas.

        A plural-looking name takes the noun lemma when the corpus uses the word mostly as a noun: `Dogs`
        opening a sentence is tagged PROPN and would otherwise stay apart from `dog`. The same mapping
        catches other occurrences spaCy leaves unlemmatized, such as `data` in "data interpretation".
        Paris stays intact, because the corpus almost never uses it as a common noun.
        """
        best: dict[str, tuple[str, int]] = {}
        for (surface, lemma), count in self.noun_lemmas.items():
            if count > best.get(surface, ("", 0))[1]:
                best[surface] = (lemma, count)

        repairs = LemmaRepairs()
        for surface in self.guessed_plurals:
            if (singular := self._singular(surface)) != surface:
                repairs.plurals[surface] = singular
        for surface, (lemma, count) in best.items():
            if surface == lemma or surface not in self.proper_nouns or count < self.proper_nouns[surface]:
                continue
            final = self._singular(surface) if _is_suffix_guess(surface, lemma) else lemma
            if final != surface:
                repairs.names[surface] = final
        return repairs


@dataclass
class TokenizedText:
    lemmas: list[str]
    stems: list[str]


def tokenize_texts(nlp, texts: Iterable[str], evidence: LemmaEvidence) -> list[TokenizedText]:
    """Tokenize each text, updating the evidence used to repair lemmas after the whole corpus is read.

    A stripped-`s` guess is emitted as a marked surface, which `LemmaRepairs` resolves. Its repair then
    touches only those occurrences: `series` correctly lemmatized stays `series` even where another
    occurrence came out as `serie`, and `marie` does not change just because `maries` also occurs.
    """
    results = []
    for doc in nlp.pipe(texts, batch_size=64):
        lemmas, stems = [], []
        for token in doc:
            normalized = normalize(token)
            if normalized is None:
                continue
            surface, lemma = normalized
            evidence.surfaces[surface] += 1
            if _is_stripped_s(surface, lemma):
                evidence.guessed_plurals[surface] += 1
                lemmas.append(PLURAL_MARK + surface)
            else:
                lemmas.append(lemma)
            stems.append(_stem(surface))
            if token.pos_ == "NOUN":
                evidence.noun_lemmas[surface, lemma] += 1
            elif token.pos_ == "PROPN":
                evidence.proper_nouns[surface] += 1
        results.append(TokenizedText(lemmas, stems))
    return results
