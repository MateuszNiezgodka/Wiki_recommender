"""Clean, tokenize and save the corpus: raw articles in, `articles_processed.parquet` out."""

import json
import logging
import os
import re
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from spacy.lang.en.stop_words import STOP_WORDS

from wiki_recommender.preprocessing.clean import (
    CATEGORY_TOKEN_PREFIX,
    clean_section_label,
    clean_wiki_text,
    first_paragraph,
    informative_categories,
    split_lead_and_body,
)
from wiki_recommender.preprocessing.tokens import LemmaEvidence, LemmaRepairs, load_nlp, tokenize_texts

log = logging.getLogger(__name__)

PROCESSED_SCHEMA = pa.schema(
    [
        ("article_id", pa.int64()),
        ("title", pa.string()),
        ("url", pa.string()),
        ("summary", pa.string()),
        ("word_count", pa.int32()),
        ("token_count", pa.int32()),
        ("lead_token_count", pa.int32()),  # tokens_lemmatized[:lead_token_count] is the lead
        ("tokens_lemmatized", pa.list_(pa.string())),
        ("tokens_stemmed", pa.list_(pa.string())),
        ("tokens_definition", pa.list_(pa.string())),
        ("tokens_definition_stemmed", pa.list_(pa.string())),
        ("categories_lemmatized", pa.list_(pa.string())),  # one space-joined entry per category
        ("processed_text", pa.string()),
        ("categories", pa.list_(pa.string())),
        ("outlinks", pa.list_(pa.string())),
        ("sections", pa.list_(pa.string())),
        ("topic", pa.string()),
        ("seed_section", pa.string()),
        ("vital_level", pa.int8()),
    ]
)

_CHUNK_SIZE = 40
_WRITE_BATCH = 200
_TOP_TOKENS = 50
_FIELDS = ("definition", "lead", "body", "categories")
_RAW_WORD = re.compile(r"[^\W\d_]+")
# spaCy's vocabulary only grows while a process runs, so each worker is replaced after this many chunks.
_CHUNKS_PER_WORKER = 25
_MAX_WORKERS = 4

_nlp = None


@dataclass
class PreprocessConfig:
    input: Path = Path("data/raw/articles_raw.parquet")
    output: Path = Path("data/processed/articles_processed.parquet")
    workers: int | None = None

    @property
    def stats_path(self) -> Path:
        return self.output.with_suffix(".stats.json")

    @property
    def repairs_path(self) -> Path:
        return repairs_path(self.output)


def repairs_path(output: Path) -> Path:
    """Where the corpus-wide lemma repairs are kept, so a page processed later is lemmatized alike."""
    return output.with_suffix(".repairs.json")


def save_repairs(repairs: LemmaRepairs, path: Path) -> None:
    payload = {"plurals": repairs.plurals, "names": repairs.names}
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def load_repairs(path: Path) -> LemmaRepairs:
    """Load saved repairs. Without them, marked tokens simply keep their surface form."""
    if not path.exists():
        log.warning("No lemma repairs at %s; new pages will be lemmatized without them", path)
        return LemmaRepairs()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LemmaRepairs(plurals=payload["plurals"], names=payload["names"])


def process_article(nlp, raw: dict[str, Any], repairs: LemmaRepairs) -> dict[str, Any]:
    """Preprocess one article outside the corpus run, such as a page fetched at query time."""
    (article,), _ = tokenize_rows(nlp, [(0, raw)])
    article.lemmas = {name: repairs.apply(text) for name, text in article.lemmas.items()}
    return build_record(raw, article)


@dataclass
class TokenizedArticle:
    """Tokens per field, as space-joined strings with one line per paragraph or category.

    Strings keep memory low: tens of millions of separate token objects would need many gigabytes.
    """

    index: int
    lemmas: dict[str, str]
    stems: dict[str, str]


def _init_worker() -> None:
    global _nlp
    _nlp = load_nlp()


def article_fields(row: dict[str, Any]) -> dict[str, list[str]]:
    """Clean one raw article and split it into the parts that are tokenized separately."""
    lead_raw, body_raw = split_lead_and_body(row["raw_text"], row["summary"])
    lead = clean_wiki_text(lead_raw)
    return {
        "definition": [first_paragraph(lead)],
        "lead": lead.splitlines(),
        "body": clean_wiki_text(body_raw).splitlines(),
        "categories": informative_categories(list(row["categories"])),
    }


def tokenize_chunk(rows: list[tuple[int, dict[str, Any]]]) -> tuple[list[TokenizedArticle], LemmaEvidence]:
    """Tokenize a chunk of articles in a worker process."""
    return tokenize_rows(_nlp, rows)


def tokenize_rows(
    nlp, rows: list[tuple[int, dict[str, Any]]]
) -> tuple[list[TokenizedArticle], LemmaEvidence]:
    """Tokenize articles with the given spaCy pipeline. Lemmas still need `LemmaRepairs.apply`."""
    evidence = LemmaEvidence()
    fields_per_article = [article_fields(row) for _, row in rows]
    texts = [text for fields in fields_per_article for name in _FIELDS for text in fields[name]]
    tokenized = iter(tokenize_texts(nlp, texts, evidence))

    articles = []
    for (index, _), fields in zip(rows, fields_per_article, strict=True):
        lemmas, stems = {}, {}
        for name in _FIELDS:
            parts = [next(tokenized) for _ in fields[name]]
            lemmas[name] = "\n".join(" ".join(part.lemmas) for part in parts if part.lemmas)
            stems[name] = "\n".join(" ".join(part.stems) for part in parts if part.stems)
        articles.append(TokenizedArticle(index, lemmas, stems))
    return articles, evidence


def tokenize_corpus(
    rows: list[dict[str, Any]], workers: int | None = None
) -> tuple[list[TokenizedArticle], LemmaRepairs]:
    """Tokenize every article and repair lemmas using corpus-wide evidence. Results keep input order."""
    workers = workers or max(1, min(_MAX_WORKERS, (os.cpu_count() or 2) - 2))
    indexed = list(enumerate(rows))
    chunks = [indexed[start : start + _CHUNK_SIZE] for start in range(0, len(indexed), _CHUNK_SIZE)]

    articles: list[TokenizedArticle] = []
    evidence = LemmaEvidence()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, max_tasks_per_child=_CHUNKS_PER_WORKER
    ) as pool:
        futures = [pool.submit(tokenize_chunk, chunk) for chunk in chunks]
        for done, future in enumerate(as_completed(futures), start=1):
            chunk_articles, chunk_evidence = future.result()
            articles.extend(chunk_articles)
            evidence.update(chunk_evidence)
            if done % 25 == 0 or done == len(futures):
                log.info("Tokenized %d/%d articles", len(articles), len(rows))

    repairs = evidence.repairs()
    for article in articles:
        article.lemmas = {name: repairs.apply(text) for name, text in article.lemmas.items()}
    articles.sort(key=lambda article: article.index)
    return articles, repairs


def build_record(raw: dict[str, Any], article: TokenizedArticle) -> dict[str, Any]:
    """Combine a raw article with its tokens into one row of the processed corpus."""
    lead = article.lemmas["lead"].split()
    tokens = lead + article.lemmas["body"].split()
    definition = article.lemmas["definition"].split()
    categories = [line for line in article.lemmas["categories"].split("\n") if line]
    category_tokens = [CATEGORY_TOKEN_PREFIX + token for line in categories for token in line.split()]
    return {
        "article_id": raw["article_id"],
        "title": raw["title"],
        "url": raw["url"],
        "summary": raw["summary"],
        "word_count": raw["word_count"],
        "token_count": len(tokens),
        "lead_token_count": len(lead),
        "tokens_lemmatized": tokens,
        "tokens_stemmed": article.stems["lead"].split() + article.stems["body"].split(),
        "tokens_definition": definition,
        "tokens_definition_stemmed": article.stems["definition"].split(),
        "categories_lemmatized": categories,
        "processed_text": " ".join(definition + tokens + category_tokens),
        "categories": informative_categories(list(raw["categories"])),
        "outlinks": raw["outlinks"],
        "sections": raw["sections"],
        "topic": raw["topic"],
        "seed_section": clean_section_label(raw["seed_section"]),
        "vital_level": raw["vital_level"],
    }


def _quantiles(values: list[int]) -> dict[str, int]:
    values = sorted(values)
    deciles = statistics.quantiles(values, n=10) if len(values) > 1 else values
    return {
        "min": values[0],
        "p10": round(deciles[0]),
        "median": round(statistics.median(values)),
        "p90": round(deciles[-1]),
        "max": values[-1],
    }


def summarize(
    raw_vocabulary: set[str],
    lemma_counts: Counter,
    stem_counts: Counter,
    token_counts: list[int],
    definition_counts: list[int],
    repairs: LemmaRepairs,
) -> dict[str, Any]:
    """Describe what preprocessing did. These numbers feed the report's vocabulary statistics."""
    return {
        "articles": len(token_counts),
        "tokens": {"total": sum(token_counts), **_quantiles(token_counts)},
        "definition_tokens": _quantiles(definition_counts),
        "vocabulary": {
            "raw_casefolded": len(raw_vocabulary),
            "raw_without_stopwords": len(raw_vocabulary - STOP_WORDS),
            "lemmas": len(lemma_counts),
            "stems": len(stem_counts),
        },
        "lemma_repairs": len(repairs),
        "top_lemmas": lemma_counts.most_common(_TOP_TOKENS),
        "top_stems": stem_counts.most_common(_TOP_TOKENS),
    }


def process_corpus(config: PreprocessConfig) -> dict[str, Any]:
    """Run the whole stage. The output appears atomically, so a failure leaves no partial file."""
    table = pq.read_table(config.input)
    rows = table.to_pylist()
    log.info("Read %d articles from %s", len(rows), config.input)
    articles, repairs = tokenize_corpus(rows, config.workers)

    config.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.output.with_suffix(".tmp")
    raw_vocabulary: set[str] = set()
    lemma_counts: Counter = Counter()
    stem_counts: Counter = Counter()
    token_counts: list[int] = []
    definition_counts: list[int] = []
    with pq.ParquetWriter(temporary, PROCESSED_SCHEMA, compression="zstd") as writer:
        batch = []
        for raw, article in zip(rows, articles, strict=True):
            record = build_record(raw, article)
            raw_vocabulary.update(word.casefold() for word in _RAW_WORD.findall(raw["raw_text"]))
            lemma_counts.update(record["tokens_lemmatized"])
            stem_counts.update(record["tokens_stemmed"])
            token_counts.append(record["token_count"])
            definition_counts.append(len(record["tokens_definition"]))
            batch.append(record)
            if len(batch) == _WRITE_BATCH:
                writer.write_table(pa.Table.from_pylist(batch, schema=PROCESSED_SCHEMA))
                batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=PROCESSED_SCHEMA))
    temporary.replace(config.output)

    save_repairs(repairs, config.repairs_path)
    stats = summarize(raw_vocabulary, lemma_counts, stem_counts, token_counts, definition_counts, repairs)
    config.stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Saved %d articles to %s", len(rows), config.output)
    return stats
