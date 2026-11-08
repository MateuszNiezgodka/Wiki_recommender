# Wikipedia Recommender System

Content-based recommender system for English Wikipedia / Fandom wiki articles built for the **Information Retrieval (`[IR]`)** course at Poznan University of Technology.

Given a collection of previously visited articles (as **titles** or **URLs**), the system recommends the most relevant unseen articles from a local corpus of 1,000+ preprocessed wiki pages along with numerical similarity scores and visual score explanations.

## Project Roadmap

See [**`plan.md`**](./plan.md) for the complete project breakdown, including:
- Grading rules and submission checklist
- Staged tasks (**Obligatory** vs. **Not Obligatory / Optional**)
- Shared Parquet data schemas (`articles_raw.parquet` & `articles_processed.parquet`)
- Two-student division of work and GitHub workflow

## Repository Structure

```text
Wiki_recommender/
├── Project-Wiki.pdf          # Course assignment specification
├── plan.md                   # Detailed project stages, tasks & team plan
├── pyproject.toml            # Python >=3.14 project configuration
├── data/
│   ├── raw/                  # Stage 1 output: scraped wiki articles (.parquet / .csv)
│   └── processed/            # Stage 2 output: stemmed & lemmatized corpus (.parquet / .csv)
├── src/
│   └── wiki_recommender/
│       ├── crawler/          # Stage 1: Crawling & scraping (>= 1,000 articles)
│       ├── preprocessing/    # Stage 2: Cleaning, stemming & lemmatization
│       ├── recommender/      # Stage 3: Vectorization, similarity & multi-article scoring
│       └── visualization/    # Stage 5: Corpus statistics & graphical prediction breakdowns
├── notebooks/                # Interactive recommender interface, EDA & notebook report
├── reports/
│   └── figures/              # Exported charts and final PDF report
└── tests/                    # Unit tests
```

## Getting Started (For Collaborators)

### Prerequisites
- Python `>= 3.14`
- [`uv`](https://docs.astral.sh/uv/) (recommended) or standard `venv` + `pip`

### Setup

Using `uv`:
```bash
git clone git@github.com:MateuszNiezgodka/Wiki_recommender.git
cd Wiki_recommender
uv sync
```

Or using `venv` + `pip`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Stage 1: Building the Corpus

```bash
uv run python -m wiki_recommender.crawler                  # 8,800 articles, about 1 h 45 min
uv run python -m wiki_recommender.crawler --per-topic 100  # a smaller corpus (the page-view step alone takes ~25 min)
uv run pytest                                              # offline tests, no network needed
```

An interrupted crawl continues where it stopped when you rerun the command. `--no-resume` deletes the checkpoint and starts over. Run with `--help` for every option.

| Output in `data/raw/` | Contents |
| :--- | :--- |
| `articles_raw.parquet` | The corpus, consumed by Stage 2 |
| `articles_raw.stats.json` | Corpus composition, filter counts and the settings used |
| `articles_raw.checkpoint.jsonl` | Every crawl decision, used for resuming |
| `articles_raw.pageviews.json` | Cached view counts, so a resumed crawl keeps its priorities |

### How articles are chosen

Articles come from English Wikipedia's [Vital articles](https://en.wikipedia.org/wiki/Wikipedia:Vital_articles) lists, levels 4 and 5, which sort about 50,000 articles into 11 topics and thousands of subsections. Each topic gets the same quota:

1. **Core first:** articles of levels 1-3 (such as *World War II* or *Dog*) are always taken.
2. **Then a 60/40 mix** of level 4 articles and narrower level 5 ones, which add niche subjects and related neighbours.
3. **Round-robin over subsections:** every subsection contributes one article before any contributes a second, most-read first (60 days of page views). Popular subjects are favoured without being able to crowd out quieter ones.

Pages are rejected if they are disambiguation pages, redirects to an article already collected, link collections (`List of ...`, `Outline of ...`), outside the main namespace, or shorter than 100 words.

### How text is cleaned

Each page is fetched with the MediaWiki `action=parse` API and its HTML is reduced to running prose. Infoboxes, tables, navigation templates, figures, formulas, pronunciation guides and citation markers are removed, as are trailing sections such as *See also*, *References* and *External links*. Bibliography entries found elsewhere in an article are dropped too. Section headings are stored apart from `raw_text`.

The crawler identifies itself with a descriptive `User-Agent`, spaces requests at least 0.25 s apart, sends `maxlag=5` so it yields when Wikipedia's servers are busy, and pauses all requests with exponential backoff on errors or `Retry-After`.

## Stage 2: Preprocessing

```bash
uv run python -m wiki_recommender.preprocessing            # about 25 min with 4 workers, peaks near 5 GB of RAM
uv run python -m wiki_recommender.preprocessing.evaluate   # compares document representations, a few minutes
```

Lower `--workers` if memory is short. Run either command with `--help` for every option.

| Output in `data/processed/` | Contents |
| :--- | :--- |
| `articles_processed.parquet` | The processed corpus, consumed by Stages 3-5 (schema in [`plan.md`](./plan.md)) |
| `articles_processed.stats.json` | Token counts, vocabulary sizes (raw, lemmas, stems) and the most frequent lemmas and stems |
| `representation_eval.json` | Retrieval quality of each compared representation, written by `evaluate` |

### Cleaning

The crawler already removes most markup, so this stage removes what is left: URLs, stray HTML tags, `[[ ]]` and `{{ }}` brackets, unfilled template parameters, IPA and score placeholders, and repeated whitespace. Categories that describe Wikipedia's own maintenance (`Articles with short description`, `Living people`) are dropped, because they say nothing about the subject.

### Tokens, lemmas and stems

Every article is tokenized once with spaCy (`en_core_web_sm`, tagger and lemmatizer only). Each word that survives one shared filter yields both a **lemma** (spaCy, part-of-speech aware) and a **stem** (NLTK Snowball), so `tokens_lemmatized` and `tokens_stemmed` stay aligned word for word:

- case and accents are folded, so *Müller*, *Muller* and *MÜLLER* meet;
- stopwords, punctuation, numbers, ordinals (`19th`) and year ranges are removed, but all-caps acronyms such as *US* or *WHO* are kept;
- tokens must be Latin-script, 2 to 40 characters long and contain a letter, so `covid-19` and `c++` survive.

spaCy guesses lemmas for words it does not know by stripping a plural ending, which gives `cryptocurrencie` or turns *series* into `serie`. Those guesses are settled after the whole corpus is read: the singular is used only if the corpus itself contains it at least a tenth as often as the plural. *Cryptocurrencies* becomes `cryptocurrency`, while *series*, *species* and *Ramesses* stay whole. Likewise, a plural tagged as a name only because it opens a sentence (*Dogs are...*) joins its noun (`dog`), while real names such as *Paris* are left alone.

### Lemmas over stems

Lemmas are the canonical representation (`processed_text`). They are real words, which keeps the report's term charts and score explanations readable (`university`, not `univers`), and they do not merge unrelated words the way stems do (*universe* and *university* share the stem `univers`). They also retrieve slightly better, as measured below.

### Choosing the representation

`evaluate` compares document representations by how often an article's 10 nearest neighbours are related to it by signals that do not come from its text: the same Vital articles subsection, and a shared category from a held-out half that no representation sees. The score below is their mean on 4,407 test articles; the weights in `vectors.py` were chosen on the other half.

| Representation | Score | Same subsection | Shared held-out category |
| :--- | ---: | ---: | ---: |
| Definition only | 0.222 | 0.266 | 0.179 |
| Lead only | 0.259 | 0.313 | 0.205 |
| Full text | 0.288 | 0.342 | 0.234 |
| Full text, stemmed | 0.283 | 0.335 | 0.231 |
| Full text + 0.5 × definition | 0.296 | 0.358 | 0.234 |
| **Final** (+ 0.25 × categories) | **0.313** | **0.374** | **0.252** |
| Final, stemmed | 0.309 | 0.371 | 0.247 |

Paired bootstrap intervals (95 %) confirm both choices: the final representation beats full text by +0.025 [+0.022, +0.029], and lemmas beat stems by +0.004 [+0.003, +0.006]. Lemmas win by a small margin, but consistently, and with readable terms.

### Document parts

Each article also keeps its **definition** (the first full lead paragraph, which says what the subject is), the length of its lead, and its lemmatized categories. `processed_text` joins definition, article and `cat:`-prefixed category tokens for a plain `TfidfVectorizer`, while `DocumentVectorizer` in [`vectors.py`](./src/wiki_recommender/preprocessing/vectors.py) weights the parts separately: whole article + 0.5 × definition for the text, plus 0.25 × categories.
