# Wikipedia Recommender System

Content-based recommender system for English Wikipedia / Fandom wiki articles built for the **Information Retrieval (`[IR]`)** course at Poznan University of Technology.

Given a collection of previously visited articles (as **titles** or **URLs**), the system recommends the most relevant unseen articles from a local corpus of 2,200 preprocessed Wikipedia pages, each with a score in [0, 1] and a graphical explanation of that score.

**Report:** [`reports/final_report.pdf`](./reports/final_report.pdf), also as an executed notebook, [`notebooks/final_report.ipynb`](./notebooks/final_report.ipynb).

## Quick start

```bash
uv sync                                          # Python >= 3.14 and uv, see Setup below
uv run python -m wiki_recommender.download       # the built corpus and model (~100 MB) into data/
uv run python -m wiki_recommender.recommender "Dog" "https://en.wikipedia.org/wiki/Cat" -k 10
```

To give your own list in a notebook, open [`notebooks/02_recommender_demo.ipynb`](./notebooks/02_recommender_demo.ipynb), edit `VISITED` in the first cell and run all cells (`uv run jupyter lab`). It prints the recommendations and draws their score breakdowns.

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
│       ├── visualization/    # Stage 5: Corpus statistics & graphical prediction breakdowns
│       └── download.py       # Fetch (or pack) the built data, so nothing has to be rebuilt
├── notebooks/
│   ├── 02_recommender_demo.ipynb  # Stage 4: give a list of articles, get explained recommendations
│   └── final_report.ipynb         # Stage 5: the report, executed
├── reports/
│   ├── final_report.pdf      # The report as PDF (and .html)
│   └── figures/              # Every chart of the report as PNG
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

### Data

`data/` is not in Git. `python -m wiki_recommender.download` fetches what Stages 3-5 read (processed corpus, fitted model, cached embeddings, statistics and measurements) from the repository's `data-v1` release and unpacks it. `--pack` builds that archive from a local `data/`, to attach it to a release. Without the archive, rebuild everything with the commands of Stages 1-3, about 25 minutes in total.

## Stage 1: Building the Corpus

```bash
uv run python -m wiki_recommender.crawler                  # 2,200 articles, about 18 min
uv run python -m wiki_recommender.crawler --per-topic 100  # a smaller corpus
uv run pytest                                              # offline tests, no network needed
```

An interrupted crawl continues where it stopped when you rerun the command. `--no-resume` deletes the checkpoint and starts over. Run with `--help` for every option.

| Output in `data/raw/` | Contents |
| :--- | :--- |
| `articles_raw.parquet` | The corpus, consumed by Stage 2 |
| `articles_raw.stats.json` | Corpus composition, filter counts and the settings used |
| `articles_raw.checkpoint.jsonl` | Every crawl decision, used for resuming |

### How articles are chosen

Articles come from English Wikipedia's [Vital articles](https://en.wikipedia.org/wiki/Wikipedia:Vital_articles) lists, levels 4 and 5, which sort about 50,000 articles into 11 topics and thousands of subsections. Each topic gets the same quota:

1. **Core first:** articles of levels 1-3 (such as *World War II* or *Dog*) are always taken.
2. **Then a 60/40 mix** of level 4 articles and narrower level 5 ones, which add niche subjects and related neighbours.
3. **Round-robin over subsections:** every subsection contributes one article before any contributes a second, shuffled with a fixed seed so no subsection or alphabetical prefix can crowd out the others.

Pages are rejected if they are disambiguation pages, redirects to an article already collected, link collections (`List of ...`, `Outline of ...`), outside the main namespace, or shorter than 100 words.

### How text is cleaned

Each page is fetched with the MediaWiki `action=parse` API and its HTML is reduced to running prose. Infoboxes, tables, navigation templates, figures, formulas, pronunciation guides and citation markers are removed, as are trailing sections such as *See also*, *References* and *External links*. Bibliography entries found elsewhere in an article are dropped too. Section headings are stored apart from `raw_text`.

The crawler identifies itself with a descriptive `User-Agent`, spaces requests at least 0.25 s apart, sends `maxlag=5` so it yields when Wikipedia's servers are busy, and pauses all requests with exponential backoff on errors or `Retry-After`.

## Stage 2: Preprocessing

```bash
uv run python -m wiki_recommender.preprocessing            # about 4 min with 4 workers, peaks near 5 GB of RAM
uv run python -m wiki_recommender.preprocessing.evaluate   # compares document representations, ~15 s
```

Lower `--workers` if memory is short. Run either command with `--help` for every option.

| Output in `data/processed/` | Contents |
| :--- | :--- |
| `articles_processed.parquet` | The processed corpus, consumed by Stages 3-5 (schema in [`plan.md`](./plan.md)) |
| `articles_processed.stats.json` | Token counts, vocabulary sizes (raw, lemmas, stems) and the most frequent lemmas and stems |
| `articles_processed.repairs.json` | Corpus-wide lemma repairs, so a page fetched later (Stage 3) is lemmatized the same way |
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

`evaluate` compares document representations by how often an article's 10 nearest neighbours are related to it by signals that do not come from its text: the same Vital articles subsection, and a shared category from a held-out half that no representation sees. The score below is their mean on 1,093 test articles; the weights in `vectors.py` were chosen on the other half.

| Representation | Score | Same subsection | Shared held-out category |
| :--- | ---: | ---: | ---: |
| Definition only | 0.226 | 0.306 | 0.147 |
| Lead only | 0.264 | 0.367 | 0.161 |
| Full text | 0.294 | 0.407 | 0.180 |
| Full text, stemmed | 0.287 | 0.399 | 0.176 |
| Full text + 0.5 × definition | 0.309 | 0.437 | 0.181 |
| **Final** (+ 0.25 × categories) | **0.323** | **0.450** | **0.197** |
| Final, stemmed | 0.317 | 0.440 | 0.194 |

Paired bootstrap intervals (95 %) confirm both choices: the final representation beats full text by +0.030 [+0.019, +0.041], and lemmas beat stems by +0.006 [+0.001, +0.011]. Lemmas win by a small margin, but consistently, and with readable terms.

### Document parts

Each article also keeps its **definition** (the first full lead paragraph, which says what the subject is), the length of its lead, and its lemmatized categories. `processed_text` joins definition, article and `cat:`-prefixed category tokens for a plain `TfidfVectorizer`, while `DocumentVectorizer` in [`vectors.py`](./src/wiki_recommender/preprocessing/vectors.py) weights the parts separately: whole article + 0.5 × definition for the text, plus 0.25 × categories.

## Stage 3: Recommendations

```bash
uv run python -m wiki_recommender.recommender "Dog" "https://en.wikipedia.org/wiki/Cat" -k 10
uv run python -m wiki_recommender.recommender --build                 # (re)build data/processed/recommender.pkl
uv run python -m wiki_recommender.recommender.evaluate                # measurements behind the settings, ~30 s
```

```python
from wiki_recommender.recommender import recommend

result = recommend(["Dog", "https://en.wikipedia.org/wiki/Cat"], top_k=10)
print(result)                                   # table of title, score and shared terms
result.recommendations[0].url                   # each item also has by_visited, by_signal and shared_terms
```

The first query builds the model if `data/processed/recommender.pkl` is missing: about 10 s with cached embeddings, or roughly 1.5 min on CPU to embed the corpus first. A query then takes a few seconds. `--offline` never contacts Wikipedia, and `--json` prints the full result.

**Input.** Titles and URLs can be mixed. Titles match regardless of case, accents, underscores and punctuation (`sao paulo` finds *São Paulo*). URLs may be desktop or mobile, percent-encoded, or carry a `#section`. An article outside the corpus is fetched from Wikipedia, processed with the Stage 1 extraction and Stage 2 preprocessing, and used like any visited article. A redirect such as *USA* resolves to the corpus article *United States*. Inputs that cannot be used are reported with the reason and close titles (`Elektron` → *Electron*; `Mercury` → *Mercury (planet)*, *Mercury (element)*).

**Output.** The top-k corpus articles, never including a visited one, each with its title, URL and a score in [0, 1]. Each also carries its similarity to every visited article, its mean similarity per signal, and the terms that contribute most to its text similarity.

### Scoring

Two signals compare two articles, each a cosine similarity in [0, 1]:

| Signal | Representation | Weight |
| :--- | :--- | ---: |
| Text | Stage 2 TF-IDF: lemmas + 0.5 × definition, + 0.25 × categories | 0.60 |
| Embedding | `BAAI/bge-small-en-v1.5` sentence embedding of the title and the first 150 words | 0.40 |

$$s(v, c) = 0.60\,\cos_{\text{text}}(v, c) + 0.40\,\max\big(0, \cos_{\text{emb}}(v, c)\big)$$

A history $V$ is first split into **interests**: visited articles are clustered with average linkage, and two stay together while their mean similarity is at least 0.3. For a candidate $c \notin V$ and interest $I$:

$$\text{score}_I(c) = \frac{1}{|I|} \sum_{v \in I} s(v, c), \qquad \text{score}(c) = \max_I\, \text{score}_I(c)$$

The list's slots are shared between interests in proportion to their size, each interest contributing its best-scoring candidates, and the final list is sorted by score. With a single interest this is the plain mean, which ranks exactly like the centroid of the visited articles.

### Choosing the settings

`recommender.evaluate` compares only the embedding variants already cached in `data/processed/embeddings/` (a fresh `--build` caches bge-small) and skips the others with a warning. It uses the relevance signals of Stage 2: the same Vital articles subsection, and a shared category from a held-out half that no signal sees. Weights and aggregation were chosen on half of the articles; the numbers below come from the other half (1,093 articles).

**Single articles** (share of the 10 nearest neighbours that are related):

| Signals | Score | Same subsection | Shared held-out category |
| :--- | ---: | ---: | ---: |
| bge-small embedding only | 0.302 | 0.423 | 0.180 |
| Text only (Stage 2) | 0.323 | 0.450 | 0.197 |
| **Text + bge-small** | **0.338** | **0.471** | **0.205** |

Text and embeddings complement each other: together they beat text alone by +0.011 [+0.005, +0.016] and the embedding alone by +0.029 [+0.022, +0.036] (95 % paired bootstrap). Embedding up to 6 passages per article instead of 1 did not help once combined with text. Larger models (`all-mpnet-base-v2`, `bge-base-en-v1.5`) embed about 3 times slower than bge-small on CPU and were not compared.

**Histories of visited articles** (two interests: 3 + 2 articles from subsections of different topics; *coverage* is the share of interests that get at least one recommendation):

| Aggregation | 1 interest, 3 articles | 1 interest, 5 articles | 2 interests | Coverage, 2 interests |
| :--- | ---: | ---: | ---: | ---: |
| Mean (centroid) | 0.349 | 0.334 | 0.459 | 0.79 |
| Power mean, p = 4 | 0.343 | 0.336 | 0.510 | 0.86 |
| Maximum | 0.323 | 0.319 | 0.518 | 0.87 |
| **Interests (threshold 0.3)** | **0.348** | **0.337** | **0.546** | **0.98** |

Splitting a history into interests costs almost nothing when it has one topic (−0.001) and gains +0.088 [+0.078, +0.098] over the mean when it has two, because the mean favours articles that sit between both topics and match neither well.

| Output in `data/processed/` | Contents |
| :--- | :--- |
| `recommender.pkl` | The fitted model: vectors of every signal, vectorizers and settings |
| `embeddings/*.npz` | Cached passage embeddings per model, reused by `--build` and `evaluate` |
| `recommender_eval.json` | Every measurement above, including all weight and aggregation candidates |

### Explanations

Each recommendation also carries the parts its score is made of. With the default mean (p = 1) they add up exactly:

$$\text{score}(c) = \sum_{\text{terms } w} 0.60\, u_w\, c_w + 0.40\,\overline{\cos}_{\text{emb}} = \frac{1}{|I|}\sum_{v \in I} s(v, c)$$

where $u$ is the mean text vector of the interest $I$. `signal_contributions` holds the two signal parts, `term_contributions` the largest shared terms of the text part, and `interest` the visited articles the score is averaged over.

## Stage 4: Giving a List of Articles

Three ways, all taking titles and URLs, mixed:

| Way | How |
| :--- | :--- |
| Notebook | [`notebooks/02_recommender_demo.ipynb`](./notebooks/02_recommender_demo.ipynb): edit `VISITED` in the first cell, run all cells |
| Command line | `uv run python -m wiki_recommender.recommender "Jazz" "Blues" -k 10` (`--json`, `--offline`, `--help`) |
| Python | `recommend(["Jazz", "Blues"], top_k=10)` from `wiki_recommender.recommender` |

## Stage 5: Report and Visualizations

```python
from wiki_recommender.recommender import recommend
from wiki_recommender.visualization import explain, explain_result, plot_visited_heatmap

result = recommend(["Dog", "Cat", "Quantum mechanics", "Electron"])
explain_result(result)        # every score split by signal (left) and by visited article (right)
explain(result, rank=1)       # one score as a waterfall of shared terms and signals
plot_visited_heatmap(result)  # similarity of every visited article to every recommendation
```

[`visualization/corpus.py`](./src/wiki_recommender/visualization/corpus.py) draws the database statistics (composition, lengths, vocabulary, frequent and characteristic terms, similarity distributions, topic heatmap, t-SNE map) and [`visualization/evaluation.py`](./src/wiki_recommender/visualization/evaluation.py) the measurements above.

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/final_report.ipynb   # ~2 min, refreshes reports/figures/
uv run jupyter nbconvert --to html --output-dir reports notebooks/final_report.ipynb
```

`reports/final_report.pdf` is that HTML printed to PDF in a browser (here with headless Edge: `msedge --headless --print-to-pdf=reports/final_report.pdf reports/final_report.html`), since `nbconvert --to pdf` needs a LaTeX installation.
