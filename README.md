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
