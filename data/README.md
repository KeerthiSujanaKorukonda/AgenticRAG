# Data

## `seed_papers.jsonl` — bundled fallback corpus

This file contains **20 real papers** identified via arXiv (real titles, real
arXiv IDs/URLs, real publication years), spanning three topic clusters:

- `low_resource_nlp` — low-resource / multilingual NLP surveys and benchmarks
- `rag` / `rag_evaluation` / `agentic_rag` — retrieval-augmented generation surveys
- `hallucination` — LLM hallucination detection/mitigation surveys

**Important — the `summary` field is NOT the paper's abstract.** It is a short,
original paraphrase written while reviewing the paper, kept intentionally brief
to avoid reproducing copyrighted abstract text. It exists only so the bundled
seed corpus is usable *offline*, as a small fallback/demo dataset, before you've
run real data acquisition.

For actual research use, run `scripts/prepare_data.py`, which fetches the
**real abstract text** for a topic directly from the public arXiv API
(`http://export.arxiv.org/api/query`, no key required) and builds the real
FAISS + BM25 indexes that the app queries. The bundled `seed_papers.jsonl` is
used only when no prepared index exists yet (e.g., a fresh clone before the
first data-prep run), purely so the app has *something* real — never
synthetic — to demonstrate against.

## Where the seed papers came from

Each entry's `url` points to the paper's real arXiv abstract page. Author
lists were not always fully visible in the sources reviewed while building
this seed set; where a full author list wasn't confirmed, the `authors` field
says `"Unknown (survey authors)"` rather than guessing — this app never
fabricates metadata it can't verify.

## Regenerating / expanding the corpus

```bash
python scripts/prepare_data.py --query "your topic" --max-results 40
```

This will:

1. Query the arXiv API for real papers matching the topic.
2. Pull each paper's real title, authors, year, abstract, and URL.
3. Clean and chunk the abstract/available text.
4. Embed chunks with `BAAI/bge-small-en-v1.5`.
5. Build a FAISS index and a BM25 index.
6. Save everything under `data/index/` for the app to load at runtime.

See `scripts/prepare_data.py` for the full implementation and
`DEPLOYMENT.md` for how this fits into the Streamlit Cloud deployment flow.
