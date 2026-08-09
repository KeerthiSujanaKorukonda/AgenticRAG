# 🔬 ResearchGapPilot

**Agentic RAG for Research Discovery, Evidence Verification & Research Gap Detection**

A real agentic RAG system — not a PDF chatbot — built on a genuine
[LangGraph](https://github.com/langchain-ai/langgraph) state machine with
conditional edges, bounded adaptive-retrieval and verification loops, hybrid
(semantic + BM25) retrieval, NLI-based contradiction detection, and
evidence-validated research gap discovery. Hugging Face models only — no
paid LLM APIs anywhere in this project.

## Problem Statement

Standard RAG (`question → vector search → LLM answer`) treats every
question as a single retrieval hop and trusts whatever the model generates
from whatever it retrieved. That's a poor fit for research discovery:
research questions have multiple dimensions, evidence disagrees across
papers, and "there isn't a clear answer yet" is often the *correct*
answer. ResearchGapPilot instead plans a research strategy, retrieves
iteratively until evidence coverage is genuinely sufficient (or a bounded
retry budget is exhausted), checks whether its own sources actually agree
with each other, and verifies its own conclusions against cited evidence
before showing them — refusing to answer confidently when it can't.

## Why Traditional RAG Is Insufficient Here

| | Traditional RAG | ResearchGapPilot |
|---|---|---|
| Retrieval | One semantic search pass | Multi-query hybrid (semantic + BM25) retrieval per sub-question, with adaptive re-retrieval |
| Question handling | Every input hits the same pipeline | An intent router filters greetings/small talk before any retrieval happens |
| Evidence | Trusted as-is | Classified (RELEVANT/WEAKLY_RELEVANT/IRRELEVANT/CONTRADICTORY), coverage explicitly scored |
| Disagreement | Silently merged | Detected via NLI, shown with a grounded guess at *why* sources disagree |
| Gaps | Not a concept | Explicitly detected, evidence-validated, and rejected if unsupported |
| Hallucination guard | None built in | Verification Agent NLI-checks every claim against its own citations |
| Memory | None / whole new query each time | Session-scoped follow-up handling reuses prior research state |

## Agentic Architecture

```
USER INPUT -> Intent Router -> [conversational reply]  or  [full research workflow]
```

See `docs/architecture.md` for the full LangGraph diagram, node-by-node
responsibilities, and the loop-termination guarantees.

### Intent Router

The mandatory first component (`src/router.py`). Deterministic rules
(word-boundary phrase matching, not naive substrings — see
`docs/methodology.md` for a bug this caught) handle greetings,
capabilities questions, and off-topic requests without ever loading a
model. A lightweight zero-shot classifier is available as an optional
fallback for genuinely ambiguous input, and its confidence is never
blindly trusted — low-confidence results fall back to asking for
clarification rather than guessing.

### Follow-up Memory

Session-scoped via `st.session_state.research_state`. A follow-up either
reuses existing evidence (if it's actually relevant to the follow-up text)
or triggers targeted additional retrieval — decided by real keyword
overlap against existing evidence, not a coin flip. Asking about a gap
that doesn't exist yet (no prior research state) returns an honest
"please start a research question first," never an invented answer.

### Planner, Retrieval, Evidence Evaluation, Contradiction Detection,
### Adaptive Retrieval, Gap Detection, Verification

Each has its own section in `docs/methodology.md` with the actual scoring
formulas used — nothing is a black box.

## Dataset

Real papers only, sourced from the public [arXiv API](https://arxiv.org/help/api)
(`export.arxiv.org`, no key required):

- `scripts/prepare_data.py` fetches real papers for any topic you give it,
  with real titles, authors, years, abstracts, and URLs.
- A small bundled fallback corpus (`data/seed_papers.jsonl`, 20 real papers
  across low-resource NLP, RAG, and hallucination-detection topics) lets
  the app demo something real before you've run data prep yourself. See
  `data/README.md` for exactly what's in it and how it was gathered.

## Models

All Hugging Face, all CPU-runnable, no paid APIs:

| Purpose | Model |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` |
| NLI (contradiction + verification) | `cross-encoder/nli-deberta-v3-xsmall` |
| Generation (planning + final synthesis) | `google/flan-t5-small` |
| Optional reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

Centralized in `src/config.py` — swap any of these by changing one line.

## Evaluation

Real baseline-vs-agentic comparison (Vector RAG / Hybrid RAG / Agentic RAG)
and a 6-variant ablation study, both implemented in `src/evaluation/`.
**No result numbers are checked into this repo** — see `docs/evaluation.md`
for exactly why (no network access to arXiv/Hugging Face in the
development sandbox) and how to produce real numbers with
`scripts/evaluate.py` once deployed.

## Limitations

- No labeled ground truth exists for retrieval recall or answer
  correctness — both are honestly reported as "Not evaluated," never
  estimated.
- Confidence scores on research gaps are an explicit, documented heuristic
  (see `docs/methodology.md`), not a calibrated probability.
- CPU-only inference (no GPU assumed); expect a few seconds per retrieval
  iteration and generation call, not sub-second responses.
- The bundled seed corpus is small (20 papers) and meant as a fallback demo
  — real research use requires running `scripts/prepare_data.py`.
- Five real bugs were found and fixed during development via actual test
  execution; see `docs/methodology.md` for the full list — documented
  rather than hidden, since the project spec explicitly prohibits claiming
  untested success.

## Deployment

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for the full Streamlit Cloud
deployment walkthrough, including exactly what was and wasn't verified in
the development environment.

## Future Work

- Real benchmark numbers once run against real data/models (this repo ships
  the framework, not fabricated results).
- A proper labeled evaluation set (qrels + gold answers) to make retrieval
  recall and answer correctness computable rather than "Not evaluated."
- GPU-accelerated deployment option for lower latency.
- Persistent (cross-session) research history, with user consent, rather
  than the current single-session memory.
- Splitting the NLI-model-gated ablation variants (contradiction detection,
  evidence verification, final verification) into independently toggleable
  components rather than sharing one underlying model switch.

## Project Structure

See the top-level layout in this repo, or `docs/architecture.md` for how
the pieces fit together.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/prepare_data.py --query "your research topic" --max-results 30
streamlit run app.py
```

Without running `prepare_data.py` first, the app still runs — it falls
back to the small bundled real-paper seed corpus automatically.
