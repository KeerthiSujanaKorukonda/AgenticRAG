# Evaluation

## What this document is (and isn't)

This describes the evaluation **methodology and framework**
(`src/evaluation/`, `scripts/evaluate.py`) that this project implements.
It does **not** contain benchmark result numbers, because running the
comparison and ablation study requires:

1. A prepared retrieval index (`scripts/prepare_data.py`, needs internet
   access to `export.arxiv.org` and `huggingface.co`), and
2. The real embedding/generation/NLI models downloaded from Hugging Face.

Neither was available in the sandbox this project was developed in (no
network access to those hosts). Running `scripts/evaluate.py` in a real
deployment (or locally, with internet access) will produce
`docs/evaluation_results.json` with real, measured numbers — this file is
intentionally not fabricated or checked in ahead of time.

## Systems compared

| System | What it does | File |
|---|---|---|
| Baseline 1: Vector RAG | Question → semantic (FAISS) retrieval only → extractive answer | `run_baseline_vector_rag` |
| Baseline 2: Hybrid RAG | Question → BM25 + semantic hybrid retrieval → extractive answer | `run_baseline_hybrid_rag` |
| System: Agentic RAG | Full LangGraph pipeline (planning, multi-query, evidence evaluation, contradiction detection, adaptive retrieval, reasoning, verification) | `run_agentic_rag` |

All three run against the **same** prepared index and the **same**
benchmark questions, so any measured difference reflects the pipeline
architecture rather than data differences.

## Metrics that are actually computable (`src/evaluation/metrics.py`)

| Metric | How it's computed |
|---|---|
| `context_relevance` | Fraction of ALL retrieved evidence classified RELEVANT by the Evidence Agent |
| `evidence_coverage` | The real coverage score computed by the Evidence Agent |
| `citation_correctness` | Fraction of citations whose `doc_id` matches a document actually retrieved this run |
| `faithfulness` | Fraction of Verification Agent checks that came back "supported" |
| `hallucination_rate` | `1 - faithfulness` |
| `retrieval_iterations` | The real `iteration` counter from the final graph state |
| `latency_ms` | Real wall-clock time for the run |

## Metrics that are honestly NOT evaluated

| Metric | Why not |
|---|---|
| `retrieval_recall` | Would require labeled relevance judgments (qrels) against a fixed corpus. None exist for this project's ad hoc arXiv-topic corpora. |
| `answer_correctness` | Would require gold reference answers per benchmark question. None exist. |

Both functions (`retrieval_recall()`, `answer_correctness()` in
`src/evaluation/metrics.py`) always return `None`, and every place that
displays them renders `"Not evaluated"` rather than a placeholder number —
per the project requirement to never fabricate a metric.

## Ablation study

`src/evaluation/benchmark.py` defines six variants, each a genuinely
separate graph execution with one component disabled:

```
full
without_planner                    (no generation model -> deterministic fallback plan)
without_evidence_verification      (no NLI model -> weaker verification fallback)
without_adaptive_retrieval         (max_iterations forced to 0)
without_contradiction_detection    (no NLI model -> contradiction_agent returns [])
without_final_verification         (no NLI model -> verification degrades to citable-evidence-only check)
```

Note that `without_evidence_verification`, `without_contradiction_detection`,
and `without_final_verification` all currently key off the same underlying
NLI-model toggle, since all three depend on it — this is a real limitation
of how cleanly separable those components are, documented rather than
worked around with a fake separate toggle.

## Running it for real

```bash
python scripts/prepare_data.py --query "your topic" --max-results 30
python scripts/evaluate.py --index-dir data/index --output docs/evaluation_results.json
```

`scripts/evaluate.py` will error out clearly (not silently produce empty
numbers) if no prepared index exists yet.
