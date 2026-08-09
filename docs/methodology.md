# Methodology

## Retrieval

Hybrid retrieval combines:

- **Semantic search**: FAISS `IndexFlatIP` over `BAAI/bge-small-en-v1.5`
  embeddings (normalized, so inner product = cosine similarity). Flat/exact
  rather than approximate, since corpus sizes here (tens to low hundreds of
  chunks) don't need an approximate index.
- **Lexical search**: BM25 (`rank_bm25.BM25Okapi`) over simple regex
  tokenization.

Each query's semantic and BM25 scores are **independently min-max
normalized across that query's own candidate set**, then combined with
configurable weights (`RetrievalConfig.semantic_weight` /
`bm25_weight`, default 0.6/0.4). See `src/rag/hybrid_retriever.py`.

## Evidence classification

Deliberately does **not** use cosine similarity alone. Each candidate's
score is `0.7 * hybrid_score + 0.3 * keyword_overlap_ratio`, where
`keyword_overlap_ratio` is the fraction of the sub-question's non-stopword
keywords that literally appear in the candidate text. This combined score
is then bucketed into RELEVANT / WEAKLY_RELEVANT / IRRELEVANT against
configurable thresholds (`AgentConfig.evidence_relevance_threshold`).

**Important nuance found during development**: a passage that flatly
contradicts already-accepted evidence (e.g. "data is abundant" vs. "data is
scarce") can share almost no literal keywords with the sub-question wording
and would be filtered out as IRRELEVANT before ever being checked for
contradiction. The per-passage contradiction check therefore runs off the
raw semantic `hybrid_score` directly, not the keyword-adjusted
classification — see `evaluate_chunk()` in `src/agents/evidence_agent.py`.

## Evidence coverage

For each sub-question: `coverage_score = min(1.0, relevant_count / 3)`. The
target of 3 independent relevant passages per sub-question is a documented,
simple heuristic ("a few independent relevant passages," not one), not a
tuned/validated number — there was no labeled data available to tune it
against. A sub-question is "sufficient" once its coverage_score clears
`AgentConfig.evidence_sufficiency_threshold` (default 0.6, i.e. ~2 of the 3
target passages).

## Contradiction detection

Runs the configured NLI model (`cross-encoder/nli-deberta-v3-xsmall` by
default) pairwise over evidence whose relevance score clears a floor —
**across the entire evidence set, not just within one sub-question** (a
bug where cross-sub-question contradictions were silently missed was found
and fixed during development; see below). Same-`doc_id` pairs are never
compared, since that's restating one paper against itself, not an
inter-source disagreement. Only pairs the NLI model actually scores
`CONTRADICTION` above `AgentConfig.contradiction_score_threshold` (default
0.55) are reported.

**Likely-reason inference** (`_infer_disagreement_reason` in
`src/agents/contradiction_agent.py`) is entirely deterministic pattern
matching over the two passages' actual metadata/text (publication year,
metric-keyword sets, language-keyword sets) — it is a heuristic explanation
grounded in real fields, not a model guess, and falls back to an honest
"differing experimental settings ... not further specified" when no signal
is found.

## Research gap detection

Candidate gaps come from exactly two real signals, never invented:

1. **Coverage gaps** — sub-questions the Evidence Agent found insufficient
   coverage for, where at least some (even weak) related evidence exists.
2. **Limitation language** — accepted evidence whose text matches a
   regex of common hedge/limitation phrasing ("remains an open challenge,"
   "is limited," "has not been investigated," etc.).

Each candidate is then validated (`_validate_gap`) before being accepted:
rejected outright if it has no citable supporting document or if all its
evidence was classified IRRELEVANT. Gaps supported by only one paper are
still accepted (per the spec: "supported by one or more papers") but marked
with a lower heuristic confidence (0.5 vs. 0.75) and an explicit note.

**Confidence scores are an explicit, documented heuristic, not a calibrated
probability** — this is stated directly in every gap's
`confidence_methodology` field so nothing is presented as more rigorous
than it is.

## Verification

Each claim (the final answer, and each accepted gap's description) is
checked against its own cited evidence using the NLI model: a claim is
"supported" only if at least one cited evidence item scores `ENTAILMENT` at
or above 0.4 against the claim text. Without an NLI model, verification
degrades to the weaker (and clearly labeled) check of "does citable
evidence exist at all" — it never fabricates an entailment score it didn't
compute.

## Real bugs found and fixed during development

Documented here rather than hidden, since the project spec explicitly
requires not fabricating successful execution:

1. **Router substring false positive**: naive `phrase in text` matching let
   `"yo"` match inside `"you"`, misclassifying "What can you do?" and "Who
   are you?" as GREETING. Fixed with word-boundary regex matching.
2. **LangGraph silently drops undeclared state keys**: a node returned
   `{"_queries_by_subquestion": ...}`, a key not declared in the
   `ResearchState` TypedDict. LangGraph filtered it out with no error,
   causing the very next node to see an empty dict and retrieve nothing.
   Fixed by declaring `queries_by_subquestion` properly in the schema.
   This is a load-bearing lesson for anyone extending this graph: **every
   field a node needs downstream must be declared in `ResearchState`.**
3. **Contradiction detection scoped too narrowly**: originally grouped
   evidence by `sub_question` before comparing pairs, so two papers
   surfaced under different sub-questions were never checked against each
   other even when they directly disagreed. Fixed by comparing across the
   whole evidence set.
4. **Limitation-language regex too narrow**: matched the noun
   `"limitation"` but not the common phrasing `"is limited"`. Fixed by
   adding the adjective form and a couple of other common phrasings.
5. **Ablation override silently clobbered**: `run_ablation`'s
   `without_adaptive_retrieval` variant tried to force `max_iterations=0`
   via the initial state, but the planner node unconditionally overwrote
   `max_iterations` from config on every run. Fixed to respect a pre-set
   value if present.

All five were found by actually running the code against test inputs, not
by inspection — consistent with the project's "do not fabricate successful
execution" requirement.

## Known methodology limitations

- No labeled ground truth (qrels, gold answers) exists for this project, so
  retrieval recall and answer correctness are reported as "Not evaluated"
  rather than estimated.
- The embedding-based theme clustering (`sklearn.cluster.KMeans`) picks a
  fixed target cluster count rather than something like silhouette-score
  model selection; with the small evidence sets typical of a single
  research question, this is a reasonable simplification but is not
  validated against labeled theme boundaries.
- Model weights and the arXiv API were never actually reachable from the
  development sandbox this project was built in (no network access to
  huggingface.co or arxiv.org there); all model-touching code paths were
  validated with deterministic stub models instead. See `DEPLOYMENT.md` for
  what remains to be verified on first real deployment.
