# Architecture

## Overview

ResearchGapPilot is built as a [LangGraph](https://github.com/langchain-ai/langgraph)
`StateGraph` — a real state machine, not a linear chain — with conditional
edges that let the graph loop back for additional retrieval or verification
cycles within bounded limits.

## Two paths through the graph

```
                       intent_router
                            |
        +-------------------+-------------------+
        |                                        |
  GREETING / CAPABILITIES /               RESEARCH_QUERY
  CLARIFICATION / UNSUPPORTED                    |
        |                                     planner
  conversational_reply                           |
        |                                 query_generator
       END                                       |
                                              retriever
                                                  |
                                          evidence_evaluator <---+
                                                  |              |
                                       contradiction_detector    |
                                                  |              |
                                    (sufficient?) -- no -------->|
                                          |                adaptive_retrieval
                                         yes
                                          |
                                       reasoning
                                          |
                                     theme_detector
                                          |
                                      gap_detector
                                          |
                                      verification <-------------+
                                          |                       |
                              (all supported?) -- no ------------>|
                                          |
                                         yes
                                          |
                                    final_generator
                                          |
                                         END


              FOLLOW_UP  -->  follow_up_handler
                                    |
                     (needs_more_evidence?) -- yes --> adaptive_retrieval (joins the loop above)
                                    |
                                   no
                                    |
                                reasoning (joins the pipeline above)
```

Every arrow with a `(condition?)` label is a real conditional edge (see
`src/graph/routing.py`), evaluated against the actual state produced by the
previous node — never a scripted/hardcoded transition.

## Loop termination

There are two independent bounded loops, both capped by values from
`src/config.py` (`AgentConfig.max_retrieval_iterations`,
`AgentConfig.max_verification_cycles`):

1. **Retrieval loop**: `contradiction_detector -> adaptive_retrieval ->
   evidence_evaluator -> contradiction_detector`, bounded by `iteration`.
2. **Verification loop**: `verification -> adaptive_retrieval -> ... ->
   verification`, bounded by `verification_cycle`.

Both counters are checked in `src/graph/routing.py` before allowing another
loop iteration; once either cap is hit, the graph proceeds forward
regardless of whether coverage/verification is fully satisfied — the
Reasoning/Verification agents are responsible for being honest about
insufficient evidence rather than the graph blocking forever.

## Why a Web Worker isn't relevant here but a similar principle applies

Streamlit's `@st.cache_resource` plays the role that a singleton service
would play in a always-on backend: the embedding model, generation model,
NLI model, and prepared retrieval index are all loaded/built exactly once
per running instance and reused across every user interaction and every
session — never reloaded per-query. See `app.py`'s `load_*` functions.

## Agent responsibilities

| Agent | File | Responsibility |
|---|---|---|
| Intent Router | `src/router.py` | Classify user input before anything expensive runs |
| Planner | `src/agents/planner.py` | Decompose a research question into sub-questions |
| Query Generator | `src/agents/query_generator.py` | Derive multiple retrieval queries per sub-question |
| Retriever | `src/agents/retrieval_agent.py` | Run hybrid retrieval, dedupe, optionally rerank |
| Evidence Agent | `src/agents/evidence_agent.py` | Classify passages, compute real coverage |
| Contradiction Agent | `src/agents/contradiction_agent.py` | NLI-compare evidence pairs, infer likely reasons |
| Reasoning Agent | `src/agents/reasoning_agent.py` | Synthesize a grounded answer or say "insufficient evidence" |
| Theme Agent | `src/agents/theme_agent.py` | Cluster evidence into research themes |
| Gap Agent | `src/agents/gap_agent.py` | Propose and validate candidate research gaps |
| Verification Agent | `src/agents/verification_agent.py` | NLI-check claims against their citations |

## State schema

See `src/state.py` for the full `ResearchState` TypedDict. One important
implementation detail: **LangGraph silently drops any state key that isn't
declared in the schema** — a partial state update returning an undeclared
key simply vanishes rather than raising an error. This was discovered
during development (see `docs/methodology.md`) and is the reason every
field a node needs to pass downstream, including internal-only ones like
`queries_by_subquestion`, is explicitly declared in `ResearchState`.
