# Deployment Guide

## What was actually verified vs. what wasn't

Being direct about this up front, per the project's own requirement not to
claim untested success:

**Actually run and verified in the development sandbox:**
- `python -m compileall .` — clean, no syntax errors anywhere in the project.
- `pytest tests/` — all 49 tests pass. These exercise the router, hybrid
  retriever (real FAISS + real BM25, fake embedding vectors), every agent,
  the compiled LangGraph `StateGraph` end-to-end (real graph execution,
  real conditional-edge routing, real loop termination), and the
  evaluation/ablation framework.
- `streamlit.testing.v1.AppTest` — the actual `app.py` script, executed
  headlessly: confirmed a greeting produces zero exceptions and loads zero
  ML models (the whole point of the lazy-loading design — see below), and
  confirmed a real research question fails with a clear, user-facing error
  (not a raw traceback) when a required package isn't installed.

**NOT verified in the development sandbox (no network access to
`huggingface.co` or `export.arxiv.org` from that environment):**
- Actually downloading `BAAI/bge-small-en-v1.5`, `cross-encoder/nli-deberta-v3-xsmall`,
  or `google/flan-t5-small` from the Hugging Face Hub.
- Actually querying the real arXiv API via `scripts/prepare_data.py`
  (the XML-parsing logic itself WAS tested, against a realistic mocked
  Atom feed response — see `tests/` — but never against a live network call).
- A full end-to-end research query with real models producing a real
  answer, on Streamlit Cloud or anywhere else.
- `torch`/`transformers`/`sentence-transformers` installing cleanly
  together at the exact pinned versions in `requirements.txt` — those
  versions are well-established stable releases, not versions actually
  installed in this sandbox (see the `[untested-pin]` comments in
  `requirements.txt`).

**The first real research query on your actual Streamlit Cloud deployment
is genuinely the first time this full path will execute.** If something
in that path breaks, the most likely culprits are a version mismatch
between `torch`/`transformers`/`sentence-transformers`, or a Streamlit
Cloud resource limit during the initial model download.

## Steps

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: ResearchGapPilot"
git remote add origin <your-repo-url>
git push -u origin main
```

### 2. (Recommended) Prepare a real index before deploying

Running `scripts/prepare_data.py` locally (with internet access) and
committing the resulting `data/index/` directory means your deployed app
starts with real, topic-specific data immediately, rather than falling
back to the small bundled seed corpus:

```bash
pip install -r requirements.txt
python scripts/prepare_data.py --query "your research topic" --max-results 30
```

Note `data/index/` is in `.gitignore` by default — remove that line if you
want to commit a prepared index rather than rebuild it on the deployed
instance.

### 3. Create the Streamlit Cloud app

1. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app.
2. Point it at your repository, branch, and `app.py` as the entrypoint.
3. Streamlit Cloud will install `requirements.txt` and any system packages
   in `packages.txt` (currently empty — no system deps are needed).
4. Deploy. First boot will be slow: it needs to download `torch`,
   `transformers`, `sentence-transformers`, and then (on the first research
   query, not before — see the lazy-loading note below) the actual model
   weights from Hugging Face.

### 4. Verify

Test the exact scenarios from the project's own test plan once deployed:

1. `Hi` → conversational greeting, no research pipeline (check the "Agent
   Activity" expander only shows the Intent Router event).
2. `What can you do?` → capabilities response.
3. A real research question → full pipeline; check the tabs (Research
   Plan, Themes, Evidence Dashboard, Conflicting Evidence, Research Gaps,
   Sources, Verification) populate with real data.
4. A follow-up referencing the prior answer → should reuse session state,
   not restart from scratch.
5. `Explain gap 2.` with no prior research this session → should say
   "Please start a research question first," not invent an answer.
6. An empty message → friendly validation, not a crash.

## Design note: why models aren't loaded on page load

Early in development, `app.py` unconditionally loaded the embedding model
before even checking the user's intent — meaning a plain "Hi" would trigger
a multi-hundred-MB model download. This directly violated the project's
own requirement ("Do NOT load the entire RAG pipeline" for greetings) and
was caught by actually testing the app, not by inspection. The fix: `app.py`
now runs the same deterministic `route_intent()` check the graph itself
uses, *before* deciding whether to load any model at all. Only
`RESEARCH_QUERY` and `FOLLOW_UP` intents trigger `load_graph()` (which
loads the embedding/generation/NLI models and the retrieval index, all
`@st.cache_resource`-cached so it only happens once per running instance).
Every other intent resolves through `load_lightweight_graph()`, which
builds the same LangGraph with every model dependency set to `None` — free
to construct, since it's just Python objects with no model loading
involved.

## Resource considerations

- `torch` CPU wheel alone is roughly 700MB. Streamlit Cloud's free tier has
  storage/memory limits — if the app fails to boot with an out-of-memory or
  disk-space error, that's the most likely cause, not a bug in this code.
- The three Hugging Face models used are all deliberately small
  (`bge-small`, `nli-deberta-v3-xsmall`, `flan-t5-small`) specifically to
  fit CPU-only, free-tier hosting — see `src/config.py` for where to swap
  in larger models if you're deploying somewhere with more resources.

## Practical alternative if Streamlit Cloud resource limits are hit

If the free tier can't fit `torch` + the three models comfortably, the
architecture in `src/` is deployment-target-agnostic — the same
`build_graph()`/`WorkflowDependencies` pattern would run unmodified behind
a small FastAPI wrapper on any Docker-capable host (Render, Railway,
Fly.io, or a paid Hugging Face Docker Space) if Streamlit Cloud's limits
become a blocker.
