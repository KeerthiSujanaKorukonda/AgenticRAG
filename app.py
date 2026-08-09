"""
ResearchGapPilot — Streamlit application entrypoint.

Chat-style UI over the LangGraph agentic RAG workflow defined in
src/graph/workflow.py. Models and the retrieval index are loaded once via
@st.cache_resource; the compiled graph is likewise built once and reused
across every user interaction in the session.
"""

import logging
import time
from pathlib import Path

import streamlit as st

from src.config import SEED_PAPERS_PATH, INDEX_DIR, agents as agents_cfg, app as app_cfg, retrieval as retrieval_cfg
from src.graph.workflow import WorkflowDependencies, build_graph
from src.models.llm import get_generation_model
from src.models.nli import get_nli_model
from src.rag.embeddings import get_embedding_model
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.reranker import get_reranker
from src.router import route_intent
from src.utils.text import chunk_text, clean_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

st.set_page_config(page_title="ResearchGapPilot", page_icon="🔬", layout="wide")


# --------------------------------------------------------------------- #
# Cached resource loading — every one of these runs exactly once per
# deployed instance, not per user query.
# --------------------------------------------------------------------- #


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    model = get_embedding_model()
    model.load()
    return model


@st.cache_resource(show_spinner="Preparing retrieval index...")
def load_retriever(_embedding_model):
    """
    Loads a prepared index from data/index if one exists (built ahead of
    time by scripts/prepare_data.py against real arXiv data). If none
    exists yet, falls back to building a small index directly from the
    bundled data/seed_papers.jsonl real-paper seed corpus, once, and saves
    it so subsequent app restarts reuse it instead of rebuilding.
    """
    retriever = HybridRetriever(embedding_model=_embedding_model)

    if (INDEX_DIR / "chunks.jsonl").exists():
        retriever.load(INDEX_DIR)
        return retriever, "prepared"

    if not SEED_PAPERS_PATH.exists():
        return retriever, "empty"

    import json

    chunks = []
    with open(SEED_PAPERS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            paper = json.loads(line)
            pieces = chunk_text(paper["summary"], chunk_size_words=retrieval_cfg.chunk_size_words, overlap_words=retrieval_cfg.chunk_overlap_words)
            for i, piece in enumerate(pieces):
                chunks.append(
                    {
                        "doc_id": paper["doc_id"],
                        "chunk_id": f"{paper['doc_id']}-{i}",
                        "title": paper["title"],
                        "authors": paper.get("authors", []),
                        "year": paper.get("year"),
                        "source": paper.get("source", ""),
                        "url": paper.get("url", ""),
                        "text": piece,
                        "topic": paper.get("topic"),
                    }
                )

    retriever.build(chunks)
    try:
        retriever.save(INDEX_DIR)
    except Exception:
        pass  # non-fatal if the filesystem is read-only in this environment
    return retriever, "seed_fallback"


@st.cache_resource(show_spinner="Loading generation model...")
def load_generation_model():
    model = get_generation_model()
    return model


@st.cache_resource(show_spinner="Loading NLI model...")
def load_nli_model():
    model = get_nli_model()
    return model


@st.cache_resource(show_spinner="Loading reranker...")
def load_reranker():
    return get_reranker()


def zero_shot_classifier_fn(text: str, candidate_labels):
    """Adapter so the router's expected call signature matches a HF
    zero-shot-classification pipeline, using the same generation model's
    underlying pipeline machinery lazily via transformers directly."""
    from transformers import pipeline as hf_pipeline
    from src.config import models as model_cfg

    if not hasattr(zero_shot_classifier_fn, "_pipeline"):
        zero_shot_classifier_fn._pipeline = hf_pipeline("zero-shot-classification", model=model_cfg.zero_shot_model)
    return zero_shot_classifier_fn._pipeline(text, candidate_labels)


@st.cache_resource(show_spinner=False)
def load_graph(enable_generation: bool, enable_nli: bool, enable_reranking: bool):
    """
    Builds the full graph WITH real models loaded. This is deliberately
    NOT called unconditionally at page load — see the routing pre-check
    below, which only calls this once a message is actually classified as
    RESEARCH_QUERY or an evidence-needing FOLLOW_UP. A plain "Hi" must
    never trigger a model download or index load.
    """
    embedding_model = load_embedding_model()
    retriever, index_status = load_retriever(embedding_model)

    deps = WorkflowDependencies(
        retriever=retriever,
        embedding_model=embedding_model,
        generation_model=load_generation_model() if enable_generation else None,
        nli_model=load_nli_model() if enable_nli else None,
        reranker=load_reranker() if enable_reranking else None,
        zero_shot_classifier=None,  # deterministic router rules cover the vast majority of input; see src/router.py
    )
    return build_graph(deps), index_status, retriever


@st.cache_resource(show_spinner=False)
def load_lightweight_graph():
    """
    A graph with every model dependency set to None. Sufficient to resolve
    GREETING / CAPABILITIES / CLARIFICATION / UNSUPPORTED_REQUEST /
    dangling-follow-up-reference intents (see src/graph/workflow.py's
    conversational_reply_node) without loading anything heavy. Building
    this costs nothing — WorkflowDependencies() with no args is just plain
    Python objects.
    """
    return build_graph(WorkflowDependencies())


# --------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------- #

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "research_state" not in st.session_state:
    st.session_state.research_state = {}


def reset_research():
    st.session_state.research_state = {}
    st.session_state.conversation_history = []


# --------------------------------------------------------------------- #
# Sidebar — research settings
# --------------------------------------------------------------------- #

st.sidebar.title("Research Settings")
max_papers = st.sidebar.slider("Max papers", min_value=5, max_value=retrieval_cfg.max_papers, value=20)
top_k = st.sidebar.slider("Top-K per query", min_value=3, max_value=15, value=retrieval_cfg.top_k_hybrid)
max_iterations = st.sidebar.slider("Maximum retrieval iterations", min_value=1, max_value=5, value=agents_cfg.max_retrieval_iterations)
evidence_threshold = st.sidebar.slider("Evidence relevance threshold", min_value=0.1, max_value=0.9, value=agents_cfg.evidence_relevance_threshold, step=0.05)
enable_contradiction = st.sidebar.checkbox("Enable contradiction detection", value=True)
enable_reranking = st.sidebar.checkbox("Enable reranking (slower on CPU)", value=False)
enable_generation = st.sidebar.checkbox("Enable generation model (final report synthesis)", value=True)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 New Research", use_container_width=True):
    reset_research()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Models are Hugging Face only — no paid APIs. First load downloads model "
    "weights and can take a few minutes on a fresh Streamlit Cloud instance."
)

# --------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------- #

st.title(app_cfg.app_title)
st.caption(app_cfg.app_subtitle)

# NOTE: models/retrieval index are intentionally NOT loaded here. They are
# loaded lazily, only once a message is actually classified as a research
# question — see the routing pre-check in the chat handler below.
index_status = None

# --------------------------------------------------------------------- #
# Chat history render
# --------------------------------------------------------------------- #

for turn in st.session_state.conversation_history[-app_cfg.max_conversation_turns :]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])


# --------------------------------------------------------------------- #
# Main chat input
# --------------------------------------------------------------------- #

user_input = st.chat_input("Ask a research question...")

if user_input:
    if len(user_input) > 2000:
        st.error("That message is quite long — please shorten it to under 2000 characters.")
        st.stop()

    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    graph_input = dict(st.session_state.research_state)
    graph_input["user_input"] = user_input
    graph_input.setdefault("agent_events", [])
    graph_input["max_iterations"] = max_iterations

    # --- Cheap routing pre-check, before touching any model ---
    # route_intent is pure Python/regex — no models, no network, no cost.
    # Only if it resolves to something that genuinely needs retrieval do we
    # pay for loading the embedding/generation/NLI models and the index.
    pre_check = route_intent(
        user_input,
        has_prior_research_state=bool(st.session_state.research_state.get("research_question")),
    )
    needs_full_pipeline = pre_check["intent"] in ("RESEARCH_QUERY",) or (
        pre_check["intent"] == "FOLLOW_UP"
    )

    with st.chat_message("assistant"):
        graph = None
        pipeline_load_failed = False

        if needs_full_pipeline:
            try:
                with st.spinner("Preparing models and retrieval index (first load can take a few minutes)..."):
                    graph, index_status, retriever = load_graph(enable_generation, enable_contradiction, enable_reranking)
                if index_status == "seed_fallback" and not st.session_state.get("_seed_notice_shown"):
                    st.info(
                        "Using the bundled real-paper seed corpus (20 arXiv papers across low-resource "
                        "NLP, RAG, and hallucination topics) as a demo dataset. Run `scripts/prepare_data.py` "
                        "with your own topic to build a larger, topic-specific index.",
                        icon="ℹ️",
                    )
                    st.session_state["_seed_notice_shown"] = True
                elif index_status == "empty":
                    st.warning(
                        "No retrieval index and no seed corpus were found. This research question "
                        "will not return evidence until you run scripts/prepare_data.py.",
                        icon="⚠️",
                    )
            except ModuleNotFoundError as exc:
                pipeline_load_failed = True
                st.error(
                    f"A required package isn't installed in this environment ({exc}). "
                    "Install the full requirements.txt (torch, transformers, sentence-transformers, "
                    "faiss-cpu, rank_bm25) to run real research queries.",
                    icon="🚫",
                )
            except Exception as exc:
                pipeline_load_failed = True
                logger.exception("Failed to load models/retrieval index")
                st.error(
                    f"Couldn't load the research pipeline's models or index: {exc}. "
                    "This is often a network issue reaching Hugging Face, or a missing prepared index — "
                    "see DEPLOYMENT.md.",
                    icon="🚫",
                )

        if pipeline_load_failed:
            reply = (
                "I couldn't load the models needed to research that question right now "
                "(see the error above). Please try again once the environment is set up correctly."
            )
            st.session_state.conversation_history.append({"role": "assistant", "content": reply})
            st.stop()

        if graph is None:
            # GREETING / CAPABILITIES / CLARIFICATION / UNSUPPORTED_REQUEST /
            # dangling follow-up reference — resolved with zero models loaded.
            graph = load_lightweight_graph()

        activity_placeholder = st.empty()
        with st.spinner("Working..."):
            try:
                final_state = graph.invoke(graph_input)
            except Exception as exc:
                logger.exception("Graph execution failed")
                st.error(f"Something went wrong while processing that request: {exc}")
                st.stop()

        # Render agent activity (real events from this run only).
        new_events = final_state.get("agent_events", [])[len(graph_input.get("agent_events", [])) :]
        if new_events:
            with st.expander("🔎 Agent Activity", expanded=False):
                for event in new_events:
                    st.markdown(f"{event['icon']} **{event['agent']}**  \n{event['message']}")

        reply = final_state.get("final_answer", "")
        st.markdown(reply)

        # If this was a real research run (not just a conversational reply),
        # render the full dashboard below the chat message.
        if final_state.get("intent") == "RESEARCH_QUERY" or (
            final_state.get("intent") == "FOLLOW_UP" and final_state.get("research_themes")
        ):
            sections = final_state.get("final_report_sections", {})

            tabs = st.tabs([
                "Research Plan", "Themes", "Evidence Dashboard",
                "Conflicting Evidence", "Research Gaps", "Sources", "Verification",
            ])

            with tabs[0]:
                st.markdown(sections.get("research_plan", "Not available."))

            with tabs[1]:
                for theme in final_state.get("research_themes", []):
                    st.markdown(f"**{theme['title']}** — {theme['evidence_count']} evidence item(s)")
                    st.caption(theme.get("description", ""))

            with tabs[2]:
                coverage = final_state.get("evidence_coverage", {})
                st.metric("Overall evidence coverage", coverage.get("overall_coverage_score", "Not evaluated"))
                for sq, stats in coverage.get("per_subquestion", {}).items():
                    st.write(f"**{sq}** — relevant: {stats['relevant_count']}, coverage: {stats['coverage_score']}, sufficient: {stats['sufficient']}")

                st.markdown("#### Retrieved Papers")
                for chunk in final_state.get("retrieved_documents", [])[:max_papers]:
                    with st.expander(f"{chunk.get('title', 'Untitled')} — hybrid score {chunk.get('hybrid_score', 0):.2f}"):
                        st.write(f"**Authors:** {', '.join(chunk.get('authors', []) or ['Unknown'])}")
                        st.write(f"**Year:** {chunk.get('year', 'Unknown')}")
                        st.write(f"**URL:** {chunk.get('url', 'N/A')}")
                        st.write(f"**Semantic score:** {chunk.get('semantic_score', 0):.3f}  |  **BM25 score:** {chunk.get('bm25_score', 0):.3f}  |  **Hybrid score:** {chunk.get('hybrid_score', 0):.3f}")
                        st.write(f"**Retrieved for sub-question:** {chunk.get('sub_question', 'N/A')}")
                        st.write("**Passage:**")
                        st.write(chunk.get("text", ""))

            with tabs[3]:
                contradictions = final_state.get("contradictions", [])
                if not contradictions:
                    st.write("No conflicting evidence detected.")
                for c in contradictions:
                    st.warning(f"⚠️ Conflicting Evidence (NLI score {c['nli_score']})")
                    st.write(f"**{c['source_a']}:** {c['statement_a']}")
                    st.write(f"**{c['source_b']}:** {c['statement_b']}")
                    st.caption(f"Likely reason: {c['likely_reason']}")

            with tabs[4]:
                gaps = final_state.get("research_gaps", [])
                if not gaps:
                    st.write("No validated research gaps were identified from the retrieved evidence.")
                for gap in gaps:
                    st.markdown(f"### {gap['title']}")
                    st.write(gap["description"])
                    st.caption(f"Category: {gap['category']} | Confidence: {gap['confidence']} ({gap['confidence_methodology']})")
                    st.write(f"**Why it matters:** {gap['why_it_matters']}")
                    st.write(f"**Supporting papers:** {', '.join(gap.get('supporting_doc_ids', [])) or 'None'}")

            with tabs[5]:
                citations = final_state.get("citations", [])
                if not citations:
                    st.write("No citable sources.")
                for c in citations:
                    authors = ", ".join(c.get("authors", [])[:3])
                    st.write(f"[{c['citation_id']}] **{c['title']}** ({c.get('year', 'n.d.')}) — {authors}")
                    if c.get("url"):
                        st.write(c["url"])

            with tabs[6]:
                results = final_state.get("verification_results", [])
                if not results:
                    st.write("Not evaluated.")
                for r in results:
                    icon = "✅" if r.get("supported") else "❌"
                    st.write(f"{icon} {r['claim'][:150]}")
                    st.caption(r.get("notes", ""))

    st.session_state.conversation_history.append({"role": "assistant", "content": reply})

    # Persist research state for follow-ups, but drop internal-only keys.
    persisted_state = {
        k: v for k, v in final_state.items()
        if k not in ("agent_events",)  # keep events out of persisted state to bound memory growth
    }
    st.session_state.research_state = persisted_state


if not st.session_state.conversation_history:
    st.markdown(
        "*Try asking:* \"What are the research gaps in low-resource multilingual NLP?\" "
        "or \"What are the main limitations of retrieval-augmented generation?\""
    )
