#!/usr/bin/env python3
"""
scripts/evaluate.py

Runs the real baseline-vs-agentic comparison and the ablation study against
a prepared index, and prints/saves the actually-measured results. This
script requires a prepared index (run prepare_data.py first) and the real
embedding/generation/NLI models (downloaded from Hugging Face on first use)
— it does not fabricate any numbers if those aren't available; it will
error out clearly instead.

Usage:
    python scripts/evaluate.py --index-dir data/index --output docs/evaluation_results.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import INDEX_DIR
from src.evaluation.benchmark import (
    ABLATION_VARIANTS,
    DEFAULT_BENCHMARK_QUESTIONS,
    run_ablation,
    run_comparison,
)
from src.graph.workflow import WorkflowDependencies
from src.models.llm import get_generation_model
from src.models.nli import get_nli_model
from src.rag.embeddings import get_embedding_model
from src.rag.hybrid_retriever import HybridRetriever


def build_real_dependencies(index_dir: Path) -> WorkflowDependencies:
    embedding_model = get_embedding_model()
    retriever = HybridRetriever(embedding_model=embedding_model)
    retriever.load(index_dir)

    return WorkflowDependencies(
        retriever=retriever,
        embedding_model=embedding_model,
        generation_model=get_generation_model(),
        nli_model=get_nli_model(),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate ResearchGapPilot against baselines and run the ablation study.")
    parser.add_argument("--index-dir", type=str, default=str(INDEX_DIR))
    parser.add_argument("--questions", type=str, help="Optional path to a text file with one benchmark question per line.")
    parser.add_argument("--output", type=str, default="docs/evaluation_results.json")
    parser.add_argument("--skip-ablation", action="store_true", help="Only run the baseline comparison, skip the ablation study.")
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    if not (index_dir / "chunks.jsonl").exists():
        print(
            f"No prepared index found at {index_dir}. Run scripts/prepare_data.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    questions = DEFAULT_BENCHMARK_QUESTIONS
    if args.questions:
        with open(args.questions) as f:
            questions = [line.strip() for line in f if line.strip()]

    print(f"Loading models and prepared index from {index_dir}...")
    deps = build_real_dependencies(index_dir)

    print(f"Running comparison across {len(questions)} benchmark question(s)...")
    comparison_results = run_comparison(questions, deps)

    ablation_results = []
    if not args.skip_ablation:
        print(f"Running ablation study ({len(ABLATION_VARIANTS)} variants) on the first benchmark question...")
        for variant in ABLATION_VARIANTS:
            print(f"  - {variant}")
            result = run_ablation(questions[0], deps, variant)
            ablation_results.append(
                {
                    "variant": variant,
                    "metrics": result["metrics"],
                    "final_answer_preview": result["final_state"].get("final_answer", "")[:200],
                }
            )

    output = {
        "benchmark_questions": questions,
        "comparison": comparison_results,
        "ablation": ablation_results,
        "note": "All figures above are measured from real runs against the prepared index. "
                "Any metric requiring ground-truth labels not available in this project "
                "(e.g. retrieval_recall, answer_correctness) is reported as null/Not evaluated.",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
