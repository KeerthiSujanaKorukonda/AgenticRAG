#!/usr/bin/env python3
"""
scripts/prepare_data.py

Real data acquisition: queries the public arXiv API (no key required) for
real papers matching a topic, extracts real metadata (title/authors/year/
abstract/URL), chunks the real abstract text, embeds it, and builds the
FAISS + BM25 indexes the app queries at runtime.

Usage:
    python scripts/prepare_data.py --query "multilingual NLP low-resource" --max-results 30
    python scripts/prepare_data.py --queries queries.txt --max-results 20  # one query per line

This script requires outbound internet access to export.arxiv.org and to
huggingface.co (to download the embedding model). It is meant to be run
once (or occasionally, to refresh the corpus) — not on every app startup;
see DEPLOYMENT.md.
"""

import argparse
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import INDEX_DIR, retrieval as retrieval_cfg
from src.rag.embeddings import EmbeddingModel
from src.rag.hybrid_retriever import HybridRetriever
from src.utils.text import chunk_text, clean_text

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def fetch_arxiv_papers(query: str, max_results: int = 30, retries: int = 3) -> List[Dict]:
    """
    Queries the real arXiv API and returns real paper metadata. No API key
    is required — this is arXiv's public export endpoint. Retries on
    transient network errors since this is meant to run unattended in CI/
    deployment prep, not interactively.
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"

    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                raw_xml = response.read()
            break
        except Exception as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"Failed to fetch from arXiv API after {retries} attempts: {last_error}")

    root = ET.fromstring(raw_xml)
    papers = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title_el = entry.find("atom:title", ATOM_NS)
        summary_el = entry.find("atom:summary", ATOM_NS)
        id_el = entry.find("atom:id", ATOM_NS)
        published_el = entry.find("atom:published", ATOM_NS)
        authors = [
            a.find("atom:name", ATOM_NS).text.strip()
            for a in entry.findall("atom:author", ATOM_NS)
            if a.find("atom:name", ATOM_NS) is not None
        ]

        if title_el is None or summary_el is None or id_el is None:
            continue  # skip malformed entries rather than fabricate fields

        arxiv_url = id_el.text.strip()
        arxiv_id = arxiv_url.rsplit("/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)  # strip version suffix for stable doc_ids
        year = None
        if published_el is not None and published_el.text:
            try:
                year = int(published_el.text[:4])
            except ValueError:
                year = None

        papers.append(
            {
                "doc_id": f"arxiv:{arxiv_id}",
                "title": clean_text(title_el.text),
                "authors": authors,
                "year": year,
                "source": "arXiv",
                "url": arxiv_url,
                "abstract": clean_text(summary_el.text),
                "topic": query,
            }
        )

    return papers


def build_chunks_from_papers(papers: List[Dict]) -> List[Dict]:
    """Chunk each paper's real abstract text into overlapping word windows,
    preserving all metadata on every chunk."""
    chunks = []
    for paper in papers:
        pieces = chunk_text(
            paper["abstract"],
            chunk_size_words=retrieval_cfg.chunk_size_words,
            overlap_words=retrieval_cfg.chunk_overlap_words,
        )
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "doc_id": paper["doc_id"],
                    "chunk_id": f"{paper['doc_id']}-{i}",
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "year": paper["year"],
                    "source": paper["source"],
                    "url": paper["url"],
                    "text": piece,
                    "topic": paper.get("topic"),
                }
            )
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Prepare the ResearchGapPilot retrieval index from real arXiv data.")
    parser.add_argument("--query", type=str, help="A single topic query, e.g. 'multilingual NLP low-resource'.")
    parser.add_argument("--queries", type=str, help="Path to a text file with one query per line.")
    parser.add_argument("--max-results", type=int, default=30, help="Max papers to fetch per query.")
    parser.add_argument("--output-dir", type=str, default=str(INDEX_DIR), help="Where to save the prepared index.")
    args = parser.parse_args()

    queries = []
    if args.query:
        queries.append(args.query)
    if args.queries:
        with open(args.queries) as f:
            queries.extend(line.strip() for line in f if line.strip())

    if not queries:
        print("No query provided. Use --query 'topic' or --queries path/to/file.txt", file=sys.stderr)
        sys.exit(1)

    all_papers: List[Dict] = []
    seen_ids = set()

    for query in queries:
        print(f"Querying arXiv for: {query!r} (max {args.max_results} results)...")
        try:
            papers = fetch_arxiv_papers(query, max_results=args.max_results)
        except Exception as exc:
            print(f"  Failed to fetch results for '{query}': {exc}", file=sys.stderr)
            continue

        new_count = 0
        for paper in papers:
            if paper["doc_id"] in seen_ids:
                continue
            seen_ids.add(paper["doc_id"])
            all_papers.append(paper)
            new_count += 1
        print(f"  Retrieved {len(papers)} papers ({new_count} new).")

    if not all_papers:
        print("No papers were retrieved from arXiv. Nothing to index.", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal unique papers: {len(all_papers)}")
    print("Chunking abstracts...")
    chunks = build_chunks_from_papers(all_papers)
    print(f"Produced {len(chunks)} chunks.")

    print("Loading embedding model (this downloads the model on first run)...")
    embedding_model = EmbeddingModel()
    embedding_model.load()

    print("Building hybrid retriever (FAISS + BM25)...")
    retriever = HybridRetriever(embedding_model=embedding_model)
    retriever.build(chunks)

    output_dir = Path(args.output_dir)
    print(f"Saving prepared index to {output_dir}...")
    retriever.save(output_dir)

    print("\nDone. The app will load this prepared index at startup instead of "
          "rebuilding it, per the deployment constraints in the project spec.")


if __name__ == "__main__":
    main()
