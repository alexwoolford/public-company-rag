#!/usr/bin/env python3
"""
Test script for RAG query functionality.

Tests semantic search, company-specific queries, and answer generation
with sample queries on 10-K filings.
"""

import logging
import sys
import time
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from turbopuffer import Turbopuffer

from public_company_rag.config import (
    get_turbopuffer_api_key,
    get_turbopuffer_region,
    get_turbopuffer_namespace,
)
from public_company_rag.query import (
    generate_answer,
    get_stats,
    query_by_company,
    semantic_search,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def print_separator(title: str = "") -> None:
    """Print a formatted separator."""
    if title:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"{title:^60}")
        logger.info(f"{'=' * 60}\n")
    else:
        logger.info(f"\n{'-' * 60}\n")


def test_semantic_search(client: Turbopuffer, namespace_name: str) -> None:
    """Test basic semantic search."""
    print_separator("Test 1: Semantic Search")

    test_queries = [
        "What are the main risk factors for technology companies?",
        "How do companies describe artificial intelligence in their business?",
        "What supply chain challenges were mentioned in recent filings?",
    ]

    for query in test_queries:
        logger.info(f"Query: {query}")

        start_time = time.time()
        chunks, timing = semantic_search(client, namespace_name, query, top_k=5)
        duration = time.time() - start_time

        logger.info(f"Found {len(chunks)} results in {duration:.3f}s")
        logger.info(f"  Embedding: {timing['embedding']:.3f}s")
        logger.info(f"  Vector search: {timing['search']:.3f}s")

        if chunks:
            # Show top result
            top_result = chunks[0]
            logger.info(f"\nTop result:")
            logger.info(f"  Company: {top_result['company_name']} ({top_result['company_ticker']})")
            logger.info(f"  Section: {top_result['section_type']} ({top_result['filing_year']})")
            logger.info(f"  Text preview: {top_result['text'][:200]}...")
            if top_result['similarity_score']:
                logger.info(f"  Similarity: {top_result['similarity_score']:.4f}")

        print_separator()


def test_company_query(client: Turbopuffer, namespace_name: str) -> None:
    """Test company-specific query."""
    print_separator("Test 2: Company-Specific Query")

    # Apple's CIK
    apple_cik = "0000320193"
    query = "What are Apple's main products and services?"

    logger.info(f"Query: {query}")
    logger.info(f"Filter: CIK = {apple_cik}")

    start_time = time.time()
    chunks, timing = query_by_company(client, namespace_name, apple_cik, query, top_k=5)
    duration = time.time() - start_time

    logger.info(f"Found {len(chunks)} results in {duration:.3f}s")
    logger.info(f"  Embedding: {timing['embedding']:.3f}s")
    logger.info(f"  Vector search (filtered): {timing['search']:.3f}s")

    if chunks:
        for i, chunk in enumerate(chunks[:3], 1):
            logger.info(f"\n[{i}] {chunk['section_type']} ({chunk['filing_year']})")
            logger.info(f"    {chunk['text'][:150]}...")

    print_separator()


def test_answer_generation(client: Turbopuffer, namespace_name: str) -> None:
    """Test answer generation with RAG."""
    print_separator("Test 3: Answer Generation")

    question = "What are the main revenue sources for Apple Inc?"

    logger.info(f"Question: {question}")

    # Retrieve context
    logger.info("Retrieving context...")
    apple_cik = "0000320193"
    chunks, search_timing = query_by_company(client, namespace_name, apple_cik, question, top_k=5)

    if not chunks:
        logger.warning("No chunks found for context")
        return

    logger.info(f"Using {len(chunks)} chunks as context")
    logger.info(f"  Embedding: {search_timing['embedding']:.3f}s")
    logger.info(f"  Vector search: {search_timing['search']:.3f}s")

    # Generate answer
    logger.info("Generating answer...")
    start_time = time.time()
    answer, generation_time = generate_answer(question, chunks, include_citations=True)
    duration = time.time() - start_time

    logger.info(f"\nTiming breakdown:")
    logger.info(f"  Answer generation: {generation_time:.3f}s")
    logger.info(f"  Total: {duration:.3f}s")
    logger.info(f"\nAnswer:")
    logger.info(f"\n{answer}\n")

    print_separator()


def test_performance(client: Turbopuffer, namespace_name: str) -> None:
    """Test query performance with multiple runs."""
    print_separator("Test 4: Performance Testing")

    query = "What are the main business activities?"
    num_runs = 5

    logger.info(f"Running query {num_runs} times to measure performance...")
    logger.info(f"Query: {query}")

    durations = []
    embedding_times = []
    search_times = []

    for i in range(num_runs):
        start_time = time.time()
        chunks, timing = semantic_search(client, namespace_name, query, top_k=10)
        duration = time.time() - start_time
        durations.append(duration)
        embedding_times.append(timing['embedding'])
        search_times.append(timing['search'])
        logger.info(f"  Run {i+1}: {duration:.3f}s (embed: {timing['embedding']:.3f}s, search: {timing['search']:.3f}s, {len(chunks)} results)")

    logger.info(f"\nAverage times:")
    logger.info(f"  Embedding: {sum(embedding_times)/len(embedding_times):.3f}s")
    logger.info(f"  Vector search: {sum(search_times)/len(search_times):.3f}s")
    logger.info(f"  Total: {sum(durations)/len(durations):.3f}s")
    logger.info(f"Min: {min(durations):.3f}s, Max: {max(durations):.3f}s")

    print_separator()


def main() -> int:
    """Main entry point."""
    print_separator("RAG Query Tests")

    # Connect to Turbopuffer
    logger.info("Connecting to Turbopuffer...")
    try:
        client = Turbopuffer(
            api_key=get_turbopuffer_api_key(),
            region=get_turbopuffer_region(),
        )
        namespace_name = get_turbopuffer_namespace()
        logger.info("✓ Connected successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Turbopuffer: {e}")
        return 1

    # Get stats
    logger.info("Checking namespace...")
    try:
        stats = get_stats(client, namespace_name)
        logger.info(f"Namespace: {stats['namespace']}")
        logger.info(f"Exists: {stats.get('exists', 'unknown')}")
        logger.info(f"Accessible: {stats['accessible']}")
    except Exception as e:
        logger.warning(f"Could not get stats: {e}")

    # Run tests
    try:
        test_semantic_search(client, namespace_name)
        test_company_query(client, namespace_name)
        test_answer_generation(client, namespace_name)
        test_performance(client, namespace_name)

        print_separator("All Tests Complete")
        logger.info("✓ All tests passed successfully")
        return 0

    except KeyboardInterrupt:
        logger.warning("\n\nTests interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"\nTest failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
