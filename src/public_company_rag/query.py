"""
Query module for RAG operations on Turbopuffer.

Provides high-level functions for semantic search, answer generation,
and company-specific queries using the Turbopuffer vector database.
"""

import logging
from typing import Any

from openai import OpenAI
from turbopuffer import Turbopuffer

from public_company_rag.config import get_settings

logger = logging.getLogger(__name__)


def create_query_embedding(text: str) -> list[float]:
    """
    Create embedding for query text using OpenAI.

    Args:
        text: Query text to embed

    Returns:
        Embedding vector (1536 dimensions)
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    response = client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )

    return response.data[0].embedding


def semantic_search(
    client: Turbopuffer,
    namespace_name: str,
    query_text: str,
    top_k: int | None = None,
    filters: tuple | None = None,
) -> list[dict[str, Any]]:
    """
    Perform semantic search on chunks.

    Args:
        client: Turbopuffer client
        namespace_name: Namespace name
        query_text: Natural language query
        top_k: Number of results to return (default from config)
        filters: Optional Turbopuffer filter tuple (e.g., ("company_cik", "Eq", "0000320193"))

    Returns:
        List of matched chunks with similarity scores and metadata
    """
    settings = get_settings()
    top_k = top_k or settings.default_top_k

    # Create query embedding
    logger.debug(f"Creating embedding for query: {query_text[:100]}...")
    query_vector = create_query_embedding(query_text)

    # Get namespace resource
    namespace = client.namespace(namespace_name)

    # Build rank_by parameter for vector search
    rank_by = ("vector", "ANN", query_vector)

    # Query Turbopuffer
    logger.debug(f"Querying Turbopuffer (top_k={top_k}, filters={filters})")
    response = namespace.query(
        rank_by=rank_by,
        top_k=top_k,
        filters=filters,
        include_attributes=True,
    )

    # Transform results to consistent format
    chunks = []
    for result in response.rows:
        # Use model_dump() to get all fields including custom attributes
        row_dict = result.model_dump() if hasattr(result, 'model_dump') else dict(result)

        chunks.append(
            {
                "chunk_id": row_dict.get("id", ""),
                "text": row_dict.get("text", ""),
                "company_cik": row_dict.get("company_cik", ""),
                "company_ticker": row_dict.get("company_ticker", ""),
                "company_name": row_dict.get("company_name", ""),
                "section_type": row_dict.get("section_type", ""),
                "filing_year": row_dict.get("filing_year", 0),
                "chunk_index": row_dict.get("chunk_index", 0),
                "similarity_score": row_dict.get("$dist"),
            }
        )

    logger.info(f"Found {len(chunks)} matching chunks")
    return chunks


def query_by_company(
    client: Turbopuffer,
    namespace_name: str,
    cik: str,
    query_text: str,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Semantic search filtered by company CIK.

    Args:
        client: Turbopuffer client
        namespace_name: Namespace name
        cik: Company CIK (e.g., "0000320193" for Apple)
        query_text: Natural language query
        top_k: Number of results to return

    Returns:
        List of matched chunks from the specified company
    """
    # Turbopuffer filter format: (attribute, operator, value)
    filters = ("company_cik", "Eq", cik)
    return semantic_search(client, namespace_name, query_text, top_k=top_k, filters=filters)


def generate_answer(
    question: str,
    context_chunks: list[dict[str, Any]],
    model: str | None = None,
    include_citations: bool = True,
) -> str:
    """
    Generate an answer to a question using retrieved chunks as context.

    Args:
        question: User's question
        context_chunks: Retrieved chunks to use as context
        model: LLM model to use (default from config)
        include_citations: If True, include chunk IDs as citations

    Returns:
        Generated answer string
    """
    settings = get_settings()
    model = model or settings.llm_model
    client = OpenAI(api_key=settings.openai_api_key)

    if not context_chunks:
        return "No relevant information found to answer the question."

    # Format context from chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        company_info = f"{chunk['company_name']} ({chunk['company_ticker']})" if chunk['company_ticker'] else chunk['company_name']
        section_info = f"{chunk['section_type']} - {chunk['filing_year']}" if chunk['filing_year'] else chunk['section_type']

        context_parts.append(
            f"[{i}] {company_info} - {section_info}\n{chunk['text']}\n"
        )

    context_text = "\n".join(context_parts)

    # Create prompt
    system_prompt = """You are a financial analyst assistant. Answer questions based on the provided context from 10-K filings.
Be accurate and concise. If the context doesn't contain enough information to answer fully, acknowledge this.
If requested, cite sources using [N] notation."""

    user_prompt = f"""Context from 10-K filings:

{context_text}

Question: {question}

Please provide a clear answer based on the context above."""

    if include_citations:
        user_prompt += " Include citations using [N] notation where appropriate."

    # Generate answer
    logger.debug(f"Generating answer with {model} (context: {len(context_chunks)} chunks)")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,  # Lower temperature for more factual responses
    )

    answer = response.choices[0].message.content

    logger.info(f"Generated answer ({len(answer)} chars)")
    return answer


def get_stats(client: Turbopuffer, namespace_name: str) -> dict[str, Any]:
    """
    Get statistics about the Turbopuffer namespace.

    Args:
        client: Turbopuffer client
        namespace_name: Namespace name

    Returns:
        Stats dict with chunk count and other metrics
    """
    try:
        # Get namespace resource
        namespace = client.namespace(namespace_name)

        # Check if namespace exists
        exists = namespace.exists()

        if not exists:
            return {
                "namespace": namespace_name,
                "exists": False,
                "accessible": False,
            }

        # Get namespace metadata
        metadata = namespace.metadata()

        return {
            "namespace": namespace_name,
            "exists": True,
            "accessible": True,
            "metadata": metadata,
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return {
            "namespace": namespace_name,
            "accessible": False,
            "error": str(e),
        }
