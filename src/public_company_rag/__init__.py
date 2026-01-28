"""Public Company RAG - Turbopuffer-based retrieval augmented generation for 10-K filings."""

from turbopuffer import Turbopuffer

from public_company_rag.config import (
    get_turbopuffer_api_key,
    get_turbopuffer_region,
    get_turbopuffer_namespace,
)
from public_company_rag.query import (
    semantic_search,
    query_by_company,
    generate_answer,
    get_stats,
)

__version__ = "1.0.0"


def create_client() -> Turbopuffer:
    """
    Create a Turbopuffer client with credentials from config.

    Returns:
        Configured Turbopuffer client

    Raises:
        ValueError: If required credentials are missing
    """
    return Turbopuffer(
        api_key=get_turbopuffer_api_key(),
        region=get_turbopuffer_region(),
    )


def get_namespace_name() -> str:
    """
    Get the configured namespace name.

    Returns:
        Namespace name from config
    """
    return get_turbopuffer_namespace()


__all__ = [
    "create_client",
    "get_namespace_name",
    "semantic_search",
    "query_by_company",
    "generate_answer",
    "get_stats",
]
