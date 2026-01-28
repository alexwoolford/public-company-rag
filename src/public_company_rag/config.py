"""
Configuration management for public_company_rag.

Provides settings for Turbopuffer vector storage and OpenAI embeddings/LLM.
"""

import os
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Required: Turbopuffer API key, OpenAI API key
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Turbopuffer Configuration
    turbopuffer_api_key: str = Field(
        default="",
        description="Turbopuffer API key (required)",
    )
    turbopuffer_region: str = Field(
        default="api",
        description="Turbopuffer region (default: api)",
    )
    turbopuffer_namespace: str = Field(
        default="public_company_10k_chunks",
        description="Turbopuffer namespace for chunks",
    )

    # OpenAI Configuration (for query embeddings and answer generation)
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (required)",
    )

    # Query Configuration
    default_top_k: int = Field(
        default=10,
        description="Default number of chunks to retrieve",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model",
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Embedding dimension",
    )
    llm_model: str = Field(
        default="gpt-4o",
        description="LLM model for answer generation",
    )

    @field_validator(
        "turbopuffer_api_key",
        "openai_api_key",
        mode="before",
    )
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Strip whitespace from string values."""
        if isinstance(v, str):
            return v.strip()
        return v


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Convenience functions for accessing configuration


def get_turbopuffer_api_key() -> str:
    """Get Turbopuffer API key from settings."""
    key = get_settings().turbopuffer_api_key
    if not key:
        raise ValueError("TURBOPUFFER_API_KEY not set in .env file")
    return key


def get_turbopuffer_region() -> str:
    """Get Turbopuffer region from settings."""
    return get_settings().turbopuffer_region


def get_turbopuffer_namespace() -> str:
    """Get Turbopuffer namespace from settings."""
    return get_settings().turbopuffer_namespace


def get_openai_api_key() -> str:
    """Get OpenAI API key from settings."""
    key = get_settings().openai_api_key
    if not key:
        raise ValueError("OPENAI_API_KEY not set in .env file")
    return key


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent
