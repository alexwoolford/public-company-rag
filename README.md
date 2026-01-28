# Public Company RAG

**Retrieval Augmented Generation (RAG) for SEC 10-K filings using Turbopuffer vector search and OpenAI.**

Query 2.85 million chunks from 5,410 public companies to answer questions about:
- Risk factors and business challenges
- Revenue sources and financial performance
- Supply chain and operational issues
- Technology adoption (AI, cloud, etc.)
- Strategic initiatives and competition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Example

Ask business questions across 5,410 public companies and get AI-generated answers with citations:

```bash
$ python scripts/chat_rag.py --top-k 200

Question: Which companies would be impacted by a shortage of helium?

Searching with Turbopuffer RAG...
Retrieved: 200 chunks
Companies: TUSK, PWR, TRCK, WHD, WDFC

================================================================================
ANSWER:
================================================================================

Several companies would be impacted by a shortage of helium:

1. **New ERA Energy & Digital, Inc. (NUAI)**: Involved in helium exploration
   and production. Operations could be affected by supply shortages and price
   fluctuations [1][2].

2. **GE HealthCare Technologies Inc. (GEHC)**: Relies on helium for products.
   Volatility in helium availability could constrain manufacturing and reduce
   profit margins [13][107].

3. **Air Products & Chemicals, Inc. (APD)**: Depends on natural gas production
   for crude helium supply. Lower natural gas production could reduce helium
   supplies [12][15].

4. **ATI Inc. (ATI)**: Relies on helium as critical supply. Price and
   availability fluctuations could impact manufacturing processes [159].

5. **ESCO Technologies Inc. (ESE)**: Uses helium in A&D segment. Shortages
   could impact operations [87].

6. **NVE Corp (NVEC)**: Requires helium for production processes. Concerned
   about supply interruptions [122].

These companies have highlighted helium as a critical component, and any
shortage could significantly impact their business activities.
```

**What makes this powerful:**
- Query across 2.85M chunks from SEC 10-K filings (2024-2025)
- Semantic search finds relevant companies even if they don't explicitly mention "helium shortage"
- AI generates comprehensive answers with source citations
- Sub-3 second response time for complex queries

## Quick Start

### 1. Install

```bash
# Clone repository
git clone https://github.com/alexwoolford/public-company-rag.git
cd public-company-rag

# Create conda environment and install
conda env create -f environment.yml
conda activate public-company-rag
pip install -e .
```

### 2. Configure API Keys

Create a `.env` file with your API keys:

```bash
cp .env.example .env
# Edit .env and add your keys:
#   TURBOPUFFER_API_KEY=your_key_here
#   OPENAI_API_KEY=your_key_here
```

**Get API keys:**
- **Turbopuffer**: https://turbopuffer.com (vector database for 2.85M chunks)
- **OpenAI**: https://platform.openai.com/api-keys (used for embeddings and answer generation)

### 3. Setup Data (One-time, ~40-70 minutes)

Download the pre-exported dataset and upload to your Turbopuffer namespace:

```bash
python scripts/setup_data.py
```

This will:
- Download ~20GB dataset from Hugging Face (cached locally)
- Upload 2.85M chunks to your Turbopuffer namespace
- Verify the upload was successful

**Note**: This is a one-time setup. The data is then permanently available in your Turbopuffer account.

### 4. Start Querying

#### Interactive Chat

```bash
python scripts/chat_rag.py

# Focus on specific company (Apple)
python scripts/chat_rag.py --company 0000320193

# Retrieve more chunks per query
python scripts/chat_rag.py --top-k 20
```

#### Python API

```python
from public_company_rag import create_client, get_namespace_name, semantic_search, query_by_company, generate_answer

# Initialize client
client = create_client()
namespace_name = get_namespace_name()

# Semantic search
chunks = semantic_search(
    client,
    namespace_name,
    "What are the main risks for technology companies?",
    top_k=10
)

# Company-specific query (Apple's CIK: 0000320193)
chunks = query_by_company(
    client,
    namespace_name,
    "0000320193",
    "What are Apple's main revenue sources?",
    top_k=10
)

# Generate answer from retrieved chunks
answer = generate_answer("What are Apple's main revenue sources?", chunks)
print(answer)
```

#### Run Tests

```bash
python scripts/test_rag_query.py
```

## Dataset

The underlying dataset is available on Hugging Face:

- **Dataset**: [alexwoolford/public-company-10k-chunks](https://huggingface.co/datasets/alexwoolford/public-company-10k-chunks)
- **Version**: 1.0.0
- **Chunks**: 2,849,633
- **Companies**: 5,410 public companies
- **Embeddings**: OpenAI text-embedding-3-small (1536-dim)
- **Filing years**: Multiple years of 10-K filings
- **Sections**: Business description, risk factors, management discussion, and full filings

### Dataset Schema

Each chunk contains:
- `id`: Unique chunk identifier (format: `{cik}_{section}_{year}_chunk_{index}`)
- `text`: Chunk text content
- `vector`: Embedding vector (1536 dimensions)
- `company_cik`: SEC Central Index Key
- `company_ticker`: Stock ticker symbol
- `company_name`: Company name
- `section_type`: Filing section (e.g., "business_description", "risk_factors")
- `filing_year`: Year of the filing
- `chunk_index`: Position within the document

### Citation

If you use this dataset in research, please cite:

```bibtex
@dataset{woolford2025public_company_rag,
  title={Public Company 10-K Chunks with Embeddings},
  author={Woolford, Alex},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/alexwoolford/public-company-10k-chunks}
}
```

## Features

- **Semantic Search**: Vector similarity search across all 10-K chunks
- **Company Filtering**: Query specific companies by CIK
- **RAG Question Answering**: Generate answers with citations from retrieved chunks
- **Interactive Chat**: Multi-turn conversations with history
- **100% Reproducible**: Versioned dataset, pinned dependencies
- **Python Library**: Clean API for integration into other projects

## Architecture

### Data Flow

```
Hugging Face Dataset  →  Local Cache  →  Turbopuffer  →  Query Results
     (download)           (~20GB)       (user namespace)    (< 1s)
```

### Query Pipeline

```
User Question  →  Embedding (OpenAI)  →  Vector Search (Turbopuffer)
                                              ↓
    Answer  ←  LLM Generation (OpenAI)  ←  Top-K Chunks
```

## Project Structure

```
public-company-rag/
├── src/
│   └── public_company_rag/
│       ├── __init__.py           # Package API
│       ├── config.py             # Configuration
│       ├── query.py              # RAG query interface
│       │
│       └── data/
│           ├── loader.py         # Download from Hugging Face
│           └── uploader.py       # Upload to Turbopuffer
│
├── scripts/
│   ├── setup_data.py             # User: One-time data setup
│   ├── chat_rag.py               # User: Interactive chat
│   └── test_rag_query.py         # User: Test queries
│
├── pyproject.toml                # Python packaging
├── environment.yml               # Conda environment
├── .env.example                  # API key template
└── README.md                     # This file
```

## Configuration

Default settings (override via environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `TURBOPUFFER_REGION` | `api` | Turbopuffer region |
| `TURBOPUFFER_NAMESPACE` | `public_company_10k_chunks` | Namespace for chunks |
| `DEFAULT_TOP_K` | `10` | Chunks to retrieve per query |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `EMBEDDING_DIMENSION` | `1536` | Embedding dimensions |
| `LLM_MODEL` | `gpt-4o` | LLM for answer generation |

See [.env.example](.env.example) for full configuration options.

## Performance

- **Data setup**: 40-70 minutes (download + upload, one-time)
- **Query latency**: < 1s for semantic search
- **Answer generation**: < 3s total (search + LLM)
- **Cost per query**: ~$0.01-0.05 (OpenAI embeddings + LLM)

## Reproducibility

This project is designed for 100% reproducible research:

### What IS Reproducible

✓ **Dataset**: Immutable snapshot on Hugging Face with version tagging
✓ **Queries**: Same query text → same chunks retrieved (deterministic)
✓ **Answers**: Same chunks + temperature=0 → consistent LLM responses
✓ **Dependencies**: Pinned versions in `pyproject.toml` and `environment.yml`

### What to Document

When using in research papers, include:

```
Dataset: public-company-10k-chunks v1.0.0
Embeddings: text-embedding-3-small (OpenAI, 2024-01-25)
LLM: gpt-4o (OpenAI, temperature=0.0)
Turbopuffer: v1.1.0
Distance Metric: cosine_distance
```

## Comparison with GraphRAG

This package provides **pure vector-based RAG**. For comparison with **graph-enhanced retrieval (GraphRAG)**, see the companion project:

- [public-company-graph](https://github.com/alexwoolford/public-company-graph): Graph database with Neo4j + GraphRAG

Both systems use the same underlying chunks and embeddings, enabling objective comparison of RAG vs. GraphRAG approaches.

## Troubleshooting

### Data Setup Issues

**"Failed to connect to Turbopuffer"**
- Verify `TURBOPUFFER_API_KEY` in `.env` file
- Check your Turbopuffer account is active
- Ensure `TURBOPUFFER_REGION=api` (not us-east-1)

**"Dataset download timeout"**
- Check internet connection
- Try with different cache directory: `--cache-dir /tmp/data`
- Download may take 10-20 minutes for first time

**"Upload failed"**
- Check Turbopuffer account quota and billing
- Verify API key permissions
- Try smaller batch size: `--batch-size 500`

### Query Issues

**"No relevant information found"**
- Verify Turbopuffer namespace is populated: Check namespace in Turbopuffer dashboard
- Try broader query terms
- Increase `top_k` parameter

**"OpenAI API error"**
- Verify `OPENAI_API_KEY` in `.env` file
- Check API quota and billing
- Ensure API key has access to embedding and chat models

### Performance Issues

**Queries are slow (> 5s)**
- Check Turbopuffer account status
- Reduce `top_k` parameter (default: 10)
- Check network latency to Turbopuffer API

## Development

### Setup Development Environment

```bash
# Clone repo
git clone https://github.com/alexwoolford/public-company-rag.git
cd public-company-rag

# Install with dev dependencies
conda env create -f environment.yml
conda activate public-company-rag
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
ruff check .
black .
```

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file

## Acknowledgments

- Dataset derived from SEC EDGAR filings (public domain)
- Embeddings powered by OpenAI text-embedding-3-small
- Vector search powered by Turbopuffer
- Inspired by the GraphRAG architecture from the companion project

## Contact

- **Issues**: https://github.com/alexwoolford/public-company-rag/issues
- **Discussions**: https://github.com/alexwoolford/public-company-rag/discussions
- **Dataset**: https://huggingface.co/datasets/alexwoolford/public-company-10k-chunks

---

**Note**: This project is for research and educational purposes. Always verify financial information from official SEC filings.
