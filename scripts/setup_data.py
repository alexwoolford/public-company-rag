#!/usr/bin/env python3
"""
One-time data setup: Download dataset and upload to Turbopuffer.

This script automates the setup process for end users:
1. Downloads the pre-exported dataset from Hugging Face (~20GB)
2. Uploads chunks to the user's Turbopuffer namespace (~30-60 minutes)
3. Verifies the upload was successful

Usage:
    python scripts/setup_data.py
    python scripts/setup_data.py --skip-download  # If already cached
    python scripts/setup_data.py --skip-upload    # Download only
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from turbopuffer import Turbopuffer

from public_company_rag.config import (
    get_settings,
    get_turbopuffer_api_key,
    get_turbopuffer_region,
    get_turbopuffer_namespace,
)
from public_company_rag.data.loader import download_dataset, get_dataset_info
from public_company_rag.data.uploader import upload_to_turbopuffer, verify_upload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Setup public-company-rag: Download dataset and upload to Turbopuffer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full setup (download + upload)
  python scripts/setup_data.py

  # Skip download if already cached
  python scripts/setup_data.py --skip-download

  # Download only (no upload)
  python scripts/setup_data.py --skip-upload

  # Use custom cache directory
  python scripts/setup_data.py --cache-dir /path/to/cache

Requirements:
  - TURBOPUFFER_API_KEY set in .env file
  - OPENAI_API_KEY set in .env file (for querying later)
  - Internet connection for downloading (~20GB)
  - Turbopuffer account (stores 2.85M vectors)

Estimated time: 40-70 minutes total
  - Download: 10-20 minutes (depends on connection speed)
  - Upload: 30-50 minutes (batch upload to Turbopuffer)
        """,
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if dataset is already cached locally",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip Turbopuffer upload (download only)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./data",
        help="Local cache directory for dataset (default: ./data)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Upload batch size (default: 1000 chunks)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="alexwoolford/public-company-10k-chunks",
        help="Hugging Face dataset name (default: alexwoolford/public-company-10k-chunks)",
    )

    args = parser.parse_args()

    # Print banner
    print("=" * 70)
    print("Public Company RAG - Data Setup")
    print("=" * 70)
    print()

    # Validate configuration
    try:
        settings = get_settings()
        logger.info("✓ Configuration loaded successfully")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please check your .env file (see .env.example for template)")
        return 1

    dataset = None

    # Step 1: Download dataset
    if not args.skip_download:
        print("\n" + "-" * 70)
        print("Step 1/3: Downloading dataset from Hugging Face")
        print("-" * 70)
        print(f"Dataset: {args.dataset_name}")
        print(f"Cache: {args.cache_dir}")
        print("Size: ~20GB (first download only, cached afterward)")
        print("Time: ~10-20 minutes (depends on connection speed)")
        print()

        try:
            dataset = download_dataset(
                dataset_name=args.dataset_name,
                cache_dir=args.cache_dir,
            )

            # Show dataset info
            info = get_dataset_info(dataset)
            logger.info(f"✓ Download complete")
            logger.info(f"  Total chunks: {info['total_chunks']:,}")
            logger.info(f"  Embedding dimension: {info.get('embedding_dimension', 'unknown')}")
            if "unique_companies" in info:
                logger.info(f"  Unique companies: {info['unique_companies']:,}")

        except Exception as e:
            logger.error(f"Download failed: {e}", exc_info=True)
            logger.error("\nTroubleshooting:")
            logger.error("1. Check internet connection")
            logger.error("2. Verify Hugging Face dataset exists: " + args.dataset_name)
            logger.error("3. Try with different cache directory")
            return 1

    else:
        logger.info("Skipping download (--skip-download specified)")

        # Load from cache
        try:
            dataset = download_dataset(
                dataset_name=args.dataset_name,
                cache_dir=args.cache_dir,
            )
            logger.info(f"✓ Loaded from cache: {len(dataset):,} chunks")
        except Exception as e:
            logger.error(f"Failed to load from cache: {e}")
            logger.error("Remove --skip-download to download fresh copy")
            return 1

    # Step 2: Upload to Turbopuffer
    if not args.skip_upload:
        if dataset is None:
            logger.error("No dataset available for upload")
            return 1

        print("\n" + "-" * 70)
        print("Step 2/3: Uploading to Turbopuffer")
        print("-" * 70)

        try:
            # Get Turbopuffer config
            turbopuffer_api_key = get_turbopuffer_api_key()
            turbopuffer_region = get_turbopuffer_region()
            namespace_name = get_turbopuffer_namespace()

            logger.info(f"Region: {turbopuffer_region}")
            logger.info(f"Namespace: {namespace_name}")
            logger.info(f"Batch size: {args.batch_size}")
            print()

            # Create Turbopuffer client
            client = Turbopuffer(
                api_key=turbopuffer_api_key,
                region=turbopuffer_region,
            )

            # Upload
            stats = upload_to_turbopuffer(
                dataset=dataset,
                client=client,
                namespace_name=namespace_name,
                batch_size=args.batch_size,
            )

            if stats["total_uploaded"] == 0:
                logger.error("No chunks were uploaded!")
                return 1

            logger.info(f"✓ Upload complete: {stats['total_uploaded']:,} chunks")

            if stats.get("failed_batches"):
                logger.warning(f"⚠ {len(stats['failed_batches'])} batches failed")
                logger.warning("You may need to re-run upload for failed batches")

        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            logger.error("Check TURBOPUFFER_API_KEY in .env file")
            return 1
        except Exception as e:
            logger.error(f"Upload failed: {e}", exc_info=True)
            logger.error("\nTroubleshooting:")
            logger.error("1. Verify TURBOPUFFER_API_KEY is valid")
            logger.error("2. Check Turbopuffer account has sufficient quota")
            logger.error("3. Check internet connection")
            return 1

    else:
        logger.info("Skipping upload (--skip-upload specified)")

    # Step 3: Verify
    if not args.skip_upload:
        print("\n" + "-" * 70)
        print("Step 3/3: Verifying upload")
        print("-" * 70)

        try:
            client = Turbopuffer(
                api_key=get_turbopuffer_api_key(),
                region=get_turbopuffer_region(),
            )
            namespace_name = get_turbopuffer_namespace()

            result = verify_upload(
                client=client,
                namespace_name=namespace_name,
                expected_count=len(dataset) if dataset else 0,
            )

            if result["verified"]:
                logger.info("✓ Verification successful!")
                logger.info(f"  Namespace: {namespace_name}")
                logger.info(f"  Chunks: {result['actual_count']:,}")
            else:
                logger.warning(f"⚠ Verification issue: {result.get('error', 'Count mismatch')}")
                if "expected_count" in result:
                    logger.warning(f"  Expected: {result['expected_count']:,}")
                    logger.warning(f"  Actual: {result.get('actual_count', 0):,}")

        except Exception as e:
            logger.warning(f"Verification failed: {e}")
            logger.warning("Upload may have succeeded despite verification error")

    # Success message
    print("\n" + "=" * 70)
    print("Setup Complete!")
    print("=" * 70)
    print()
    print("You can now use the RAG system:")
    print()
    print("  # Interactive chat")
    print("  python scripts/chat_rag.py")
    print()
    print("  # Test queries")
    print("  python scripts/test_rag_query.py")
    print()
    print("  # Python API")
    print("  from public_company_rag import create_client, get_namespace_name, semantic_search")
    print("  client = create_client()")
    print("  namespace = get_namespace_name()")
    print('  results = semantic_search(client, namespace, "What are the main risks for tech companies?")')
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
