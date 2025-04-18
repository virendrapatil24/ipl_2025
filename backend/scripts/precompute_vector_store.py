#!/usr/bin/env python
"""
Script to pre-compute the vector store for the IPL RAG application.
This script should be run before starting the application to ensure
the vector store is properly populated with all necessary data.
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the modules
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

from src.config.settings import settings
from src.rag.vector_store import VectorStore
from src.utils.logger import logger


def precompute_vector_store(data_dir=None):
    """
    Pre-compute the vector store by initializing it with the processed data.
    The vector store will automatically check and populate itself if needed.

    Args:
        data_dir: Optional path to the data directory. If None, uses the default
                 from settings.
    """
    logger.info("Starting vector store pre-computation...")

    # Use the provided data directory or the default from settings
    data_dir = data_dir or settings.data_dir
    logger.info(f"Using data directory: {data_dir}")

    # Create the vector store - it will automatically check and populate if needed
    logger.info("Initializing vector store...")
    VectorStore()  # Initialize and let it auto-populate

    logger.info("Vector store pre-computation completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute the vector store for the IPL RAG application"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing the data files",
        default=None,
    )
    args = parser.parse_args()

    precompute_vector_store(args.data_dir)
