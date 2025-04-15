#!/usr/bin/env python
"""
Script to pre-compute the vector store for the IPL RAG application.
This script should be run before starting the application to ensure
the vector store is properly populated with all necessary data.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import settings
from src.data_processing.precompute_pipeline import run_precompute_pipeline
from src.rag.vector_store import VectorStore
from src.utils.logger import logger


def precompute_vector_store(data_dir=None):
    """
    Pre-compute the vector store by running the pre-computation pipeline
    and then populating the vector store with the processed data.

    Args:
        data_dir: Optional path to the data directory. If None, uses the default
                 from settings.
    """
    logger.info("Starting vector store pre-computation...")

    # Use the provided data directory or the default from settings
    data_dir = data_dir or settings.data_dir
    logger.info(f"Using data directory: {data_dir}")

    # Step 1: Run the pre-computation pipeline to generate all necessary data
    # logger.info("Running pre-computation pipeline...")
    # run_precompute_pipeline(data_dir)

    # Step 2: Create and populate the vector store
    logger.info("Creating vector store...")
    vector_store = VectorStore()

    # Step 3: Explicitly populate the collections
    logger.info("Populating matches collection...")
    vector_store._populate_matches_collection()

    logger.info("Populating players collection...")
    vector_store._populate_players_collection()

    logger.info("Populating analysis collection...")
    vector_store._populate_venue_collection()

    logger.info("Vector store pre-computation completed successfully")


if __name__ == "__main__":
    import argparse

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
