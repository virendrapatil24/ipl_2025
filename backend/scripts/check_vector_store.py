#!/usr/bin/env python
"""
Script to check if the vector store is properly populated.
This script can be used to verify that the vector store has been
pre-computed correctly and is ready for use.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.vector_store import VectorStore
from src.utils.logger import logger


def check_vector_store():
    """Check if the vector store is properly populated."""
    print("Checking vector store...")

    # Initialize the vector store
    vector_store = VectorStore()

    # Check if collections have data
    matches_count = vector_store.matches_collection.count()
    players_count = vector_store.players_collection.count()
    analysis_count = vector_store.analysis_collection.count()

    print(f"Vector store status:")
    print(f"  - Matches collection: {matches_count} records")
    print(f"  - Players collection: {players_count} records")
    print(f"  - Analysis collection: {analysis_count} records")

    # Check if all collections have data
    if matches_count > 0 and players_count > 0 and analysis_count > 0:
        print("Vector store is properly populated and ready for use.")
    else:
        print(
            "Vector store is not properly populated. Please run the pre-computation script."
        )


if __name__ == "__main__":
    check_vector_store()
