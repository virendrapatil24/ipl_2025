#!/usr/bin/env python3
"""
Script to run the player analysis processor.
This will generate four types of player analysis:
1. Player all-time stats
2. Player at venue stats
3. Player vs team stats
4. Player vs player stats
"""

import argparse
from pathlib import Path

from src.data_processing.player_analysis_processor import PlayerAnalysisProcessor
from src.utils.logger import logger


def main():
    """Run the player analysis processor."""
    parser = argparse.ArgumentParser(description="Run the player analysis processor")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing the data files",
        default=None,
    )
    args = parser.parse_args()

    logger.info("Starting player analysis processor...")
    processor = PlayerAnalysisProcessor(args.data_dir)
    processor.process_all_player_analysis()
    logger.info("Player analysis processor completed successfully")


if __name__ == "__main__":
    main()
