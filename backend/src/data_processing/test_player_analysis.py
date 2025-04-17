#!/usr/bin/env python3
"""
Test script for the player analysis processor.
"""

from pathlib import Path

from ..config.settings import settings
from ..utils.logger import logger
from .player_analysis_processor import PlayerAnalysisProcessor


def main():
    """Run a test of the player analysis processor."""
    logger.info("Starting player analysis test...")

    # Initialize the processor
    processor = PlayerAnalysisProcessor()

    # Get a list of players from the squad data
    squads_df = processor.data_loader.load_squads()
    if squads_df.empty:
        logger.error("No squad data available")
        return

    # Get the first player for testing
    test_player = squads_df["Delivery Name"].iloc[0]
    logger.info(f"Testing with player: {test_player}")

    # Process all four types of analysis for this player
    processor.process_player_all_time_stats(test_player)
    processor.process_player_venue_stats(test_player)
    processor.process_player_vs_team_stats(test_player)
    processor.process_player_vs_player_stats(test_player)

    # Retrieve and print the results
    all_time_stats = processor.get_player_all_time_stats(test_player)
    logger.info(f"All-time stats for {test_player}: {all_time_stats}")

    venue_stats = processor.get_player_venue_stats(test_player)
    logger.info(f"Venue stats for {test_player}: {venue_stats}")

    vs_team_stats = processor.get_player_vs_team_stats(test_player)
    logger.info(f"Vs team stats for {test_player}: {vs_team_stats}")

    vs_player_stats = processor.get_player_vs_player_stats(test_player)
    logger.info(f"Vs player stats for {test_player}: {vs_player_stats}")

    logger.info("Player analysis test completed successfully")


if __name__ == "__main__":
    main()
