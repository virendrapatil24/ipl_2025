import argparse
from pathlib import Path
from typing import Optional

from ..utils.logger import logger
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineering
from .player_stats_processor import PlayerStatsProcessor


def run_precompute_pipeline(data_dir: Optional[Path] = None) -> None:
    """Run the pre-computation pipeline for data processing tasks."""
    logger.info("Starting pre-computation pipeline...")

    # Initialize components
    data_loader = DataLoader(data_dir)
    feature_engineering = FeatureEngineering(data_loader)
    player_stats_processor = PlayerStatsProcessor(data_dir)

    # Load data
    matches_df = data_loader.load_matches()
    if matches_df.empty:
        logger.error("No matches data available")
        return

    squads_df = data_loader.load_squads()
    if squads_df.empty:
        logger.error("No squads data available")
        return

    # Process player statistics
    logger.info("Processing player statistics...")
    player_stats_processor.process_all_player_stats()

    # Process team statistics
    logger.info("Processing team statistics...")
    teams = squads_df["team"].unique()

    for team in teams:
        # Get match IDs for the team
        match_ids = data_loader.get_match_ids_for_team(team)
        if not match_ids:
            logger.warning(f"No matches found for team {team}")
            continue

        # Load deliveries data for the team's matches
        deliveries_df = data_loader.load_deliveries_for_matches(match_ids)
        if deliveries_df.empty:
            logger.warning(f"No deliveries data found for team {team}")
            continue

        # Calculate venue statistics
        logger.info(f"Calculating venue statistics for team {team}...")
        feature_engineering.calculate_venue_statistics(team, deliveries_df)

        # Calculate head-to-head statistics
        logger.info(f"Calculating head-to-head statistics for team {team}...")
        for opponent in teams:
            if opponent != team:
                feature_engineering.calculate_head_to_head_statistics(
                    team, opponent, deliveries_df
                )

    logger.info("Pre-computation pipeline completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pre-computation pipeline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing the data files",
        default=None,
    )
    args = parser.parse_args()

    print("pipeline started")
    run_precompute_pipeline(args.data_dir)
