import argparse
from pathlib import Path
from typing import Optional

from ..utils.logger import logger
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineering
from .player_analysis_processor import PlayerAnalysisProcessor


def run_precompute_pipeline(data_dir: Optional[Path] = None) -> None:
    """Run the pre-computation pipeline for data processing tasks."""
    logger.info("Starting pre-computation pipeline...")

    # Initialize components
    data_loader = DataLoader(data_dir)
    feature_engineering = FeatureEngineering(data_loader)
    player_analysis_processor = PlayerAnalysisProcessor(data_dir)

    # Load data
    matches_df = data_loader.load_matches()
    if matches_df.empty:
        logger.error("No matches data available")
        return

    squads_df = data_loader.load_squads()
    if squads_df.empty:
        logger.error("No squads data available")
        return

    # STEP 1
    # Process venue statistics
    feature_engineering.calculate_venue_statistics(matches_df)

    # STEP 2
    # Process team at a venue statistics
    feature_engineering.calculate_team_at_venue_statistics(matches_df)

    # STEP 3
    # Process team head-to-head statistics
    feature_engineering.calculate_team_h2h_statistics(matches_df)

    # STEP 4
    # Process player analysis (all four types)
    player_analysis_processor.process_all_player_analysis()

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

    run_precompute_pipeline(args.data_dir)
