from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..config.settings import settings
from ..utils.logger import logger


class DataLoader:
    def __init__(self, base_path: Optional[str] = None):
        """Initialize DataLoader with base path for data files."""
        self.base_path = Path(base_path or settings.DATA_DIR)
        self.processed_data_path = self.base_path / settings.PROCESSED_DATA_DIR

    def load_matches_data(self) -> pd.DataFrame:
        """Load the matches dataset."""
        try:
            path = self.processed_data_path / "matches_data/matches.csv"
            return pd.read_csv(path)
        except Exception as e:
            logger.error(f"Error loading matches data: {e}")
            raise

    def load_deliveries_data(self, match_id: int) -> pd.DataFrame:
        """Load ball-by-ball data for a specific match."""
        try:
            deliveries = "deliveries_per_match_data"
            path = (self.processed_data_path/f"{deliveries}/{match_id}.csv")
            return pd.read_csv(path)
        except Exception as e:
            logger.error(
                f"Error loading deliveries data for match {match_id}: {e}"
            )
            raise

    def load_squad_data(self, team: str) -> pd.DataFrame:
        """Load squad data for a specific team."""
        try:
            squads = "squads_per_season_data/2025"
            path = self.processed_data_path / f"{squads}/{team}.csv"
            return pd.read_csv(path)
        except Exception as e:
            logger.error(f"Error loading squad data for team {team}: {e}")
            raise

    def load_match_analysis(self, analysis_type: str) -> Dict:
        """Load match analysis data of specific type."""
        try:
            analysis_types = {
                "bowler_vs_batter": "bowler_vs_batter_matchup.json",
                "match_situation": "match_situation_and_stragies.json",
                "pitch_condition": "pitch_condition_and_their_impact.json",
            }

            if analysis_type not in analysis_types:
                raise ValueError(
                    f"Invalid analysis type. Must be one of "
                    f"{list(analysis_types.keys())}"
                )

            path = (
                self.processed_data_path
                / f"match_analysis/{analysis_types[analysis_type]}"
            )
            return pd.read_json(path).to_dict()
        except Exception as e:
            logger.error(
                f"Error loading match analysis data for type "
                f"{analysis_type}: {e}"
            )
            raise

    def get_available_teams(self) -> List[str]:
        """Get list of available teams from squad data."""
        try:
            squads = "squads_per_season_data/2025"
            squad_path = (self.processed_data_path/squads)
            return [f.stem for f in squad_path.glob("*.csv")]
        except Exception as e:
            logger.error(f"Error getting available teams: {e}")
            raise

    def get_venue_list(self) -> List[str]:
        """Get list of all venues from matches data."""
        try:
            matches_df = self.load_matches_data()
            return matches_df["venue"].unique().tolist()
        except Exception as e:
            logger.error(f"Error getting venue list: {e}")
            raise
