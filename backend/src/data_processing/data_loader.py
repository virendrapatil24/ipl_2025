from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from ..config.settings import settings
from ..utils.logger import logger


class DataLoader:
    """Loads and manages access to various data files."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the data loader with the data directory."""
        self.data_dir = data_dir or settings.data_dir
        self.processed_data_dir = settings.processed_data_dir
        self.processed_data_dir.mkdir(exist_ok=True)
        self.cleaned_data_dir = settings.cleaned_data_dir

        # Cache for loaded data
        self._matches_df = None
        self._squads_df = None
        self._deliveries_cache = {}
        self._team_map = None

        logger.info(f"DataLoader initialized with data directory: {self.data_dir}")

    def load_team_map(self) -> Dict:
        """Load team name mapping from JSON file."""
        if self._team_map is not None:
            return self._team_map

        try:
            team_map_file = self.data_dir / "cleaned_data" / "team_map.json"
            if not team_map_file.exists():
                logger.error(f"Team map file not found: {team_map_file}")
                return {}

            import json

            with open(team_map_file, "r") as f:
                self._team_map = json.load(f)
            logger.info(f"Loaded team map with {len(self._team_map)} entries")
            return self._team_map
        except Exception as e:
            logger.error(f"Error loading team map: {e}")
            return {}

    def standardize_team_name(self, team_name: str) -> str:
        """Standardize team name using the team mapping."""
        team_map = self.load_team_map()
        return team_map.get(team_name, team_name)

    def load_matches(self) -> pd.DataFrame:
        """Load matches data."""
        if self._matches_df is not None:
            return self._matches_df

        try:
            matches_file = (
                self.data_dir / "cleaned_data" / "matches_data" / "matches.csv"
            )

            if not matches_file.exists():
                logger.error(f"Matches file not found: {matches_file}")
                return pd.DataFrame()

            self._matches_df = pd.read_csv(matches_file)
            print("self._matches_df", self._matches_df)
            logger.info(f"Loaded {len(self._matches_df)} matches")
            return self._matches_df
        except Exception as e:
            logger.error(f"Error loading matches data: {e}")
            return pd.DataFrame()

    def load_squads(self) -> pd.DataFrame:
        """Load squads data from all team-specific squad files."""
        if self._squads_df is not None:
            return self._squads_df

        try:
            # Look for all team-specific squad files
            squad_files = list(
                (
                    self.data_dir / "cleaned_data" / "squads_per_season_data" / "2025"
                ).glob("*.csv")
            )

            if not squad_files:
                logger.error("No squad files found in data directory")
                return pd.DataFrame()

            # Load and combine all squad files
            all_squads = []
            print("squad_files", squad_files)
            for squad_file in squad_files:
                team_name = squad_file.stem.replace("_squad", "")
                squad_df = pd.read_csv(squad_file)
                # Add team column if it doesn't exist
                if "team" not in squad_df.columns:
                    squad_df["team"] = team_name
                all_squads.append(squad_df)

            if not all_squads:
                logger.error("No valid squad data found in any squad files")
                return pd.DataFrame()

            # Combine all squad data
            self._squads_df = pd.concat(all_squads, ignore_index=True)
            logger.info(
                f"Loaded {len(self._squads_df)} squad entries from {len(squad_files)} files"
            )
            return self._squads_df
        except Exception as e:
            logger.error(f"Error loading squads data: {e}")
            return pd.DataFrame()

    def load_squad_data(self, team: str) -> pd.DataFrame:
        """Load squad data for a specific team."""
        try:
            # Standardize team name
            standardized_team = self.standardize_team_name(team)

            # First try loading from the team-specific squad file
            squad_file = self.data_dir / f"{standardized_team}_squad.csv"
            if squad_file.exists():
                squad_df = pd.read_csv(squad_file)
                logger.info(
                    f"Loaded squad data for team {standardized_team}: {len(squad_df)} players"
                )
                return squad_df

            # Fallback to the old method if team-specific file doesn't exist
            squads_df = self.load_squads()
            if squads_df.empty:
                return pd.DataFrame()

            team_squad = squads_df[squads_df["team"] == standardized_team]
            logger.info(
                f"Loaded squad data for team {standardized_team}: {len(team_squad)} players"
            )
            return team_squad
        except Exception as e:
            logger.error(f"Error loading squad data for team {team}: {e}")
            return pd.DataFrame()

    def load_deliveries(self, match_id: str) -> pd.DataFrame:
        """Load deliveries data for a specific match."""
        # Check cache first
        if match_id in self._deliveries_cache:
            return self._deliveries_cache[match_id]

        try:
            # Look for the deliveries file in the deliveries directory
            deliveries_dir = (
                self.data_dir / "cleaned_data" / "deliveries_per_match_data"
            )
            if not deliveries_dir.exists():
                # logger.error(f"Deliveries directory not found: {deliveries_dir}")
                return pd.DataFrame()

            deliveries_file = deliveries_dir / f"{match_id}.csv"
            if not deliveries_file.exists():
                logger.warning(
                    f"Deliveries file not found for match {match_id}: {deliveries_file}"
                )
                return pd.DataFrame()

            deliveries_df = pd.read_csv(deliveries_file)
            # logger.info(
            #     f"Loaded deliveries data for match {match_id}: {len(deliveries_df)} records"
            # )

            # Cache the result
            self._deliveries_cache[match_id] = deliveries_df
            return deliveries_df
        except Exception as e:
            logger.error(f"Error loading deliveries data for match {match_id}: {e}")
            return pd.DataFrame()

    def load_deliveries_for_matches(self, match_ids: List[str]) -> pd.DataFrame:
        """Load deliveries data for multiple matches and combine them."""
        all_deliveries = []

        for match_id in match_ids:
            deliveries_df = self.load_deliveries(match_id)
            if not deliveries_df.empty:
                all_deliveries.append(deliveries_df)

        if not all_deliveries:
            logger.warning("No deliveries data found for any of the specified matches")
            return pd.DataFrame()

        combined_deliveries = pd.concat(all_deliveries, ignore_index=True)
        logger.info(
            f"Combined deliveries data for {len(match_ids)} matches: {len(combined_deliveries)} records"
        )
        return combined_deliveries

    def get_match_ids_for_player(self, player_name: str) -> List[str]:
        """Get match IDs where a specific player participated by checking all deliveries data."""
        logger.info(f"Finding matches for player {player_name}")

        # Get all match IDs
        matches_df = self.load_matches()
        if matches_df.empty:
            logger.warning("No matches data available")
            return []

        all_match_ids = matches_df["match_id"].tolist()
        player_matches = set()

        # Check each match's deliveries for the player
        for match_id in all_match_ids:
            deliveries_df = self.load_deliveries(match_id)
            if deliveries_df.empty:
                continue

            # Check if player appears in batting or bowling
            # Use the correct column names from the deliveries data
            if (
                player_name in deliveries_df["batter"].values
                or player_name in deliveries_df["bowler"].values
            ):
                player_matches.add(match_id)

        logger.info(f"Found {len(player_matches)} matches for player {player_name}")
        return list(player_matches)

    def get_venue_for_match(self, match_id: str) -> str:
        """Get the venue for a specific match from matches.csv."""
        matches_df = self.load_matches()
        if matches_df.empty:
            logger.warning("No matches data available")
            return ""

        match_row = matches_df[matches_df["match_id"] == match_id]
        if match_row.empty:
            logger.warning(f"No match found with ID {match_id}")
            return ""

        venue = match_row["venue"].iloc[0]
        return venue

    def get_match_ids_for_team(self, team: str) -> List[str]:
        """Get match IDs where a specific team played."""
        matches_df = self.load_matches()
        if matches_df.empty:
            return []

        # Standardize team name
        standardized_team = self.standardize_team_name(team)

        team_matches = matches_df[
            (matches_df["team1"] == standardized_team.replace("_", " "))
            | (matches_df["team2"] == standardized_team.replace("_", " "))
        ]

        match_ids = team_matches["match_id"].tolist()
        logger.info(f"Found {len(match_ids)} matches for team {standardized_team}")
        return match_ids

    def get_match_ids_for_venue(self, venue: str) -> List[str]:
        """Get match IDs played at a specific venue."""
        matches_df = self.load_matches()
        if matches_df.empty:
            return []

        venue_matches = matches_df[matches_df["venue"] == venue]

        match_ids = venue_matches["match_id"].tolist()
        logger.info(f"Found {len(match_ids)} matches at venue {venue}")
        return match_ids

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
                self.processed_data_dir
                / f"match_analysis/{analysis_types[analysis_type]}"
            )
            return pd.read_json(path).to_dict()
        except Exception as e:
            logger.error(
                f"Error loading match analysis data for type " f"{analysis_type}: {e}"
            )
            raise

    def get_available_teams(self) -> List[str]:
        """Get list of available teams from squad data."""
        try:
            squads = "squads_per_season_data/2025"
            squad_path = self.processed_data_dir / squads
            return [f.stem for f in squad_path.glob("*.csv")]
        except Exception as e:
            logger.error(f"Error getting available teams: {e}")
            raise

    def get_venue_list(self) -> List[str]:
        """Get list of all venues from matches data."""
        try:
            matches_df = self.load_matches()
            return matches_df["venue"].unique().tolist()
        except Exception as e:
            logger.error(f"Error getting venue list: {e}")
            raise
