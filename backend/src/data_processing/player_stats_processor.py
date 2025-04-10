import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config.settings import settings
from ..utils.logger import logger
from .data_loader import DataLoader


class PlayerStatsProcessor:
    """Processes and pre-computes comprehensive player statistics."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the player stats processor."""
        self.data_dir = data_dir or settings.data_dir
        self.processed_data_dir = settings.processed_data_dir
        self.processed_data_dir.mkdir(exist_ok=True)

        # Initialize data loader
        self.data_loader = DataLoader(self.data_dir)

        # Create player stats directory
        self.player_stats_dir = self.processed_data_dir / "player_stats"
        self.player_stats_dir.mkdir(exist_ok=True)

        logger.info("PlayerStatsProcessor initialized")

    def process_all_player_stats(self) -> None:
        """Process statistics for all players."""
        # Get unique players from squads data
        squads_df = self.data_loader.load_squads()
        if squads_df.empty:
            logger.error("No squad data available")
            return

        unique_players = squads_df["Delivery Name"].unique()
        logger.info(f"Processing stats for {len(unique_players)} players")

        for player in unique_players:
            self.process_player_stats(player)

    def process_player_stats(self, player_name: str) -> None:
        """Process and save statistics for a specific player."""
        logger.info(f"Processing stats for player: {player_name}")

        # Get match IDs where the player participated
        match_ids = self.data_loader.get_match_ids_for_player(player_name)
        if not match_ids:
            logger.warning(f"No matches found for player {player_name}")
            return

        # Load all deliveries data for the player's matches
        deliveries_df = self.data_loader.load_deliveries_for_matches(match_ids)
        if deliveries_df.empty:
            logger.warning(f"No deliveries data found for player {player_name}")
            return

        # Get player's team
        squads_df = self.data_loader.load_squads()
        player_team = squads_df[squads_df["Delivery Name"] == player_name]["team"].iloc[
            0
        ]

        # Calculate statistics
        stats = {
            "player_name": player_name,
            "team": player_team,
            "batting": self._calculate_batting_stats(deliveries_df, player_name),
            "bowling": self._calculate_bowling_stats(deliveries_df, player_name),
            "recent_form": self._calculate_recent_form(deliveries_df, player_name),
            "venue_performance": self._calculate_venue_performance(
                deliveries_df, player_name
            ),
        }

        # Save statistics
        self._save_player_stats(player_name, stats)
        logger.info(f"Saved stats for player {player_name}")

    def _calculate_batting_stats(
        self, deliveries_df: pd.DataFrame, player_name: str
    ) -> Dict:
        """Calculate comprehensive batting statistics."""
        # Filter deliveries where player batted
        batting_data = deliveries_df[deliveries_df["batter"] == player_name]

        if batting_data.empty:
            return {}

        stats = {
            "matches_played": len(batting_data["match_id"].unique()),
            "innings": len(batting_data["match_id"].unique()),
            "runs_scored": batting_data["batsman_runs"].sum(),
            "balls_faced": len(batting_data),
            "fours": len(batting_data[batting_data["batsman_runs"] == 4]),
            "sixes": len(batting_data[batting_data["batsman_runs"] == 6]),
            "dismissals": len(
                batting_data[batting_data["player_dismissed"] == player_name]
            ),
        }

        # Calculate derived statistics
        if stats["balls_faced"] > 0:
            stats["strike_rate"] = (stats["runs_scored"] * 100) / stats["balls_faced"]
        if stats["dismissals"] > 0:
            stats["average"] = stats["runs_scored"] / stats["dismissals"]

        return stats

    def _calculate_bowling_stats(
        self, deliveries_df: pd.DataFrame, player_name: str
    ) -> Dict:
        """Calculate comprehensive bowling statistics."""
        # Filter deliveries where player bowled
        bowling_data = deliveries_df[deliveries_df["bowler"] == player_name]

        if bowling_data.empty:
            return {}

        stats = {
            "matches_played": len(bowling_data["match_id"].unique()),
            "innings": len(bowling_data["match_id"].unique()),
            "overs_bowled": len(bowling_data) / 6,
            "runs_conceded": bowling_data["total_runs"].sum(),
            "wickets": len(bowling_data[bowling_data["is_wicket"] == 1]),
            "maidens": len(bowling_data[bowling_data["total_runs"] == 0]) / 6,
        }

        # Calculate derived statistics
        if stats["overs_bowled"] > 0:
            stats["economy"] = stats["runs_conceded"] / stats["overs_bowled"]
        if stats["wickets"] > 0:
            stats["average"] = stats["runs_conceded"] / stats["wickets"]
            stats["strike_rate"] = (stats["overs_bowled"] * 6) / stats["wickets"]

        return stats

    def _calculate_recent_form(
        self, deliveries_df: pd.DataFrame, player_name: str
    ) -> Dict:
        """Calculate player's recent form (last 5 matches)."""
        # Sort by match_id to get most recent matches
        recent_matches = sorted(deliveries_df["match_id"].unique())[-5:]
        recent_data = deliveries_df[deliveries_df["match_id"].isin(recent_matches)]

        return {
            "batting": self._calculate_batting_stats(recent_data, player_name),
            "bowling": self._calculate_bowling_stats(recent_data, player_name),
        }

    def _calculate_venue_performance(
        self, deliveries_df: pd.DataFrame, player_name: str
    ) -> Dict:
        """Calculate player's performance at different venues."""
        venue_stats = {}

        # Get unique match IDs from the deliveries data
        match_ids = deliveries_df["match_id"].unique()

        # Group deliveries by match ID
        for match_id in match_ids:
            # Get venue for this match from matches.csv
            venue = self.data_loader.get_venue_for_match(match_id)
            if not venue:
                continue

            # Get deliveries for this match
            match_deliveries = deliveries_df[deliveries_df["match_id"] == match_id]

            # Initialize venue stats if not already done
            if venue not in venue_stats:
                venue_stats[venue] = {
                    "batting": {
                        "matches_played": 0,
                        "innings": 0,
                        "runs_scored": 0,
                        "balls_faced": 0,
                        "fours": 0,
                        "sixes": 0,
                        "dismissals": 0,
                    },
                    "bowling": {
                        "matches_played": 0,
                        "innings": 0,
                        "overs_bowled": 0,
                        "runs_conceded": 0,
                        "wickets": 0,
                        "maidens": 0,
                    },
                }

            # Calculate batting stats for this match
            batting_data = match_deliveries[match_deliveries["batter"] == player_name]
            if not batting_data.empty:
                venue_stats[venue]["batting"]["matches_played"] += 1
                venue_stats[venue]["batting"]["innings"] += 1
                venue_stats[venue]["batting"]["runs_scored"] += batting_data[
                    "batsman_runs"
                ].sum()
                venue_stats[venue]["batting"]["balls_faced"] += len(batting_data)
                venue_stats[venue]["batting"]["fours"] += len(
                    batting_data[batting_data["batsman_runs"] == 4]
                )
                venue_stats[venue]["batting"]["sixes"] += len(
                    batting_data[batting_data["batsman_runs"] == 6]
                )
                venue_stats[venue]["batting"]["dismissals"] += len(
                    batting_data[batting_data["player_dismissed"] == player_name]
                )

            # Calculate bowling stats for this match
            bowling_data = match_deliveries[match_deliveries["bowler"] == player_name]
            if not bowling_data.empty:
                venue_stats[venue]["bowling"]["matches_played"] += 1
                venue_stats[venue]["bowling"]["innings"] += 1
                venue_stats[venue]["bowling"]["overs_bowled"] += len(bowling_data) / 6
                venue_stats[venue]["bowling"]["runs_conceded"] += bowling_data[
                    "total_runs"
                ].sum()
                venue_stats[venue]["bowling"]["wickets"] += len(
                    bowling_data[bowling_data["is_wicket"] == 1]
                )
                venue_stats[venue]["bowling"]["maidens"] += (
                    len(bowling_data[bowling_data["total_runs"] == 0]) / 6
                )

        # Calculate derived statistics for each venue
        for venue in venue_stats:
            # Batting derived stats
            if venue_stats[venue]["batting"]["balls_faced"] > 0:
                venue_stats[venue]["batting"]["strike_rate"] = (
                    venue_stats[venue]["batting"]["runs_scored"] * 100
                ) / venue_stats[venue]["batting"]["balls_faced"]
            if venue_stats[venue]["batting"]["dismissals"] > 0:
                venue_stats[venue]["batting"]["average"] = (
                    venue_stats[venue]["batting"]["runs_scored"]
                ) / venue_stats[venue]["batting"]["dismissals"]

            # Bowling derived stats
            if venue_stats[venue]["bowling"]["overs_bowled"] > 0:
                venue_stats[venue]["bowling"]["economy"] = (
                    venue_stats[venue]["bowling"]["runs_conceded"]
                ) / venue_stats[venue]["bowling"]["overs_bowled"]
            if venue_stats[venue]["bowling"]["wickets"] > 0:
                venue_stats[venue]["bowling"]["average"] = (
                    venue_stats[venue]["bowling"]["runs_conceded"]
                ) / venue_stats[venue]["bowling"]["wickets"]
                venue_stats[venue]["bowling"]["strike_rate"] = (
                    venue_stats[venue]["bowling"]["overs_bowled"] * 6
                ) / venue_stats[venue]["bowling"]["wickets"]

        return venue_stats

    def _save_player_stats(self, player_name: str, stats: Dict) -> None:
        """Save player statistics to a JSON file."""
        stats_file = self.player_stats_dir / f"{player_name}.json"

        try:
            # Convert NumPy types to Python native types
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {
                        key: convert_numpy_types(value) for key, value in obj.items()
                    }
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj

            # Convert all NumPy types in the stats dictionary
            converted_stats = convert_numpy_types(stats)

            with open(stats_file, "w") as f:
                json.dump(converted_stats, f, indent=2)
            logger.info(f"Saved player stats to {stats_file}")
        except Exception as e:
            logger.error(f"Error saving player stats: {e}")

    def get_player_stats(self, player_name: str) -> Dict:
        """Retrieve pre-computed statistics for a player."""
        stats_file = self.player_stats_dir / f"{player_name}.json"
        print(stats_file)

        try:
            if not stats_file.exists():
                logger.warning(f"No stats file found for player {player_name}")
                return {}

            with open(stats_file, "r") as f:
                stats = json.load(f)
            return stats
        except Exception as e:
            logger.error(f"Error loading player stats: {e}")
            return {}
