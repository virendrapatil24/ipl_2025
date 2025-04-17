import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config.settings import settings
from ..utils.logger import logger
from .data_loader import DataLoader


class PlayerAnalysisProcessor:
    """Processes and pre-computes comprehensive player analysis in four categories:
    1. Player all-time stats
    2. Player at venue stats
    3. Player vs team stats
    4. Player vs player stats
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the player analysis processor."""
        self.data_dir = data_dir or settings.data_dir
        self.processed_data_dir = settings.processed_data_dir
        self.processed_data_dir.mkdir(exist_ok=True)

        # Initialize data loader
        self.data_loader = DataLoader(self.data_dir)

        # Create directories for different types of player analysis
        self.all_time_stats_dir = self.processed_data_dir / "player_all_time_stats"
        self.all_time_stats_dir.mkdir(exist_ok=True)

        self.venue_stats_dir = self.processed_data_dir / "player_venue_stats"
        self.venue_stats_dir.mkdir(exist_ok=True)

        self.vs_team_stats_dir = self.processed_data_dir / "player_vs_team_stats"
        self.vs_team_stats_dir.mkdir(exist_ok=True)

        self.vs_player_stats_dir = self.processed_data_dir / "player_vs_player_stats"
        self.vs_player_stats_dir.mkdir(exist_ok=True)

        logger.info("PlayerAnalysisProcessor initialized")

    def process_all_player_analysis(self) -> None:
        """Process all four types of analysis for all players in the current IPL 2025 squad."""
        # Get unique players from squads data
        squads_df = self.data_loader.load_squads()
        if squads_df.empty:
            logger.error("No squad data available")
            return

        unique_players = squads_df["Delivery Name"].unique()
        logger.info(f"Processing analysis for {len(unique_players)} players")

        # Create a mapping of player to team
        player_team_map = {}
        for _, row in squads_df.iterrows():
            player_team_map[row["Delivery Name"]] = row["team"]

        # Initialize data structures for all players
        all_time_data = {player: [] for player in unique_players}
        venue_data = {player: {} for player in unique_players}
        team_data = {player: {} for player in unique_players}
        player_data = {player: {} for player in unique_players}

        # Get all match IDs
        matches_df = self.data_loader.load_matches()
        if matches_df.empty:
            logger.error("No matches data available")
            return

        all_match_ids = matches_df["match_id"].unique()
        logger.info(f"Found {len(all_match_ids)} matches to process")

        # Process all matches in a single pass
        for match_id in all_match_ids:
            # Load deliveries for this match
            match_deliveries = self.data_loader.load_deliveries(match_id)
            if match_deliveries.empty:
                continue

            # Get venue for this match
            venue = self.data_loader.get_venue_for_match(match_id)

            # Get teams for this match
            match_row = matches_df[matches_df["match_id"] == match_id]
            if match_row.empty:
                continue

            # Process each delivery
            for _, delivery in match_deliveries.iterrows():
                # Get batter and bowler
                batter = delivery["batter"]
                bowler = delivery["bowler"]

                # Process batter if in our player list
                if batter in unique_players:
                    player = batter
                    player_team = player_team_map[player]

                    # Add to all-time data
                    all_time_data[player].append(delivery)

                    # Add to venue data
                    if venue:
                        if venue not in venue_data[player]:
                            venue_data[player][venue] = []
                        venue_data[player][venue].append(delivery)

                    # Add to team data (if player is playing against this team)
                    opponent_team = delivery["bowling_team"]
                    if opponent_team != player_team:
                        if opponent_team not in team_data[player]:
                            team_data[player][opponent_team] = []
                        team_data[player][opponent_team].append(delivery)

                    # Add to player data (only if opponent is in our player list)
                    opponent_player = bowler
                    if opponent_player in unique_players:
                        if opponent_player not in player_data[player]:
                            player_data[player][opponent_player] = []
                        player_data[player][opponent_player].append(delivery)

                # Process bowler if in our player list
                if bowler in unique_players:
                    player = bowler
                    player_team = player_team_map[player]

                    # Add to all-time data
                    all_time_data[player].append(delivery)

                    # Add to venue data
                    if venue:
                        if venue not in venue_data[player]:
                            venue_data[player][venue] = []
                        venue_data[player][venue].append(delivery)

                    # Add to team data (if player is playing against this team)
                    opponent_team = delivery["batting_team"]
                    if opponent_team != player_team:
                        if opponent_team not in team_data[player]:
                            team_data[player][opponent_team] = []
                        team_data[player][opponent_team].append(delivery)

                    # Add to player data (only if opponent is in our player list)
                    opponent_player = batter
                    if opponent_player in unique_players:
                        if opponent_player not in player_data[player]:
                            player_data[player][opponent_player] = []
                        player_data[player][opponent_player].append(delivery)

        # Process statistics for each player
        for player in unique_players:
            logger.info(f"Processing statistics for player: {player}")

            # Convert collected data to DataFrames
            all_time_df = (
                pd.DataFrame(all_time_data[player])
                if all_time_data[player]
                else pd.DataFrame()
            )

            # Process all-time stats
            if not all_time_df.empty:
                stats = {
                    "player_name": player,
                    "batting": self._calculate_batting_stats(all_time_df, player),
                    "bowling": self._calculate_bowling_stats(all_time_df, player),
                }
                self._save_player_stats(player, stats, self.all_time_stats_dir)
                logger.info(f"Saved all-time stats for player {player}")

            # Process venue stats
            venue_stats = {}
            for venue, deliveries in venue_data[player].items():
                venue_df = pd.DataFrame(deliveries)
                venue_stats[venue] = {
                    "batting": self._calculate_batting_stats(venue_df, player),
                    "bowling": self._calculate_bowling_stats(venue_df, player),
                }
            if venue_stats:
                self._save_player_stats(player, venue_stats, self.venue_stats_dir)
                logger.info(f"Saved venue stats for player {player}")

            # Process team stats
            team_stats = {}
            for team, deliveries in team_data[player].items():
                team_df = pd.DataFrame(deliveries)
                team_stats[team] = {
                    "batting": self._calculate_batting_stats(team_df, player),
                    "bowling": self._calculate_bowling_stats(team_df, player),
                }
            if team_stats:
                self._save_player_stats(player, team_stats, self.vs_team_stats_dir)
                logger.info(f"Saved vs team stats for player {player}")

            # Process player stats
            player_stats = {}
            for opponent, deliveries in player_data[player].items():
                player_df = pd.DataFrame(deliveries)
                player_stats[opponent] = {
                    "batting": self._calculate_batting_stats(player_df, player),
                    "bowling": self._calculate_bowling_stats(player_df, player),
                }
            if player_stats:
                self._save_player_stats(player, player_stats, self.vs_player_stats_dir)
                logger.info(f"Saved vs player stats for player {player}")

    def _calculate_batting_stats(
        self, deliveries_df: pd.DataFrame, player_name: str
    ) -> Dict:
        """Calculate comprehensive batting statistics."""
        # Filter deliveries where player batted
        batting_data = deliveries_df[deliveries_df["batter"] == player_name]

        if batting_data.empty:
            return {}

        # Basic stats
        stats = {
            "matches": len(batting_data["match_id"].unique()),
            "runs": batting_data["batsman_runs"].sum(),
            "balls": len(batting_data),
            "fours": len(batting_data[batting_data["batsman_runs"] == 4]),
            "sixes": len(batting_data[batting_data["batsman_runs"] == 6]),
            "dots": len(batting_data[batting_data["batsman_runs"] == 0]),
            "dismissals": len(
                batting_data[batting_data["player_dismissed"] == player_name]
            ),
        }

        # Calculate highest score
        match_scores = batting_data.groupby("match_id")["batsman_runs"].sum()
        stats["highest"] = match_scores.max() if not match_scores.empty else 0

        # Calculate 50s and 100s
        stats["50s"] = len(match_scores[(match_scores >= 50) & (match_scores < 100)])
        stats["100s"] = len(match_scores[match_scores >= 100])

        # Calculate derived statistics
        if stats["balls"] > 0:
            stats["strike_rate"] = (stats["runs"] * 100) / stats["balls"]
        if stats["dismissals"] > 0:
            stats["average"] = stats["runs"] / stats["dismissals"]

        return stats

    def _calculate_bowling_stats(
        self, deliveries_df: pd.DataFrame, player_name: str
    ) -> Dict:
        """Calculate comprehensive bowling statistics."""
        # Filter deliveries where player bowled
        bowling_data = deliveries_df[deliveries_df["bowler"] == player_name]

        if bowling_data.empty:
            return {}

        # Basic stats
        stats = {
            "matches": len(bowling_data["match_id"].unique()),
            "balls": len(bowling_data),
            "runs": bowling_data["total_runs"].sum(),
            "wickets": len(bowling_data[bowling_data["is_wicket"] == 1]),
            "maidens": len(bowling_data[bowling_data["total_runs"] == 0]) / 6,
        }

        # Calculate best bowling
        match_wickets = bowling_data.groupby("match_id")["is_wicket"].sum()
        match_runs = bowling_data.groupby("match_id")["total_runs"].sum()

        if not match_wickets.empty:
            best_wickets = match_wickets.max()
            best_runs = match_runs[match_wickets == best_wickets].min()
            stats["best_bowling"] = f"{best_wickets}/{best_runs}"

        # Calculate derived statistics
        if stats["balls"] > 0:
            stats["overs"] = stats["balls"] / 6
            stats["economy"] = stats["runs"] / stats["overs"]
        if stats["wickets"] > 0:
            stats["average"] = stats["runs"] / stats["wickets"]
            stats["strike_rate"] = stats["balls"] / stats["wickets"]

        return stats

    def _save_player_stats(
        self, player_name: str, stats: Dict, directory: Path
    ) -> None:
        """Save player statistics to a JSON file."""
        stats_file = directory / f"{player_name}.json"

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

    def get_player_all_time_stats(self, player_name: str) -> Dict:
        """Retrieve pre-computed all-time statistics for a player."""
        stats_file = self.all_time_stats_dir / f"{player_name}.json"
        return self._load_player_stats(stats_file)

    def get_player_venue_stats(self, player_name: str) -> Dict:
        """Retrieve pre-computed venue statistics for a player."""
        stats_file = self.venue_stats_dir / f"{player_name}.json"
        return self._load_player_stats(stats_file)

    def get_player_vs_team_stats(self, player_name: str) -> Dict:
        """Retrieve pre-computed team-specific statistics for a player."""
        stats_file = self.vs_team_stats_dir / f"{player_name}.json"
        return self._load_player_stats(stats_file)

    def get_player_vs_player_stats(self, player_name: str) -> Dict:
        """Retrieve pre-computed player-specific statistics for a player."""
        stats_file = self.vs_player_stats_dir / f"{player_name}.json"
        return self._load_player_stats(stats_file)

    def _load_player_stats(self, stats_file: Path) -> Dict:
        """Load player statistics from a JSON file."""
        try:
            if not stats_file.exists():
                logger.warning(f"No stats file found: {stats_file}")
                return {}

            with open(stats_file, "r") as f:
                stats = json.load(f)
            return stats
        except Exception as e:
            logger.error(f"Error loading player stats: {e}")
            return {}
