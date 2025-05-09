from typing import Dict

import numpy as np
import pandas as pd

from ..config.settings import settings
from ..utils.logger import logger
from .data_loader import DataLoader


class FeatureEngineering:
    """Processes and calculates various cricket statistics and features."""

    def __init__(self, data_loader: DataLoader):
        """Initialize the feature engineering class."""
        self.data_loader = data_loader
        self.processed_data_dir = settings.processed_data_dir
        self.processed_data_dir.mkdir(exist_ok=True)

        # Create directories for processed data
        self.venue_stats_dir = self.processed_data_dir / "venue_stats"
        self.venue_stats_dir.mkdir(exist_ok=True)

        self.team_at_venue_stats_dir = self.processed_data_dir / "team_at_venue_stats"
        self.team_at_venue_stats_dir.mkdir(exist_ok=True)

        self.team_h2h_stats_dir = self.processed_data_dir / "team_h2h_stats"
        self.team_h2h_stats_dir.mkdir(exist_ok=True)

        logger.info("FeatureEngineering initialized")

    def calculate_venue_statistics(self, matches_df: pd.DataFrame) -> None:
        """Calculate and save venue statistics."""
        logger.info("Calculating venue statistics...")
        venues = matches_df["venue"].unique()
        for venue in venues:
            venue_stats = self.calculate_venue_stats(matches_df, venue)
            self._save_venue_stats(venue, venue_stats)

    def calculate_venue_stats(self, matches_df: pd.DataFrame, venue: str) -> Dict:
        """Calculate statistics for a specific venue."""
        try:
            venue_matches = matches_df[matches_df["venue"] == venue]
            total_matches = len(venue_matches)

            # Batting first/second stats
            batting_first_wins = len(
                venue_matches[
                    venue_matches.apply(
                        lambda x: (
                            x["winner"] == x["team1"]
                            if x["toss_decision"] == "bat"
                            else (
                                x["winner"] == x["team2"]
                                if x["toss_decision"] == "field"
                                else False
                            )
                        ),
                        axis=1,
                    )
                ]
            )

            # Average runs scored by the team batting first
            match_ids = venue_matches["match_id"].unique()
            total_first_innings_runs = 0
            total_second_innings_runs = 0
            total_first_innings_wickets = 0
            total_second_innings_wickets = 0
            for match_id in match_ids:
                match_data = self.data_loader.load_deliveries(match_id)
                if not match_data.empty:
                    first_innings_runs = match_data[match_data["inning"] == 1][
                        "total_runs"
                    ].sum()
                    second_innings_runs = match_data[match_data["inning"] == 2][
                        "total_runs"
                    ].sum()
                    total_first_innings_runs += first_innings_runs
                    total_second_innings_runs += second_innings_runs
                    total_first_innings_wickets += len(
                        match_data[
                            (match_data["inning"] == 1)
                            & (match_data["is_wicket"] == True)
                        ]
                    )
                    total_second_innings_wickets += len(
                        match_data[
                            (match_data["inning"] == 2)
                            & (match_data["is_wicket"] == True)
                        ]
                    )
            if total_matches > 0:
                avg_first_innings_runs = total_first_innings_runs / total_matches
                avg_second_innings_runs = total_second_innings_runs / total_matches
                avg_first_innings_wickets = (
                    (total_first_innings_wickets / total_matches)
                    if total_first_innings_wickets > 0
                    else 0
                )
                avg_second_innings_wickets = (
                    (total_second_innings_wickets / total_matches)
                    if total_second_innings_wickets > 0
                    else 0
                )

            return {
                "total_matches": total_matches,
                "batting_first_wins": batting_first_wins,
                "batting_second_wins": total_matches - batting_first_wins,
                "win_percentage_batting_first": round(
                    batting_first_wins / total_matches * 100, 2
                ),
                "avg_first_innings_runs": avg_first_innings_runs,
                "avg_second_innings_runs": avg_second_innings_runs,
                "avg_first_innings_wickets": avg_first_innings_wickets,
                "avg_second_innings_wickets": avg_second_innings_wickets,
            }
        except Exception as e:
            logger.error(f"Error calculating venue stats: {e}")
            raise

    def calculate_team_at_venue_statistics(self, matches_df: pd.DataFrame) -> None:
        """Calculate and save team-specific statistics at each venue."""
        logger.info("Calculating team at venue statistics...")
        teams = matches_df["team1"].unique()
        for team in teams:
            team_stats = self.calculate_team_at_venue_stats(matches_df, team)
            self._save_team_at_venue_stats(team, team_stats)

    def calculate_team_at_venue_stats(
        self, matches_df: pd.DataFrame, team: str
    ) -> Dict:
        """Calculate statistics for a specific team at each venue."""
        try:
            team_matches = matches_df[
                (matches_df["team1"] == team) | (matches_df["team2"] == team)
            ]
            total_matches = len(team_matches)

            venues = team_matches["venue"].unique()
            venue_stats = {}
            for venue in venues:
                venue_data = team_matches[team_matches["venue"] == venue]
                total_venue_matches = len(venue_data)

                # Batting first/second stats
                batting_first_wins = len(
                    venue_data[
                        venue_data.apply(
                            lambda x: (
                                x["winner"] == team
                                if x["toss_decision"] == "bat"
                                else (
                                    x["winner"] != team
                                    if x["toss_decision"] == "field"
                                    else False
                                )
                            ),
                            axis=1,
                        )
                    ]
                )

                total_first_innings_runs = 0
                total_second_innings_runs = 0
                total_first_innings_wickets = 0
                total_second_innings_wickets = 0
                for match_id in venue_data["match_id"].unique():
                    match_data = self.data_loader.load_deliveries(match_id)
                    if not match_data.empty:
                        first_innings_runs = match_data[match_data["inning"] == 1][
                            "total_runs"
                        ].sum()
                        second_innings_runs = match_data[match_data["inning"] == 2][
                            "total_runs"
                        ].sum()
                        total_first_innings_runs += first_innings_runs
                        total_second_innings_runs += second_innings_runs
                        total_first_innings_wickets += len(
                            match_data[
                                (match_data["inning"] == 1)
                                & (match_data["is_wicket"] == True)
                            ]
                        )
                        total_second_innings_wickets += len(
                            match_data[
                                (match_data["inning"] == 2)
                                & (match_data["is_wicket"] == True)
                            ]
                        )
                if total_venue_matches > 0:
                    avg_first_innings_runs = (
                        total_first_innings_runs / total_venue_matches
                    )
                    avg_second_innings_runs = (
                        total_second_innings_runs / total_venue_matches
                    )
                    avg_first_innings_wickets = (
                        (total_first_innings_wickets / total_venue_matches)
                        if total_first_innings_wickets > 0
                        else 0
                    )
                    avg_second_innings_wickets = (
                        (total_second_innings_wickets / total_venue_matches)
                        if total_second_innings_wickets > 0
                        else 0
                    )

                venue_stats[venue] = {
                    "total_matches": total_venue_matches,
                    "batting_first_wins": batting_first_wins,
                    "batting_second_wins": total_venue_matches - batting_first_wins,
                    "win_percentage_batting_first": round(
                        batting_first_wins / total_venue_matches * 100, 2
                    ),
                    "avg_first_innings_runs": avg_first_innings_runs,
                    "avg_second_innings_runs": avg_second_innings_runs,
                    "avg_first_innings_wickets": avg_first_innings_wickets,
                    "avg_second_innings_wickets": avg_second_innings_wickets,
                }

            return venue_stats
        except Exception as e:
            logger.error(f"Error calculating team stats: {e}")
            raise

    def _save_team_at_venue_stats(self, team: str, stats: Dict) -> None:
        """Save team-specific statistics at each venue to a JSON file."""
        import json

        def convert_to_serializable(obj):
            """Convert NumPy types to Python native types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {
                    key: convert_to_serializable(value) for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        stats_file = self.team_at_venue_stats_dir / f"{team}_at_venue_stats.json"

        try:
            # Convert NumPy types to Python native types
            serializable_stats = convert_to_serializable(stats)
            with open(stats_file, "w") as f:
                json.dump(serializable_stats, f, indent=2)
            logger.info(f"Saved team at venue stats for {team}")
        except Exception as e:
            logger.error(f"Error saving team at venue stats: {e}")

    def calculate_team_h2h_statistics(self, matches_df: pd.DataFrame) -> None:
        """Calculate and save head-to-head statistics for all teams."""
        logger.info("Calculating team head-to-head statistics...")
        teams = matches_df["team1"].unique()
        for team1 in teams:
            h2h_stats = self.calculate_team_h2h_stats(matches_df, team1)
            self._save_team_h2h_stats(team1, h2h_stats)

    def calculate_team_h2h_stats(self, matches_df: pd.DataFrame, team1: str) -> Dict:
        """Calculate head-to-head statistics between two teams."""
        try:
            teams = matches_df["team2"].unique()
            h2h_stats = {}
            for team2 in teams:
                if team1 != team2:
                    h2h_stats[team2] = self.calculate_h2h_stats(
                        matches_df, team1, team2
                    )
            return h2h_stats
        except Exception as e:
            logger.error(f"Error calculating team H2H stats: {e}")
            raise

    def _save_team_h2h_stats(self, team: str, stats: Dict) -> None:
        """Save head-to-head statistics to a JSON file."""
        import json

        def convert_to_serializable(obj):
            """Convert NumPy types to Python native types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {
                    key: convert_to_serializable(value) for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        stats_file = self.team_h2h_stats_dir / f"{team}_h2h_stats.json"

        try:
            # Convert NumPy types to Python native types
            serializable_stats = convert_to_serializable(stats)
            with open(stats_file, "w") as f:
                json.dump(serializable_stats, f, indent=2)
            logger.info(f"Saved H2H stats for {team}")
        except Exception as e:
            logger.error(f"Error saving H2H stats: {e}")

    def calculate_head_to_head_statistics(
        self, team1: str, team2: str, deliveries_df: pd.DataFrame
    ) -> Dict:
        """Calculate head-to-head statistics between two teams."""
        # Standardize team names
        standardized_team1 = self.data_loader.standardize_team_name(team1).replace(
            "_", " "
        )
        standardized_team2 = self.data_loader.standardize_team_name(team2).replace(
            "_", " "
        )

        logger.info(
            f"Calculating H2H statistics for {standardized_team1} vs {standardized_team2.replace('_', ' ')}"
        )

        # Filter data for matches between these teams
        h2h_data = deliveries_df[
            (
                (deliveries_df["batting_team"] == standardized_team1)
                & (deliveries_df["bowling_team"] == standardized_team2)
            )
            | (
                (deliveries_df["batting_team"] == standardized_team2)
                & (deliveries_df["bowling_team"] == standardized_team1)
            )
        ]

        if h2h_data.empty:
            logger.warning(
                f"No H2H data found for {standardized_team1} vs {standardized_team2}"
            )
            return {}

        # Calculate team1's stats against team2
        team1_batting = h2h_data[h2h_data["batting_team"] == standardized_team1]
        team1_bowling = h2h_data[h2h_data["bowling_team"] == standardized_team1]

        team1_stats = {
            "matches_played": len(h2h_data["match_id"].unique()),
            "batting": {
                "total_runs": team1_batting["total_runs"].sum(),
                "wickets_lost": len(team1_batting[team1_batting["is_wicket"] == 1]),
                "fours": len(team1_batting[team1_batting["batsman_runs"] == 4]),
                "sixes": len(team1_batting[team1_batting["batsman_runs"] == 6]),
            },
            "bowling": {
                "runs_conceded": team1_bowling["total_runs"].sum(),
                "wickets_taken": len(team1_bowling[team1_bowling["is_wicket"] == 1]),
                "maidens": len(team1_bowling[team1_bowling["total_runs"] == 0]) / 6,
            },
        }

        # Calculate team2's stats against team1
        team2_batting = h2h_data[h2h_data["batting_team"] == standardized_team2]
        team2_bowling = h2h_data[h2h_data["bowling_team"] == standardized_team2]

        team2_stats = {
            "matches_played": len(h2h_data["match_id"].unique()),
            "batting": {
                "total_runs": team2_batting["total_runs"].sum(),
                "wickets_lost": len(team2_batting[team2_batting["is_wicket"] == 1]),
                "fours": len(team2_batting[team2_batting["batsman_runs"] == 4]),
                "sixes": len(team2_batting[team2_batting["batsman_runs"] == 6]),
            },
            "bowling": {
                "runs_conceded": team2_bowling["total_runs"].sum(),
                "wickets_taken": len(team2_bowling[team2_bowling["is_wicket"] == 1]),
                "maidens": len(team2_bowling[team2_bowling["total_runs"] == 0]) / 6,
            },
        }

        # Calculate derived statistics
        for team_stats in [team1_stats, team2_stats]:
            if team_stats["matches_played"] > 0:
                # Batting averages
                if team_stats["batting"]["wickets_lost"] > 0:
                    team_stats["batting"]["average"] = (
                        team_stats["batting"]["total_runs"]
                        / team_stats["batting"]["wickets_lost"]
                    )
                team_stats["batting"]["runs_per_match"] = (
                    team_stats["batting"]["total_runs"] / team_stats["matches_played"]
                )

                # Bowling averages
                if team_stats["bowling"]["wickets_taken"] > 0:
                    team_stats["bowling"]["average"] = (
                        team_stats["bowling"]["runs_conceded"]
                        / team_stats["bowling"]["wickets_taken"]
                    )
                team_stats["bowling"]["runs_per_match"] = (
                    team_stats["bowling"]["runs_conceded"]
                    / team_stats["matches_played"]
                )

        h2h_stats = {
            standardized_team1: team1_stats,
            standardized_team2: team2_stats,
        }

        # Save H2H statistics
        self._save_h2h_stats(standardized_team1, standardized_team2, h2h_stats)
        return h2h_stats

    def _save_venue_stats(self, venue: str, stats: Dict) -> None:
        """Save venue statistics to a JSON file."""
        import json

        def convert_to_serializable(obj):
            """Convert NumPy types to Python native types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {
                    key: convert_to_serializable(value) for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        stats_file = self.venue_stats_dir / f"{venue}_venue_stats.json"

        try:
            # Convert NumPy types to Python native types
            serializable_stats = convert_to_serializable(stats)
            with open(stats_file, "w") as f:
                json.dump(serializable_stats, f, indent=2)
            logger.info(f"Saved venue stats for {venue}")
        except Exception as e:
            logger.error(f"Error saving venue stats: {e}")

    def _save_h2h_stats(self, team1: str, team2: str, stats: Dict) -> None:
        """Save head-to-head statistics to a JSON file."""
        import json

        def convert_to_serializable(obj):
            """Convert NumPy types to Python native types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {
                    key: convert_to_serializable(value) for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        stats_file = self.h2h_stats_dir / f"{team1}_vs_{team2}_h2h_stats.json"

        try:
            # Convert NumPy types to Python native types
            serializable_stats = convert_to_serializable(stats)
            with open(stats_file, "w") as f:
                json.dump(serializable_stats, f, indent=2)
            logger.info(f"Saved H2H stats for {team1} vs {team2}")
        except Exception as e:
            logger.error(f"Error saving H2H stats: {e}")

    def get_venue_stats(self, team: str) -> Dict:
        """Retrieve pre-computed venue statistics for a team."""
        import json

        stats_file = self.venue_stats_dir / f"{team}_venue_stats.json"

        try:
            if not stats_file.exists():
                logger.warning(f"No venue stats file found for team {team}")
                return {}

            with open(stats_file, "r") as f:
                stats = json.load(f)
            return stats
        except Exception as e:
            logger.error(f"Error loading venue stats: {e}")
            return {}

    def get_h2h_stats(self, team1: str, team2: str) -> Dict:
        """Retrieve pre-computed head-to-head statistics."""
        import json

        stats_file = self.h2h_stats_dir / f"{team1}_vs_{team2}_h2h_stats.json"

        try:
            if not stats_file.exists():
                logger.warning(f"No H2H stats file found for {team1} vs {team2}")
                return {}

            with open(stats_file, "r") as f:
                stats = json.load(f)
            return stats
        except Exception as e:
            logger.error(f"Error loading H2H stats: {e}")
            return {}

    @staticmethod
    def calculate_h2h_stats(
        matches_df: pd.DataFrame, team1: str, team2: str, last_n: int = 50
    ) -> Dict:
        """Calculate head-to-head statistics between two teams."""
        try:
            h2h_matches = matches_df[
                ((matches_df["team1"] == team1) & (matches_df["team2"] == team2))
                | ((matches_df["team1"] == team2) & (matches_df["team2"] == team1))
            ].tail(last_n)

            team1_wins = len(h2h_matches[h2h_matches["winner"] == team1])
            team2_wins = len(h2h_matches[h2h_matches["winner"] == team2])

            return {
                "matches_played": len(h2h_matches),
                f"{team1}_wins": team1_wins,
                f"{team2}_wins": team2_wins,
                "recent_form": h2h_matches[["date", "winner", "result"]].to_dict(
                    "records"
                ),
            }
        except Exception as e:
            logger.error(f"Error calculating H2H stats: {e}")
            raise

    @staticmethod
    def calculate_player_stats(
        deliveries_df: pd.DataFrame, player_name: str, role: str
    ) -> Dict:
        """Calculate player-specific statistics."""
        try:
            if role in ["Batsman", "All-Rounder", "Wicket-Keeper"]:
                batting_stats = {
                    "runs": deliveries_df[deliveries_df["batter"] == player_name][
                        "batsman_runs"
                    ].sum(),
                    "balls_faced": len(
                        deliveries_df[deliveries_df["batter"] == player_name]
                    ),
                    "dismissals": len(
                        deliveries_df[
                            (deliveries_df["player_dismissed"] == player_name)
                            & (deliveries_df["is_wicket"] == True)
                        ]
                    ),
                }
                batting_stats["strike_rate"] = (
                    round(
                        (batting_stats["runs"] / batting_stats["balls_faced"]) * 100, 2
                    )
                    if batting_stats["balls_faced"] > 0
                    else 0
                )

                return {"batting_stats": batting_stats}

            if role in ["Bowler", "All-Rounder"]:
                bowling_stats = {
                    "wickets": len(
                        deliveries_df[
                            (deliveries_df["bowler"] == player_name)
                            & (deliveries_df["is_wicket"] is True)
                        ]
                    ),
                    "runs_conceded": deliveries_df[
                        deliveries_df["bowler"] == player_name
                    ]["total_runs"].sum(),
                    "overs_bowled": len(
                        deliveries_df[deliveries_df["bowler"] == player_name]
                    )
                    / 6,
                }
                bowling_stats["economy"] = (
                    round(
                        bowling_stats["runs_conceded"] / bowling_stats["overs_bowled"],
                        2,
                    )
                    if bowling_stats["overs_bowled"] > 0
                    else 0
                )

                return {"bowling_stats": bowling_stats}

            return {}
        except Exception as e:
            logger.error(f"Error calculating player stats: {e}")
            raise
