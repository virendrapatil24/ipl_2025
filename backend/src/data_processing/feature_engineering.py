from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils.logger import logger


class FeatureEngineering:
    @staticmethod
    def calculate_venue_stats(matches_df: pd.DataFrame, venue: str) -> Dict:
        """Calculate venue-specific statistics."""
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

            return {
                "total_matches": total_matches,
                "batting_first_wins": batting_first_wins,
                "batting_second_wins": total_matches - batting_first_wins,
                "win_percentage_batting_first": round(
                    batting_first_wins / total_matches * 100, 2
                ),
            }
        except Exception as e:
            logger.error(f"Error calculating venue stats: {e}")
            raise

    @staticmethod
    def calculate_h2h_stats(
        matches_df: pd.DataFrame, team1: str, team2: str, last_n: int = 4
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
