import csv
from pathlib import Path
from typing import Dict, Optional

from ..config.settings import settings
from ..utils.logger import logger
from .vector_store import VectorStore


class RAGRetriever:
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
    ):
        """Initialize the RAG retriever."""
        self.vector_store = vector_store or VectorStore()
        self.current_squads = self._load_current_squads()

    def _load_current_squads(self) -> Dict:
        """Load current IPL 2025 squad information from CSV files."""
        try:
            squads_dir = (
                Path(settings.data_dir)
                / "cleaned_data"
                / "squads_per_season_data"
                / "2025"
            )
            squads = {}

            # Check if directory exists
            if not squads_dir.exists():
                logger.warning(f"Squads directory not found: {squads_dir}")
                return {}

            # Load each team's squad
            for squad_file in squads_dir.glob("*_squad.csv"):
                team_name = squad_file.stem.replace("_squad", "").replace("_", " ")
                squads[team_name] = []

                with open(squad_file, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        squads[team_name].append(row)

            return squads
        except Exception as e:
            logger.error(f"Error loading current squads: {e}")
            return {}

    def _format_team_squad_info(self, team: str) -> str:
        """Format current squad information for a team."""
        if team not in self.current_squads:
            return f"No current squad information available for {team}"

        squad = self.current_squads[team]
        if not squad:
            return f"No players found in the squad for {team}"

        formatted_info = [f"Current Squad for {team}:"]

        # Add all player names
        for player in squad:
            player_name = player.get("Player Name", "")
            if player_name:
                formatted_info.append(f"  - {player_name}")

        return "\n".join(formatted_info)

    def get_relevant_context(
        self, query: str, team1: str, team2: str, venue: str, pitch_report: str
    ) -> Dict:
        """
        Retrieve relevant context for the given query and match details.
        """
        try:
            # Initialize context dictionary
            context = {
                "venue_stats": [],
                "h2h_stats": [],
                "team1_player_stats": [],
                "team2_player_stats": [],
                "team1_squad": self._format_team_squad_info(team1),
                "team2_squad": self._format_team_squad_info(team2),
                "pitch_report": pitch_report,
            }

            # Query for venue statistics
            venue_results = self.vector_store.similarity_search(
                query=f"venue statistics for {venue}",
                filter_dict={"type": "venue_stats"},
                n_results=3,
            )
            # Filter results by venue in Python
            context["venue_stats"] = [
                result
                for result in venue_results
                if result["metadata"].get("venue") == venue
            ]

            # Query for team head-to-head statistics
            h2h_results = self.vector_store.similarity_search(
                query=f"head to head statistics between {team1} and {team2}",
                filter_dict={"type": "team_h2h"},
                n_results=5,
            )
            # Filter results by team1 and team2 in Python
            context["h2h_stats"] = [
                result
                for result in h2h_results
                if (
                    result["metadata"].get("team1") == team1
                    and result["metadata"].get("team2") == team2
                )
                or (
                    result["metadata"].get("team1") == team2
                    and result["metadata"].get("team2") == team1
                )
            ]

            # Get player statistics for each player in team1
            if team1 in self.current_squads:
                for player in self.current_squads[team1]:
                    player_name = player.get("Delivery Name", "")
                    if player_name:
                        # Query for player statistics - all four types
                        player_types = [
                            "player_vs_player",
                            "player_vs_team",
                            "player_venue",
                            "player_all_time",
                        ]

                        for player_type in player_types:
                            # Query by player name and type
                            player_results = self.vector_store.similarity_search(
                                query=f"{player_type} statistics for {player_name}",
                                filter_dict={"type": player_type},
                                n_results=5,
                            )

                            # Filter results in Python
                            filtered_results = [
                                result
                                for result in player_results
                                if (
                                    result["metadata"].get("team") == team2
                                    or result["metadata"].get("venue") == venue
                                    or result["metadata"].get("opponent") == team2
                                )
                            ]

                            context["team1_player_stats"].extend(filtered_results)

            # Get player statistics for each player in team2
            if team2 in self.current_squads:
                for player in self.current_squads[team2]:
                    player_name = player.get("Delivery Name", "")
                    if player_name:
                        # Query for player statistics - all four types
                        player_types = [
                            "player_vs_player",
                            "player_vs_team",
                            "player_venue",
                            "player_all_time",
                        ]

                        for player_type in player_types:
                            # Query by player name and type
                            player_results = self.vector_store.similarity_search(
                                query=f"{player_type} statistics for {player_name}",
                                filter_dict={"type": player_type},
                                n_results=5,
                            )

                            # Filter results in Python
                            filtered_results = [
                                result
                                for result in player_results
                                if (
                                    result["metadata"].get("team") == team1
                                    or result["metadata"].get("venue") == venue
                                    or result["metadata"].get("opponent") == team1
                                )
                            ]

                            context["team2_player_stats"].extend(filtered_results)

            return context

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            # Return empty context instead of raising an exception
            return {
                "venue_stats": [],
                "h2h_stats": [],
                "team1_player_stats": [],
                "team2_player_stats": [],
                "team1_squad": self._format_team_squad_info(team1),
                "team2_squad": self._format_team_squad_info(team2),
            }

    def format_context(self, context: Dict) -> str:
        """Format retrieved context into a string for the LLM."""
        try:
            formatted_text = []

            # Format venue statistics
            if context.get("venue_stats"):
                formatted_text.append("Venue Statistics:")
                for stat in context["venue_stats"]:
                    formatted_text.append(stat["content"])

            # Format head-to-head statistics
            if context.get("h2h_stats"):
                formatted_text.append("\nHead-to-Head Statistics:")
                for stat in context["h2h_stats"]:
                    formatted_text.append(stat["content"])

            # Format team1 player statistics
            if context.get("team1_player_stats"):
                formatted_text.append("\nTeam 1 Player Statistics:")
                for stat in context["team1_player_stats"]:
                    formatted_text.append(stat["content"])

            # Format team2 player statistics
            if context.get("team2_player_stats"):
                formatted_text.append("\nTeam 2 Player Statistics:")
                for stat in context["team2_player_stats"]:
                    formatted_text.append(stat["content"])

            return "\n".join(formatted_text)

        except Exception as e:
            logger.error(f"Error formatting context: {e}")
            return ""

    def generate_prompt(
        self, query: str, team1: str, team2: str, venue: str, pitch_report: str
    ) -> str:
        """
        Generate a prompt for the LLM based on the query and match details.
        """
        try:
            # Get relevant context
            context = self.get_relevant_context(
                query, team1, team2, venue, pitch_report
            )

            # Format the context
            formatted_context = self.format_context(context)

            # Generate the prompt
            prompt = f"""
            You are an IPL cricket expert. Based on the following information, 
            provide a detailed analysis and prediction for the match between 
            {team1} and {team2} at {venue} with the following pitch report: {pitch_report}.

            Below is the current squad information for the teams and only consider the players in the squad for predictions

            Team 1 Current Squad:
            {context["team1_squad"]}

            Team 2 Current Squad:
            {context["team2_squad"]}
            
            IMPORTANT CONTEXT:
            - The below statistics provided are from IPL seasons 2008-2024 (historical data)
            - So don't assume below statistics are for IPL 2025 season (season 18)
            
            Here is the relevant cricket statistics:
            
            {formatted_context}
            
            Based on this information, please provide:
            1. A detailed analysis of the match conditions and team strengths
            2. Key players to watch out for from both teams
            3. A prediction for the match outcome
            
            User query: {query}
            """

            return prompt

        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            return (
                f"Analyze the match between {team1} and {team2} at {venue}. "
                f"User query: {query}"
            )
