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
                    player_name = player.get("Player Name", "")
                    delivery_name = player.get("Delivery Name", "")

                    if player_name and delivery_name:
                        # Query for player statistics - all four types
                        player_types = [
                            "player_vs_player",
                            "player_vs_team",
                            "player_venue",
                            "player_all_time",
                        ]

                        for player_type in player_types:
                            # Query by delivery name and type
                            player_results = self.vector_store.similarity_search(
                                query=f"{player_type} statistics for {delivery_name}",
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

                            # Add player name to metadata for context
                            for result in filtered_results:
                                result["metadata"]["player_name"] = player_name
                                # Replace delivery name with player name in content
                                if "content" in result:
                                    result["content"] = result["content"].replace(
                                        delivery_name, player_name
                                    )

                            context["team1_player_stats"].extend(filtered_results)

            # Get player statistics for each player in team2
            if team2 in self.current_squads:
                for player in self.current_squads[team2]:
                    player_name = player.get("Player Name", "")
                    delivery_name = player.get("Delivery Name", "")

                    if player_name and delivery_name:
                        # Query for player statistics - all four types
                        player_types = [
                            "player_vs_player",
                            "player_vs_team",
                            "player_venue",
                            "player_all_time",
                        ]

                        for player_type in player_types:
                            # Query by delivery name and type
                            player_results = self.vector_store.similarity_search(
                                query=f"{player_type} statistics for {delivery_name}",
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

                            # Add player name to metadata for context
                            for result in filtered_results:
                                result["metadata"]["player_name"] = player_name
                                # Replace delivery name with player name in content
                                if "content" in result:
                                    result["content"] = result["content"].replace(
                                        delivery_name, player_name
                                    )

                            context["team2_player_stats"].extend(filtered_results)

            # Get player vs player statistics for key matchups
            if team1 in self.current_squads and team2 in self.current_squads:
                # Get key players from both teams
                team1_key_players = [
                    player
                    for player in self.current_squads[team1]
                    if player.get("Role", "")
                    in ["top-order batter", "allrounder", "bowler"]
                ]

                team2_key_players = [
                    player
                    for player in self.current_squads[team2]
                    if player.get("Role", "")
                    in ["top-order batter", "allrounder", "bowler"]
                ]

                # For each key player in team1, get stats against key players in team2
                for player1 in team1_key_players:
                    player1_name = player1.get("Player Name", "")
                    player1_delivery = player1.get("Delivery Name", "")

                    if player1_name and player1_delivery:
                        for player2 in team2_key_players:
                            player2_name = player2.get("Player Name", "")
                            player2_delivery = player2.get("Delivery Name", "")

                            if player2_name and player2_delivery:
                                # Query for player vs player statistics
                                pvp_results = self.vector_store.similarity_search(
                                    query=f"player vs player statistics for {player1_delivery} against {player2_delivery}",
                                    filter_dict={"type": "player_vs_player"},
                                    n_results=3,
                                )

                                # Filter results to ensure they match both players
                                filtered_pvp = [
                                    result
                                    for result in pvp_results
                                    if (
                                        result["metadata"].get("player")
                                        == player1_delivery
                                        and result["metadata"].get("opponent")
                                        == player2_delivery
                                    )
                                ]

                                # Add player names to metadata for context
                                for result in filtered_pvp:
                                    result["metadata"]["player_name"] = player1_name
                                    result["metadata"]["opponent_name"] = player2_name
                                    # Replace delivery names with player names in content
                                    if "content" in result:
                                        result["content"] = (
                                            result["content"]
                                            .replace(player1_delivery, player1_name)
                                            .replace(player2_delivery, player2_name)
                                        )

                                # Add to team1 player stats
                                context["team1_player_stats"].extend(filtered_pvp)

                # For each key player in team2, get stats against key players in team1
                for player2 in team2_key_players:
                    player2_name = player2.get("Player Name", "")
                    player2_delivery = player2.get("Delivery Name", "")

                    if player2_name and player2_delivery:
                        for player1 in team1_key_players:
                            player1_name = player1.get("Player Name", "")
                            player1_delivery = player1.get("Delivery Name", "")

                            if player1_name and player1_delivery:
                                # Query for player vs player statistics
                                pvp_results = self.vector_store.similarity_search(
                                    query=f"player vs player statistics for {player2_delivery} against {player1_delivery}",
                                    filter_dict={"type": "player_vs_player"},
                                    n_results=3,
                                )

                                # Filter results to ensure they match both players
                                filtered_pvp = [
                                    result
                                    for result in pvp_results
                                    if (
                                        result["metadata"].get("player")
                                        == player2_delivery
                                        and result["metadata"].get("opponent")
                                        == player1_delivery
                                    )
                                ]

                                # Add player names to metadata for context
                                for result in filtered_pvp:
                                    result["metadata"]["player_name"] = player2_name
                                    result["metadata"]["opponent_name"] = player1_name
                                    # Replace delivery names with player names in content
                                    if "content" in result:
                                        result["content"] = (
                                            result["content"]
                                            .replace(player2_delivery, player2_name)
                                            .replace(player1_delivery, player1_name)
                                        )

                                # Add to team2 player stats
                                context["team2_player_stats"].extend(filtered_pvp)

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
        self,
        query: str,
        context: Dict,
        summarize_response: str,
        team1: str,
        team2: str,
        venue: str,
        pitch_report: str,
    ) -> str:
        """
        Generate a prompt for the LLM based on the query and match details.
        """
        try:
            # Generate the prompt
            prompt = f"""
            You are an IPL cricket expert. Based on the following information, 
            provide a detailed analysis and prediction for the match between 
            {team1} and {team2} at {venue} with the following pitch report: {pitch_report}.
            
            IMPORTANT CONTEXT:
            - The statistics provided are from IPL seasons 2008-2024 (historical data)
            - Do not assume these statistics are for IPL 2025 season (season 18)
            
            Here is a summary of the relevant cricket statistics:
            
            {summarize_response}
            
            Based on this information, please provide:
            1. A detailed analysis of the match conditions and team strengths
            2. Get the details for all the players if possible
            3. A prediction for the match outcome with a confidence level (high/medium/low)
            4. Key factors that could influence the match result

            Below is the current squad information for the teams. Only consider the players in the squad for predictions:

            Team 1 Current Squad:
            {context["team1_squad"]}

            Team 2 Current Squad:
            {context["team2_squad"]}
            
            User query: {query}
            """

            return prompt

        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            return (
                f"Analyze the match between {team1} and {team2} at {venue}. "
                f"User query: {query}"
            )

    def generate_summarize_prompt(self, formatted_context: str) -> str:
        """
        Generate a prompt for the LLM to summarize the formatted context.
        """
        try:
            # Generate the summarize prompt
            summarize_prompt = f"""
            You are an IPL cricket expert. Please summarize the following cricket statistics 
            in a concise and informative way. Focus on the most important insights that would 
            influence a match prediction.
            
            Here is the cricket statistics to summarize:
            
            {formatted_context}
            
            Please provide a clear, structured summary that highlights:
            1. Key venue insights
            2. Important head-to-head statistics
            3. Notable player performances
            4. Any other critical factors that would impact the match outcome
            
            Keep your summary concise but comprehensive, focusing on the most relevant information.
            """

            return summarize_prompt

        except Exception as e:
            logger.error(f"Error generating summarize prompt: {e}")
            return (
                "Please summarize the following cricket statistics: "
                + formatted_context
            )
