from typing import Dict, Optional

from ..utils.logger import logger
from .vector_store import VectorStore


class RAGRetriever:
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
    ):
        """Initialize the RAG retriever."""
        self.vector_store = vector_store or VectorStore()

    def get_relevant_context(
        self, query: str, team1: str, team2: str, venue: str
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

            # Query for team1 player statistics - all four types
            team1_player_types = [
                "player_vs_player",
                "player_vs_team",
                "player_venue",
                "player_all_time",
            ]

            for player_type in team1_player_types:
                # Query by type only
                player_results = self.vector_store.similarity_search(
                    query=f"{player_type} statistics for {team1}",
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

            # Query for team2 player statistics - all four types
            for player_type in team1_player_types:
                # Query by type only
                player_results = self.vector_store.similarity_search(
                    query=f"{player_type} statistics for {team2}",
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

    def generate_prompt(self, query: str, team1: str, team2: str, venue: str) -> str:
        """
        Generate a prompt for the LLM based on the query and match details.
        """
        try:
            # Get relevant context
            context = self.get_relevant_context(query, team1, team2, venue)

            # Format the context
            formatted_context = self.format_context(context)

            # Generate the prompt
            prompt = f"""
            You are an IPL cricket expert. Based on the following information, 
            provide a detailed analysis and prediction for the match between 
            {team1} and {team2} at {venue}.
            
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
