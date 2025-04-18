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
            venue_filter = {"type": "venue_stats", "venue": venue}
            venue_results = self.vector_store.similarity_search(
                query=f"venue statistics for {venue}",
                filter_dict=venue_filter,
                n_results=3,
            )
            context["venue_stats"] = venue_results

            # Query for team head-to-head statistics
            h2h_filter = {
                "type": "team_h2h",
                "$or": [
                    {"team1": team1, "team2": team2},
                    {"team1": team2, "team2": team1},
                ],
            }
            h2h_results = self.vector_store.similarity_search(
                query=f"head to head statistics between {team1} and {team2}",
                filter_dict=h2h_filter,
                n_results=3,
            )
            context["h2h_stats"] = h2h_results

            # Query for team1 player statistics - all four types
            team1_player_types = [
                "player_vs_player",
                "player_vs_team",
                "player_venue",
                "player_all_time",
            ]

            for player_type in team1_player_types:
                player_filter = {
                    "type": player_type,
                    "$or": [{"team": team2}, {"venue": venue}, {"opponent": team2}],
                }

                query_text = (
                    f"player statistics for {team1} "
                    f"({player_type}) against {team2} at {venue}"
                )

                player_results = self.vector_store.similarity_search(
                    query=query_text, filter_dict=player_filter, n_results=2
                )
                context["team1_player_stats"].extend(player_results)

            # Query for team2 player statistics - all four types
            for player_type in team1_player_types:
                player_filter = {
                    "type": player_type,
                    "$or": [{"team": team1}, {"venue": venue}, {"opponent": team1}],
                }

                query_text = (
                    f"player statistics for {team2} "
                    f"({player_type}) against {team1} at {venue}"
                )

                player_results = self.vector_store.similarity_search(
                    query=query_text, filter_dict=player_filter, n_results=2
                )
                context["team2_player_stats"].extend(player_results)

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
