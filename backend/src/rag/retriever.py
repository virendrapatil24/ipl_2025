from typing import Dict, Optional

from ..utils.logger import logger
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore


class RAGRetriever:
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
    ):
        """Initialize the RAG retriever."""
        self.vector_store = vector_store or VectorStore()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()

    def get_relevant_context(
        self, query: str, team1: str, team2: str, venue: str
    ) -> Dict:
        """
        Retrieve relevant context for the given query and match details.
        """
        try:
            # Create specific queries for different aspects
            match_query = f"Matches between {team1} " f"and {team2} at {venue}"
            player_query = f"Players from {team1} and {team2}"
            venue_query = f"Analysis of matches at {venue}"

            # Get relevant matches
            relevant_matches = self.vector_store.query_matches(match_query, n_results=3)

            # If no matches found, try a more general query
            if not relevant_matches:
                logger.warning(
                    f"No specific matches found for {team1} vs {team2} at {venue}. Trying general query."
                )
                match_query = f"Matches at {venue}"
                relevant_matches = self.vector_store.query_matches(
                    match_query, n_results=3
                )

            # Get relevant players
            relevant_players = self.vector_store.query_players(
                player_query, n_results=5
            )

            # If no players found, try a more general query
            if not relevant_players:
                logger.warning(
                    f"No specific players found for {team1} and {team2}. Trying general query."
                )
                player_query = "Players in IPL"
                relevant_players = self.vector_store.query_players(
                    player_query, n_results=5
                )

            # Get relevant analysis
            relevant_analysis = self.vector_store.query_analysis(
                venue_query, n_results=2
            )

            # If no analysis found, try a more general query
            if not relevant_analysis:
                logger.warning(
                    f"No specific analysis found for {venue}. Trying general query."
                )
                venue_query = "IPL match analysis"
                relevant_analysis = self.vector_store.query_analysis(
                    venue_query, n_results=2
                )

            print("relevant_matches", relevant_matches)
            print("relevant_players", relevant_players)
            print("relevant_analysis", relevant_analysis)

            # Combine all context
            context = {
                "matches": relevant_matches,
                "players": relevant_players,
                "analysis": relevant_analysis,
            }

            return context

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            # Return empty context instead of raising an exception
            return {"matches": [], "players": [], "analysis": []}

    def format_context(self, context: Dict) -> str:
        """Format retrieved context into a string for the LLM."""
        try:
            formatted_text = []

            # Format match information
            if context.get("matches"):
                formatted_text.append("Recent Matches:")
                for match in context["matches"]:
                    formatted_text.append(match["document"])
            else:
                formatted_text.append("No specific match information available.")

            # Format player information
            if context.get("players"):
                formatted_text.append("\nKey Players:")
                for player in context["players"]:
                    formatted_text.append(player["document"])
            else:
                formatted_text.append("\nNo specific player information available.")

            # Format analysis information
            if context.get("analysis"):
                formatted_text.append("\nVenue Analysis:")
                for analysis in context["analysis"]:
                    formatted_text.append(analysis["document"])
            else:
                formatted_text.append("\nNo specific venue analysis available.")

            return "\n".join(formatted_text)

        except Exception as e:
            logger.error(f"Error formatting context: {e}")
            return "No context information available."

    def generate_prompt(self, query: str, team1: str, team2: str, venue: str) -> str:
        """Generate a prompt combining the query and retrieved context."""
        try:
            # Get relevant context
            context = self.get_relevant_context(query, team1, team2, venue)
            formatted_context = self.format_context(context)
            print("context", context)
            print("formatted_context", formatted_context)

            # Create the final prompt template
            template = (
                "User Query: {query}\n\n"
                "Context:\n"
                "{context}\n\n"
                "Based on the above context, provide a detailed analysis for "
                "the match between {team1} and {team2} at {venue}.\n"
                "Include:\n"
                "1. Historical performance at the venue\n"
                "2. Head-to-head record\n"
                "3. Key player matchups\n"
                "4. Pitch conditions and their impact\n"
                "5. Predictions for the match\n\n"
                "Response:\n"
            )

            # Format the prompt with the actual values
            prompt = template.format(
                query=query,
                context=formatted_context,
                team1=team1,
                team2=team2,
                venue=venue,
            )

            return prompt

        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            # Return a basic prompt if there's an error
            return f"User Query: {query}\n\nPlease provide an analysis for the match between {team1} and {team2} at {venue}."
