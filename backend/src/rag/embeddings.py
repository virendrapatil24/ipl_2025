import json
from pathlib import Path
from typing import Dict, List, Union

from sentence_transformers import SentenceTransformer

from ..utils.logger import logger


class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding generator with specified model."""
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            return self.model.encode(texts).tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def create_match_document(self, match_data: Dict) -> str:
        """Create a document from match data for embedding."""
        return (
            f"Match between {match_data['team1']} and {match_data['team2']} "
            f"at {match_data['venue']} on {match_data['date']}. "
            f"Toss won by {match_data['toss_winner']} who chose to {match_data['toss_decision']}. "
            f"Winner: {match_data.get('winner', 'Not available')}. "
            f"Result: {match_data['result']}"
        )

    def create_player_document(self, player_data: Dict) -> str:
        """Create a document from player data for embedding."""
        return (
            f"Player {player_data['name']} plays for {player_data['team']} "
            f"as a {player_data['role']}. "
            f"Batting style: {player_data.get('batting_style', 'Not specified')}. "
            f"Bowling style: {player_data.get('bowling_style', 'Not specified')}. "
            f"{'Overseas player.' if player_data.get('is_overseas') else 'Local player.'}"
        )

    def create_analysis_document(self, analysis_data: Dict) -> str:
        """Create a document from analysis data for embedding."""
        return (
            f"Analysis for {analysis_data.get('venue', 'venue')}:\n"
            f"Total matches: {analysis_data.get('total_matches', 0)}\n"
            f"Batting first wins: {analysis_data.get('batting_first_wins', 0)}\n"
            f"Average score: {analysis_data.get('average_score', 'Not available')}\n"
            f"Pitch characteristics: {analysis_data.get('pitch_behavior', 'Not specified')}"
        )
