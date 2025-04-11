from typing import Dict, List

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
            f"Toss won by {match_data['toss_winner']} who chose to "
            f"{match_data['toss_decision']}. "
            f"Winner: {match_data.get('winner', 'Not available')}. "
            f"Result: {match_data['result']}"
        )

    def create_player_document(self, player_data: Dict) -> str:
        """Create a document from player data for embedding."""
        batting = player_data.get("batting", {})
        bowling = player_data.get("bowling", {})

        batting_stats = (
            f"Batting: {batting.get('matches_played', 0)} matches, "
            f"{batting.get('runs_scored', 0)} runs, "
            f"SR: {batting.get('strike_rate', 0):.2f}, "
            f"Avg: {batting.get('average', 0):.2f}"
        )

        bowling_stats = (
            f"Bowling: {bowling.get('matches_played', 0)} matches, "
            f"{bowling.get('wickets', 0)} wickets, "
            f"Econ: {bowling.get('economy', 0):.2f}"
        )

        return (
            f"Player {player_data['player_name']} plays for {player_data['team']}. "
            f"{batting_stats}. {bowling_stats}."
        )

    def create_analysis_document(self, analysis_data: Dict) -> str:
        """Create a document from analysis data for embedding."""
        if analysis_data.get("type") == "venue_stats":
            team = analysis_data.get("team", "")
            venue_data = analysis_data.get("data", {})

            # Create a document for each venue
            venue_docs = []
            for venue, stats in venue_data.items():
                batting = stats.get("batting", {})
                bowling = stats.get("bowling", {})

                venue_doc = (
                    f"Venue: {venue}. Team: {team}. "
                    f"Batting: {batting.get('matches_played', 0)} matches, "
                    f"{batting.get('total_runs', 0)} runs. "
                    f"Bowling: {bowling.get('matches_played', 0)} matches, "
                    f"{bowling.get('wickets_taken', 0)} wickets."
                )
                venue_docs.append(venue_doc)

            return " ".join(venue_docs)

        elif analysis_data.get("type") == "h2h_stats":
            team1 = analysis_data.get("team1", "")
            team2 = analysis_data.get("team2", "")
            h2h_data = analysis_data.get("data", {})

            team1_stats = h2h_data.get(team1, {})
            team2_stats = h2h_data.get(team2, {})

            team1_batting = team1_stats.get("batting", {})
            team1_bowling = team1_stats.get("bowling", {})
            team2_batting = team2_stats.get("batting", {})
            team2_bowling = team2_stats.get("bowling", {})

            return (
                f"Head-to-head between {team1} and {team2}. "
                f"{team1}: {team1_batting.get('matches_played', 0)} matches, "
                f"{team1_batting.get('total_runs', 0)} runs, "
                f"{team1_bowling.get('wickets_taken', 0)} wickets. "
                f"{team2}: {team2_batting.get('matches_played', 0)} matches, "
                f"{team2_batting.get('total_runs', 0)} runs, "
                f"{team2_bowling.get('wickets_taken', 0)} wickets."
            )

        return "Unknown analysis type"
