import json
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions
from langchain.schema import Document

from ..config.settings import settings
from ..utils.logger import logger


class VectorStore:
    def __init__(
        self,
        persist_directory: Optional[str] = None,
    ):
        """Initialize the vector store."""
        self.persist_directory = persist_directory or str(settings.vector_store_dir)
        logger.info(
            "Initializing vector store with persist directory: "
            f"{self.persist_directory}"
        )

        # Initialize the embedding function using sentence-transformers
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )

        # Create a single collection for all IPL data
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.ipl_collection = self.client.get_or_create_collection(
            name="ipl_data",
            metadata={"description": "IPL match data, statistics and analysis"},
            embedding_function=self.embedding_function,
        )

        # Check if collection is empty and populate if needed
        self._check_and_populate_collection()

    def _check_and_populate_collection(self):
        """Check if collection is empty and populate it if needed."""
        try:
            count = self.ipl_collection.count()
            logger.info(f"IPL collection count: {count}")
            if count == 0:
                logger.info("IPL collection is empty. Populating with data...")
                self._populate_ipl_collection()
        except Exception as e:
            logger.error(f"Error checking/populating collection: {str(e)}")
            raise

    def _load_json_file(self, file_path: Path) -> Dict:
        """Load and parse a JSON file."""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {str(e)}")
            return {}

    def _process_venue_stats(self, venue_stats_dir: Path) -> List[Document]:
        """Process venue statistics from JSON files."""
        documents = []
        for file_path in venue_stats_dir.glob("*.json"):
            data = self._load_json_file(file_path)
            if not data:
                continue

            # Clean venue name by removing _venue_stats suffix
            venue_name = file_path.stem.replace("_venue_stats", "")

            # Calculate win percentages
            total_matches = data.get("total_matches", 0)
            batting_first_wins = data.get("batting_first_wins", 0)
            batting_second_wins = data.get("batting_second_wins", 0)

            batting_first_win_pct = (
                (batting_first_wins / total_matches * 100) if total_matches > 0 else 0
            )
            batting_second_win_pct = (
                (batting_second_wins / total_matches * 100) if total_matches > 0 else 0
            )

            # Format content with correct key names
            content = (
                f"Venue {venue_name} statistics: "
                f"Total matches played: {total_matches}, "
                f"Batting first wins: {batting_first_wins} "
                f"({batting_first_win_pct:.1f}%), "
                f"Batting second wins: {batting_second_wins} "
                f"({batting_second_win_pct:.1f}%), "
                f"Average first innings runs: "
                f"{data.get('avg_first_innings_runs', 'N/A')}, "
                f"Average second innings runs: "
                f"{data.get('avg_second_innings_runs', 'N/A')}, "
                f"Average first innings wickets: "
                f"{data.get('avg_first_innings_wickets', 'N/A')}, "
                f"Average second innings wickets: "
                f"{data.get('avg_second_innings_wickets', 'N/A')}"
            )

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "type": "venue_stats",
                        "venue": venue_name,  # Clean venue name without suffix
                    },
                )
            )
        return documents

    def _process_team_stats(self, team_stats_dir: Path) -> List[Document]:
        """Process team head-to-head statistics."""
        documents = []
        for file_path in team_stats_dir.glob("*.json"):
            data = self._load_json_file(file_path)
            if not data:
                continue

            team_name = file_path.stem.replace("_h2h_stats", "")
            for opponent, stats in data.items():
                # Process recent form (last 4 matches)
                recent_matches = stats.get("recent_form", [])
                recent_form = []
                for match in recent_matches[:4]:  # Get last 4 matches
                    result = "Won" if match.get("winner") == team_name else "Lost"
                    date = match.get("date", "Unknown")
                    margin = match.get("result", "Unknown")
                    recent_form.append(f"{date}: {result} by {margin}")

                # Format recent form text
                recent_form_text = (
                    "Recent form (last 4 matches): " + "; ".join(recent_form)
                    if recent_form
                    else "No recent matches data available"
                )

                content = (
                    f"Head-to-head stats between {team_name} and {opponent}: "
                    f"Matches played: {stats.get('matches_played', 'N/A')}, "
                    f"{team_name} wins: "
                    f"{stats.get(f'{team_name}_wins', 'N/A')}, "
                    f"{opponent} wins: "
                    f"{stats.get(f'{opponent}_wins', 'N/A')}. "
                    f"{recent_form_text}"
                )

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "type": "team_h2h",
                            "team1": team_name,
                            "team2": opponent,
                        },
                    )
                )
        return documents

    def _process_player_stats(self, player_stats_dir: Path) -> List[Document]:
        """Process all types of player statistics."""
        documents = []

        # Process player vs player stats
        player_vs_player_dir = player_stats_dir / "player_vs_player_stats"
        for file_path in player_vs_player_dir.glob("*.json"):
            data = self._load_json_file(file_path)
            if not data:
                continue

            player_name = file_path.stem
            for opponent, stats in data.items():
                batting = stats.get("batting", {})
                bowling = stats.get("bowling", {})

                content = (
                    f"Player {player_name} vs {opponent} stats: "
                    f"Batting - Matches: {batting.get('matches', 'N/A')}, "
                    f"Runs: {batting.get('runs', 'N/A')}, "
                    f"Strike Rate: {batting.get('strike_rate', 'N/A')}, "
                    f"Average: {batting.get('average', 'N/A')}, "
                    f"50s: {batting.get('50s', 'N/A')}, "
                    f"100s: {batting.get('100s', 'N/A')}; "
                    f"Bowling - Matches: {bowling.get('matches', 'N/A')}, "
                    f"Wickets: {bowling.get('wickets', 'N/A')}, "
                    f"Economy: {bowling.get('economy', 'N/A')}, "
                    f"Best: {bowling.get('best_bowling', 'N/A')}"
                )

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "type": "player_vs_player",
                            "player": player_name,
                            "opponent": opponent,
                        },
                    )
                )

        # Process player vs team stats
        player_vs_team_dir = player_stats_dir / "player_vs_team_stats"
        for file_path in player_vs_team_dir.glob("*.json"):
            data = self._load_json_file(file_path)
            if not data:
                continue

            player_name = file_path.stem
            for team, stats in data.items():
                batting = stats.get("batting", {})
                bowling = stats.get("bowling", {})

                content = (
                    f"Player {player_name} vs {team} stats: "
                    f"Batting - Matches: {batting.get('matches', 'N/A')}, "
                    f"Runs: {batting.get('runs', 'N/A')}, "
                    f"Strike Rate: {batting.get('strike_rate', 'N/A')}, "
                    f"Average: {batting.get('average', 'N/A')}, "
                    f"50s: {batting.get('50s', 'N/A')}, "
                    f"100s: {batting.get('100s', 'N/A')}; "
                    f"Bowling - Matches: {bowling.get('matches', 'N/A')}, "
                    f"Wickets: {bowling.get('wickets', 'N/A')}, "
                    f"Economy: {bowling.get('economy', 'N/A')}, "
                    f"Best: {bowling.get('best_bowling', 'N/A')}"
                )

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "type": "player_vs_team",
                            "player": player_name,
                            "team": team,
                        },
                    )
                )

        # Process player venue stats
        player_venue_dir = player_stats_dir / "player_venue_stats"
        for file_path in player_venue_dir.glob("*.json"):
            data = self._load_json_file(file_path)
            if not data:
                continue

            player_name = file_path.stem
            for venue, stats in data.items():
                batting = stats.get("batting", {})
                bowling = stats.get("bowling", {})

                content = (
                    f"Player {player_name} at {venue} stats: "
                    f"Batting - Matches: {batting.get('matches', 'N/A')}, "
                    f"Runs: {batting.get('runs', 'N/A')}, "
                    f"Strike Rate: {batting.get('strike_rate', 'N/A')}, "
                    f"Average: {batting.get('average', 'N/A')}, "
                    f"50s: {batting.get('50s', 'N/A')}, "
                    f"100s: {batting.get('100s', 'N/A')}; "
                    f"Bowling - Matches: {bowling.get('matches', 'N/A')}, "
                    f"Wickets: {bowling.get('wickets', 'N/A')}, "
                    f"Economy: {bowling.get('economy', 'N/A')}, "
                    f"Best: {bowling.get('best_bowling', 'N/A')}"
                )

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "type": "player_venue",
                            "player": player_name,
                            "venue": venue,
                        },
                    )
                )

        # Process player all-time stats
        player_all_time_dir = player_stats_dir / "player_all_time_stats"
        for file_path in player_all_time_dir.glob("*.json"):
            data = self._load_json_file(file_path)
            if not data:
                continue

            player_name = data.get("player_name", file_path.stem)
            batting = data.get("batting", {})
            bowling = data.get("bowling", {})

            content = (
                f"Player {player_name} all-time stats: "
                f"Batting - Matches: {batting.get('matches', 'N/A')}, "
                f"Runs: {batting.get('runs', 'N/A')}, "
                f"Strike Rate: {batting.get('strike_rate', 'N/A')}, "
                f"Average: {batting.get('average', 'N/A')}, "
                f"50s: {batting.get('50s', 'N/A')}, "
                f"100s: {batting.get('100s', 'N/A')}; "
                f"Bowling - Matches: {bowling.get('matches', 'N/A')}, "
                f"Wickets: {bowling.get('wickets', 'N/A')}, "
                f"Economy: {bowling.get('economy', 'N/A')}, "
                f"Best: {bowling.get('best_bowling', 'N/A')}"
            )

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "type": "player_all_time",
                        "player": player_name,
                    },
                )
            )

        return documents

    def _process_team_venue_stats(self, team_venue_stats_dir: Path) -> List[Document]:
        """Process team performance statistics at specific venues."""
        documents = []
        for file_path in team_venue_stats_dir.glob("*.json"):
            data = self._load_json_file(file_path)
            if not data:
                continue

            team_name = file_path.stem.replace("_at_venue_stats", "")
            for venue, stats in data.items():
                # Calculate win percentages
                total_matches = stats.get("total_matches", 0)
                batting_first_wins = stats.get("batting_first_wins", 0)
                batting_second_wins = stats.get("batting_second_wins", 0)

                batting_first_win_pct = (
                    (batting_first_wins / total_matches * 100)
                    if total_matches > 0
                    else 0
                )
                batting_second_win_pct = (
                    (batting_second_wins / total_matches * 100)
                    if total_matches > 0
                    else 0
                )

                content = (
                    f"Team {team_name} performance at {venue}: "
                    f"Total matches: {total_matches}, "
                    f"Batting first wins: {batting_first_wins} "
                    f"({batting_first_win_pct:.1f}%), "
                    f"Batting second wins: {batting_second_wins} "
                    f"({batting_second_win_pct:.1f}%), "
                    f"Average first innings runs: "
                    f"{stats.get('avg_first_innings_runs', 'N/A')}, "
                    f"Average second innings runs: "
                    f"{stats.get('avg_second_innings_runs', 'N/A')}, "
                    f"Average first innings wickets: "
                    f"{stats.get('avg_first_innings_wickets', 'N/A')}, "
                    f"Average second innings wickets: "
                    f"{stats.get('avg_second_innings_wickets', 'N/A')}"
                )

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "type": "team_venue",
                            "team": team_name,
                            "venue": venue,
                        },
                    )
                )
        return documents

    def _populate_ipl_collection(self):
        """Populate the IPL collection with all available data."""
        processed_data_dir = Path(settings.processed_data_dir)

        # Process venue statistics
        venue_stats = self._process_venue_stats(processed_data_dir / "venue_stats")

        # Process team head-to-head statistics
        team_stats = self._process_team_stats(processed_data_dir / "team_h2h_stats")

        # Process team venue statistics
        team_venue_stats = self._process_team_venue_stats(
            processed_data_dir / "team_at_venue_stats"
        )

        # Process all player statistics
        player_stats = self._process_player_stats(processed_data_dir)

        # Combine all documents
        all_documents = venue_stats + team_stats + team_venue_stats + player_stats

        # Add documents to collection
        for i, doc in enumerate(all_documents):
            self.ipl_collection.add(
                documents=[doc.page_content],
                metadatas=[doc.metadata],
                ids=[f"doc_{i}"],
            )

        logger.info(
            "Successfully populated IPL collection with "
            f"{len(all_documents)} documents"
        )

    def similarity_search(
        self,
        query: str,
        filter_dict: Optional[Dict] = None,
        n_results: int = 5,
    ) -> List[Dict]:
        """
        Perform a similarity search with optional filtering.

        Args:
            query: The search query
            filter_dict: Optional dictionary of metadata filters
            n_results: Number of results to return

        Returns:
            List of dictionaries containing the search results
        """
        try:
            results = self.ipl_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_dict,
            )

            return [
                {
                    "content": doc,
                    "metadata": meta,
                }
                for doc, meta in zip(results["documents"][0], results["metadatas"][0])
            ]
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            return []
