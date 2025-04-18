import json
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

from ..config.settings import settings
from ..utils.logger import logger


class VectorStore:
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_model: Optional[OpenAIEmbeddings] = None,
    ):
        """Initialize the vector store."""
        self.persist_directory = persist_directory or str(settings.vector_store_dir)
        logger.info(
            "Initializing vector store with persist directory: "
            f"{self.persist_directory}"
        )

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.embedding_model = embedding_model or OpenAIEmbeddings()

        # Create a single collection for all IPL data
        self.ipl_collection = self.client.get_or_create_collection(
            name="ipl_data",
            metadata={"description": "IPL match data, statistics and analysis"},
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

            venue_name = file_path.stem
            content = (
                f"Venue {venue_name} statistics: "
                f"Average first innings score: "
                f"{data.get('avg_first_innings_score', 'N/A')}, "
                f"Average second innings score: "
                f"{data.get('avg_second_innings_score', 'N/A')}, "
                f"Average wickets per match: "
                f"{data.get('avg_wickets_per_match', 'N/A')}"
            )

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "type": "venue_stats",
                        "venue": venue_name,
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

            team_name = file_path.stem
            for opponent, stats in data.items():
                content = (
                    f"Head-to-head stats between {team_name} and {opponent}: "
                    f"Matches played: {stats.get('matches_played', 'N/A')}, "
                    f"{team_name} wins: {stats.get('team_wins', 'N/A')}, "
                    f"{opponent} wins: {stats.get('opponent_wins', 'N/A')}"
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
        """Process player statistics."""
        documents = []
        for file_path in player_stats_dir.glob("*.json"):
            data = self._load_json_file(file_path)
            if not data:
                continue

            player_name = file_path.stem
            # Process player vs player stats
            if "vs_player" in data:
                for opponent, stats in data["vs_player"].items():
                    content = (
                        f"Player {player_name} vs {opponent} stats: "
                        f"Matches: {stats.get('matches', 'N/A')}, "
                        f"Runs: {stats.get('runs', 'N/A')}, "
                        f"Strike Rate: {stats.get('strike_rate', 'N/A')}"
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

            # Process player venue stats
            if "venue_stats" in data:
                for venue, stats in data["venue_stats"].items():
                    content = (
                        f"Player {player_name} at {venue} stats: "
                        f"Matches: {stats.get('matches', 'N/A')}, "
                        f"Runs: {stats.get('runs', 'N/A')}, "
                        f"Strike Rate: {stats.get('strike_rate', 'N/A')}"
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
        return documents

    def _populate_ipl_collection(self):
        """Populate the IPL collection with all available data."""
        processed_data_dir = Path(settings.processed_data_dir)

        # Process venue statistics
        venue_stats = self._process_venue_stats(processed_data_dir / "venue_stats")

        # Process team head-to-head statistics
        team_stats = self._process_team_stats(processed_data_dir / "team_h2h_stats")

        # Process player statistics
        player_stats = self._process_player_stats(
            processed_data_dir / "player_vs_player_stats"
        )

        # Combine all documents
        all_documents = venue_stats + team_stats + player_stats

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
