import json
from typing import Dict, List, Optional

import chromadb
import pandas as pd
from chromadb.config import Settings as ChromaSettings

from ..config.settings import settings
from ..utils.logger import logger
from .embeddings import EmbeddingGenerator


class VectorStore:
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
    ):
        """Initialize the vector store."""
        self.persist_directory = persist_directory or str(settings.vector_store_dir)
        logger.info(
            f"Initializing vector store with persist directory: {self.persist_directory}"
        )

        self.client = chromadb.Client(
            ChromaSettings(
                persist_directory=self.persist_directory, anonymized_telemetry=False
            )
        )
        self.embedding_generator = embedding_generator or EmbeddingGenerator()

        # Create collections for different types of data
        self.matches_collection = self.client.get_or_create_collection(
            name="matches", metadata={"description": "Match data and statistics"}
        )
        self.players_collection = self.client.get_or_create_collection(
            name="players",
            metadata={"description": "Player information and statistics"},
        )
        self.analysis_collection = self.client.get_or_create_collection(
            name="analysis", metadata={"description": "Match analysis and insights"}
        )

        # Check if collections are empty and populate if needed
        self._check_and_populate_collections()

    def _check_and_populate_collections(self):
        """Check if collections are empty and populate them if needed."""
        try:
            # Check matches collection
            matches_count = self.matches_collection.count()
            logger.info(f"Matches collection count: {matches_count}")
            if matches_count == 0:
                logger.info("Matches collection is empty. Populating with data...")
                self._populate_matches_collection()

            # Check players collection
            players_count = self.players_collection.count()
            logger.info(f"Players collection count: {players_count}")
            if players_count == 0:
                logger.info("Players collection is empty. Populating with data...")
                self._populate_players_collection()

            # Check analysis collection
            analysis_count = self.analysis_collection.count()
            logger.info(f"Analysis collection count: {analysis_count}")
            if analysis_count == 0:
                logger.info("Analysis collection is empty. Populating with data...")
                self._populate_analysis_collection()

        except Exception as e:
            logger.error(f"Error checking and populating collections: {e}")

    def _populate_matches_collection(self):
        """Populate the matches collection with data from matches.csv."""
        try:
            # Load matches data
            matches_file = (
                settings.data_dir / "cleaned_data" / "matches_data" / "matches.csv"
            )
            logger.info(f"Loading matches data from: {matches_file}")

            if not matches_file.exists():
                logger.warning(f"Matches file not found at {matches_file}")
                return

            matches_df = pd.read_csv(matches_file)
            logger.info(f"Loaded {len(matches_df)} matches from CSV")

            if matches_df.empty:
                logger.warning("Matches data is empty")
                return

            # Convert DataFrame to list of dictionaries
            matches_data = matches_df.to_dict(orient="records")
            logger.info(f"Converted {len(matches_data)} matches to dictionary format")

            # Add to vector store
            self.add_match_data(matches_data)
            logger.info(
                f"Successfully populated matches collection with "
                f"{len(matches_data)} records"
            )

        except Exception as e:
            logger.error(f"Error populating matches collection: {e}")

    def _populate_players_collection(self):
        """Populate the players collection with data from processed player stats."""
        try:
            # Update path to point to backend/processed_data
            player_stats_dir = (
                settings.data_dir.parent / "processed_data" / "player_stats"
            )
            logger.info(f"Loading player stats from: {player_stats_dir}")

            if not player_stats_dir.exists():
                logger.warning(
                    f"Player stats directory not found at {player_stats_dir}"
                )
                return

            players_data = []
            for player_file in player_stats_dir.glob("*.json"):
                try:
                    with open(player_file, "r") as f:
                        player_data = json.load(f)
                        players_data.append(player_data)
                except Exception as e:
                    logger.error(f"Error loading player file {player_file}: {e}")

            logger.info(f"Loaded {len(players_data)} player stats files")

            if not players_data:
                logger.warning("No player data found in processed files")
                return

            # Add to vector store
            self.add_player_data(players_data)
            logger.info(
                f"Successfully populated players collection with "
                f"{len(players_data)} records"
            )

        except Exception as e:
            logger.error(f"Error populating players collection: {e}")

    def _populate_analysis_collection(self):
        """Populate the analysis collection with pre-computed analysis data."""
        try:
            analysis_data = []

            # Load venue statistics with updated path
            venue_stats_dir = (
                settings.data_dir.parent / "processed_data" / "venue_stats"
            )
            logger.info(f"Loading venue stats from: {venue_stats_dir}")

            if venue_stats_dir.exists():
                for venue_file in venue_stats_dir.glob("*.json"):
                    try:
                        with open(venue_file, "r") as f:
                            venue_data = json.load(f)
                            analysis_data.append(
                                {
                                    "type": "venue_stats",
                                    "team": venue_file.stem.replace("_venue_stats", ""),
                                    "data": venue_data,
                                }
                            )
                    except Exception as e:
                        logger.error(f"Error loading venue file {venue_file}: {e}")

            # Load head-to-head statistics with updated path
            h2h_stats_dir = settings.data_dir.parent / "processed_data" / "h2h_stats"
            logger.info(f"Loading h2h stats from: {h2h_stats_dir}")

            if h2h_stats_dir.exists():
                for h2h_file in h2h_stats_dir.glob("*.json"):
                    try:
                        with open(h2h_file, "r") as f:
                            h2h_data = json.load(f)
                            teams = h2h_file.stem.replace("_h2h_stats", "").split(
                                "_vs_"
                            )
                            analysis_data.append(
                                {
                                    "type": "h2h_stats",
                                    "team1": teams[0],
                                    "team2": teams[1],
                                    "data": h2h_data,
                                }
                            )
                    except Exception as e:
                        logger.error(f"Error loading h2h file {h2h_file}: {e}")

            logger.info(f"Loaded {len(analysis_data)} analysis files")

            if not analysis_data:
                logger.warning("No analysis data found in processed files")
                return

            # Add to vector store
            self.add_analysis_data(analysis_data)
            logger.info(
                f"Successfully populated analysis collection with "
                f"{len(analysis_data)} records"
            )

        except Exception as e:
            logger.error(f"Error populating analysis collection: {e}")

    def add_match_data(self, matches_data: List[Dict]):
        """Add match data to the vector store."""
        try:
            documents = []
            metadatas = []
            ids = []

            for match in matches_data:
                doc = self.embedding_generator.create_match_document(match)
                documents.append(doc)
                metadatas.append(match)
                ids.append(f"match_{match['match_id']}")

            self.matches_collection.add(
                documents=documents, metadatas=metadatas, ids=ids
            )
            logger.info(f"Added {len(documents)} matches to vector store")
        except Exception as e:
            logger.error(f"Error adding match data: {e}")
            raise

    def add_player_data(self, players_data: List[Dict]):
        """Add player data to the vector store."""
        try:
            documents = []
            metadatas = []
            ids = []

            for player in players_data:
                doc = self.embedding_generator.create_player_document(player)
                documents.append(doc)

                # Convert player data to a string representation for metadata
                # This ensures ChromaDB can store it properly
                player_name = player.get("player_name", "unknown")
                player_id = f"player_{player_name.replace(' ', '_')}"

                # Create a simplified metadata object with only string values
                metadata = {
                    "player_name": player_name,
                    "player_id": player_id,
                    "stats_summary": f"Matches: {player.get('matches_played', 0)}, "
                    f"Runs: {player.get('runs_scored', 0)}, "
                    f"Avg: {player.get('average', 0):.2f}",
                }

                metadatas.append(metadata)
                ids.append(player_id)

            self.players_collection.add(
                documents=documents, metadatas=metadatas, ids=ids
            )
            logger.info(f"Added {len(documents)} players to vector store")
        except Exception as e:
            logger.error(f"Error adding player data: {e}")
            raise

    def add_analysis_data(self, analysis_data: List[Dict]):
        """Add analysis data to the vector store."""
        try:
            documents = []
            metadatas = []
            ids = []

            for analysis in analysis_data:
                doc = self.embedding_generator.create_analysis_document(analysis)
                documents.append(doc)

                # Create a simplified metadata object with only string values
                analysis_type = analysis.get("type", "unknown")
                analysis_id = f"analysis_{analysis_type}_{len(ids)}"

                # Create a simplified metadata object based on analysis type
                if analysis_type == "venue_stats":
                    team = analysis.get("team", "unknown")
                    metadata = {
                        "type": analysis_type,
                        "team": team,
                        "summary": f"Venue statistics for {team}",
                    }
                elif analysis_type == "h2h_stats":
                    team1 = analysis.get("team1", "unknown")
                    team2 = analysis.get("team2", "unknown")
                    metadata = {
                        "type": analysis_type,
                        "team1": team1,
                        "team2": team2,
                        "summary": f"Head-to-head statistics for {team1} vs {team2}",
                    }
                else:
                    metadata = {
                        "type": analysis_type,
                        "summary": f"Analysis of type {analysis_type}",
                    }

                metadatas.append(metadata)
                ids.append(analysis_id)

            self.analysis_collection.add(
                documents=documents, metadatas=metadatas, ids=ids
            )
            logger.info(f"Added {len(documents)} analysis documents to vector store")
        except Exception as e:
            logger.error(f"Error adding analysis data: {e}")
            raise

    def query_matches(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query match data from vector store."""
        try:
            results = self.matches_collection.query(
                query_texts=[query], n_results=n_results
            )
            print("results", results)

            # Handle empty results
            if not results["documents"][0]:
                logger.warning(f"No matches found for query: {query}")
                return []

            return [
                {"document": doc, "metadata": meta, "distance": dist}
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ]
        except Exception as e:
            logger.error(f"Error querying matches: {e}")
            raise

    def query_players(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query player data from vector store."""
        try:
            results = self.players_collection.query(
                query_texts=[query], n_results=n_results
            )

            # Handle empty results
            if not results["documents"][0]:
                logger.warning(f"No players found for query: {query}")
                return []

            return [
                {"document": doc, "metadata": meta, "distance": dist}
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ]
        except Exception as e:
            logger.error(f"Error querying players: {e}")
            raise

    def query_analysis(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query analysis data from vector store."""
        try:
            results = self.analysis_collection.query(
                query_texts=[query], n_results=n_results
            )

            # Handle empty results
            if not results["documents"][0]:
                logger.warning(f"No analysis found for query: {query}")
                return []

            return [
                {"document": doc, "metadata": meta, "distance": dist}
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ]
        except Exception as e:
            logger.error(f"Error querying analysis: {e}")
            raise
