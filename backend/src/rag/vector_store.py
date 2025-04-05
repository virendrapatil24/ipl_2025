from pathlib import Path
from typing import Dict, List, Optional

import chromadb
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
                ids.append(f"match_{match['id']}")

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
                metadatas.append(player)
                ids.append(f"player_{player['name'].replace(' ', '_')}")

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
                metadatas.append(analysis)
                ids.append(f"analysis_{analysis.get('id', len(ids))}")

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
