import re
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...config import settings
from ...data_processing.data_loader import DataLoader
from ...data_processing.feature_engineering import FeatureEngineering
from ...data_processing.player_stats_processor import PlayerStatsProcessor
from ...data_processing.query_standardizer import standardize_query
from ...llm.factory import LLMFactory
from ...rag.retriever import RAGRetriever
from ...utils.logger import logger

router = APIRouter()
retriever = RAGRetriever()
llm_factory = LLMFactory()
data_loader = DataLoader()
feature_engineering = FeatureEngineering(data_loader=data_loader)
player_stats_processor = PlayerStatsProcessor()


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str
    model: str = "gpt-3.5-turbo"


class MatchPrediction(BaseModel):
    team: str
    win_probability: float
    key_players: List[str]


class Analysis(BaseModel):
    venue_stats: Dict
    h2h_stats: Dict
    team1_analysis: Dict
    team2_analysis: Dict
    pitch_analysis: Optional[Dict] = None


class ChatResponse(BaseModel):
    response: str
    analysis: Optional[Analysis] = None
    predictions: Optional[MatchPrediction] = None


def extract_match_info(msg: str) -> tuple[str, str, str]:
    """Extract team names and venue from the message."""
    pattern = r"([A-Za-z\s]+)\s+vs\s+" r"([A-Za-z\s]+)\s+at\s+" r"([A-Za-z\s]+)"
    match = re.search(pattern, msg)
    if not match:
        err = (
            "Could not extract match information. "
            "Use format: 'Team1 vs Team2 at Venue'"
        )
        raise ValueError(err)
    return (match.group(1).strip(), match.group(2).strip(), match.group(3).strip())


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Process chat requests and generate responses."""
    try:
        # Standardize the query to extract team and venue information
        standardized_info = standardize_query(request.message, model_name=request.model)
        print("standardized_info", standardized_info)

        if not standardized_info["team1"] or not standardized_info["team2"]:
            raise HTTPException(
                status_code=400,
                detail="Could not identify teams from the query. Please specify two teams.",
            )

        team1 = standardized_info["team1"]
        team2 = standardized_info["team2"]
        venue = standardized_info["venue"]
        print("team1", team1)
        print("team2", team2)
        print("venue", venue)

        # Load and process data
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>0")
        matches = data_loader.load_matches()
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>1")
        team1_squad = data_loader.load_squad_data(team1.replace(" ", "_"))
        team2_squad = data_loader.load_squad_data(team2.replace(" ", "_"))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>2")
        print("team1_squad", team1_squad)
        print("team2_squad", team2_squad)

        # Calculate statistics
        venue_stats = FeatureEngineering.calculate_venue_stats(matches, venue)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>3")
        h2h_stats = FeatureEngineering.calculate_h2h_stats(matches, team1, team2)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>4")
        # Get pre-computed player statistics
        team1_player_stats = {}
        team2_player_stats = {}

        # Process each player in team1
        for _, player in team1_squad.iterrows():
            player_name = player["Delivery Name"]
            player_stats = player_stats_processor.get_player_stats(player_name)
            if player_stats:
                team1_player_stats[player_name] = player_stats

        # Process each player in team2
        for _, player in team2_squad.iterrows():
            player_name = player["Delivery Name"]
            player_stats = player_stats_processor.get_player_stats(player_name)
            if player_stats:
                team2_player_stats[player_name] = player_stats

        print("venue_stats", venue_stats)
        print("h2h_stats", h2h_stats)
        # print("team1_player_stats", team1_player_stats)
        # print("team2_player_stats", team2_player_stats)

        # Generate prompt using RAG
        prompt = retriever.generate_prompt(request.message, team1, team2, venue)
        print("prompt", prompt)

        # Get LLM instance
        model = request.model or settings.default_model
        print("model", model)
        llm = llm_factory.create_llm(model)

        # Generate response
        response = llm.predict(prompt)

        return ChatResponse(
            response=response,
            confidence=0.95,  # Placeholder for now
            model=model,
            context={
                "venue_stats": venue_stats,
                "h2h_stats": h2h_stats,
                "team1_stats": team1_player_stats,
                "team2_stats": team2_player_stats,
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
