import re
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...config import settings
from ...data_processing.data_loader import DataLoader
from ...data_processing.feature_engineering import FeatureEngineering
from ...llm.factory import LLMFactory
from ...rag.retriever import RAGRetriever
from ...utils.logger import logger

router = APIRouter()
retriever = RAGRetriever()
llm_factory = LLMFactory()
data_loader = DataLoader()


class ChatRequest(BaseModel):
    message: str
    model: str


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
        # Extract match information
        team1, team2, venue = extract_match_info(request.message)

        # Load and process data
        matches = data_loader.load_matches()
        squads = data_loader.load_squads()

        # Calculate statistics
        venue_stats = FeatureEngineering.calculate_venue_stats(matches, venue)
        h2h_stats = FeatureEngineering.calculate_h2h_stats(matches, team1, team2)
        t1_stats = FeatureEngineering.calculate_player_stats(matches, squads, team1)
        t2_stats = FeatureEngineering.calculate_player_stats(matches, squads, team2)

        # Generate prompt using RAG
        prompt = retriever.generate_prompt(request.message, team1, team2, venue)

        # Get LLM instance
        model = request.model or settings.default_model
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
                "team1_stats": t1_stats,
                "team2_stats": t2_stats,
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
