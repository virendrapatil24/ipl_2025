import re
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...config import settings
from ...data_processing.data_loader import DataLoader
from ...data_processing.feature_engineering import FeatureEngineering
from ...data_processing.query_standardizer import standardize_query
from ...llm.factory import LLMFactory
from ...rag.retriever import RAGRetriever
from ...utils.logger import logger

router = APIRouter()
retriever = RAGRetriever()
llm_factory = LLMFactory()
data_loader = DataLoader()
feature_engineering = FeatureEngineering(data_loader=data_loader)


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str
    model: str = "gpt-3.5-turbo"


class ChatResponse(BaseModel):
    """Chat response model."""

    response: str


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
        logger.info(f"Standardized info: {standardized_info}")

        if not standardized_info["team1"] or not standardized_info["team2"]:
            raise HTTPException(
                status_code=400,
                detail="Could not identify teams from the query. Please specify two teams.",
            )

        team1 = standardized_info["team1"]
        team2 = standardized_info["team2"]
        venue = standardized_info["venue"]
        pitch_report = standardized_info["pitch_report"]
        logger.info(
            f"Team1: {team1}, Team2: {team2}, Venue: {venue}, Pitch Report: {pitch_report}"
        )

        # Get LLM instance
        model = request.model or settings.default_model
        llm = llm_factory.create_llm(model)

        context = retriever.get_relevant_context(
            request.message, team1, team2, venue, pitch_report
        )

        formatted_context = retriever.format_context(context)

        summarize_prompt = retriever.generate_summarize_prompt(formatted_context)

        summarize_response = llm.invoke(summarize_prompt)

        # Generate prompt using RAG
        prompt = retriever.generate_prompt(
            request.message,
            context,
            summarize_response,
            team1,
            team2,
            venue,
            pitch_report,
        )

        # Generate response
        response = llm.invoke(prompt)

        return ChatResponse(response=response)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
