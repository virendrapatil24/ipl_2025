import pytest
from fastapi.testclient import TestClient

from ..src.api.main import app
from ..src.api.schemas.request import ChatRequest
from ..src.config import settings

client = TestClient(app)


def test_chat_endpoint_success():
    """Test successful chat request processing."""
    msg = "MI vs CSK at Wankhede Stadium"
    req = ChatRequest(message=msg, model="gpt-3.5-turbo")

    response = client.post("/api/chat", json=req.model_dump())
    assert response.status_code == 200

    data = response.json()
    assert "response" in data
    assert "confidence" in data
    assert "model" in data
    assert "context" in data

    assert data["model"] == "gpt-3.5-turbo"
    assert isinstance(data["confidence"], float)
    assert 0 <= data["confidence"] <= 1

    context = data["context"]
    assert "venue_stats" in context
    assert "h2h_stats" in context
    assert "team1_stats" in context
    assert "team2_stats" in context


def test_chat_endpoint_invalid_format():
    """Test chat request with invalid message format."""
    req = ChatRequest(message="Invalid message format", model="gpt-3.5-turbo")

    response = client.post("/api/chat", json=req.model_dump())
    assert response.status_code == 400

    data = response.json()
    assert "detail" in data
    assert "format" in data["detail"].lower()


def test_chat_endpoint_unsupported_model():
    """Test chat request with unsupported model."""
    msg = "MI vs CSK at Wankhede Stadium"
    req = ChatRequest(message=msg, model="unsupported-model")

    response = client.post("/api/chat", json=req.model_dump())
    assert response.status_code == 400

    data = response.json()
    assert "detail" in data
    assert "not supported" in data["detail"].lower()


def test_chat_endpoint_default_model():
    """Test chat request without specifying a model."""
    msg = "MI vs CSK at Wankhede Stadium"
    req = ChatRequest(message=msg)

    response = client.post("/api/chat", json=req.model_dump())
    assert response.status_code == 200

    data = response.json()
    assert data["model"] == settings.default_model


@pytest.mark.parametrize(
    "message,expected_teams,expected_venue",
    [
        ("MI vs CSK at Wankhede Stadium", ("MI", "CSK", "Wankhede Stadium")),
        ("RCB vs CSK at Chinnaswamy", ("RCB", "CSK", "Chinnaswamy")),
    ],
)
def test_match_info_extraction(
    message: str, expected_teams: tuple[str, str, str], expected_venue: str
):
    """Test match information extraction from different message formats."""
    from ..src.api.routes.chat import extract_match_info

    team1, team2, venue = extract_match_info(message)
    assert team1 == expected_teams[0]
    assert team2 == expected_teams[1]
    assert venue == expected_teams[2]
