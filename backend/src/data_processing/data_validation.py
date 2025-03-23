from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class BattingStyle(str, Enum):
    RIGHT = "Right Hand Bat"
    LEFT = "Left Hand Bat"


class BowlingStyle(str, Enum):
    FAST = "Fast"
    MEDIUM = "Medium"
    SPIN = "Spin"
    OFF_SPIN = "Off Spin"
    LEG_SPIN = "Leg Spin"


class PlayerRole(str, Enum):
    BATSMAN = "Batsman"
    BOWLER = "Bowler"
    ALL_ROUNDER = "All-Rounder"
    WICKET_KEEPER = "Wicket-Keeper"


class Player(BaseModel):
    name: str
    role: PlayerRole
    batting_style: Optional[BattingStyle]
    bowling_style: Optional[BowlingStyle]
    team: str
    is_overseas: bool


class Match(BaseModel):
    id: int
    season: int
    city: str
    date: str
    venue: str
    team1: str
    team2: str
    toss_winner: str
    toss_decision: str
    winner: Optional[str]
    result: str
    result_margin: Optional[float]


class Delivery(BaseModel):
    match_id: int
    inning: int
    batting_team: str
    bowling_team: str
    over: int
    ball: int
    batter: str
    bowler: str
    non_striker: str
    batsman_runs: int
    extra_runs: int
    total_runs: int
    extras_type: Optional[str]
    is_wicket: bool
    player_dismissed: Optional[str]
    dismissal_kind: Optional[str]
    fielder: Optional[str]


def validate_player_data(data: dict) -> Player:
    """Validate player data against the Player model."""
    return Player(**data)


def validate_match_data(data: dict) -> Match:
    """Validate match data against the Match model."""
    return Match(**data)


def validate_delivery_data(data: dict) -> Delivery:
    """Validate delivery data against the Delivery model."""
    return Delivery(**data)
