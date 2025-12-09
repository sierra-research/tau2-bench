from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field
from tau2.environment.db import DB


class Address(BaseModel):
    """Represents a physical address"""
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    country: str = Field(description="Country name")
    state: str = Field(description="State or province name")
    zip: str = Field(description="Postal code")


class PlayerName(BaseModel):
    """Represents a player's full name"""
    first_name: str = Field(description="Player's first name")
    last_name: str = Field(description="Player's last name")


class PlayerProfile(BaseModel):
    """Represents a player's profile details"""
    position: str = Field(description="Player position (e.g., G, F, C, G/F)")
    number: Union[int, str] = Field(description="Jersey number")
    height_cm: float = Field(description="Player's height in centimeters")
    weight_kg: float = Field(description="Player's weight in kilograms")


class ClubArena(BaseModel):
    """Represents a club's arena"""
    name: str = Field(description="Arena name")
    address: Address = Field(description="Arena address")


class RosterEntry(BaseModel):
    """Represents a roster entry for a club"""
    player_id: str = Field(description="Unique identifier for the player")
    profile: PlayerProfile = Field(description="Player profile information")
    active: bool = Field(description="Whether the player is active")
    salary: float = Field(description="Player salary under current club (per season)")


class Club(BaseModel):
    """Represents a basketball club"""
    name: str = Field(description="Club name")
    club_id: str = Field(description="Unique identifier for the club")
    league: str = Field(description="League in which the club competes")
    city: str = Field(description="Home city of the club")
    arena: ClubArena = Field(description="Club's home arena")
    roster: Dict[str, RosterEntry] = Field(description="Dictionary of roster entries indexed by player ID")
    games: List[str] = Field(description="List of game IDs associated with this club")


ContractSource = Literal["draft", "trade", "free agency"]


class PlayerContractExtraInfo(BaseModel):
    """Represents extra information for a player's contract"""
    salary: float = Field(description="Contract salary per year")
    years: int = Field(description="Contract duration in years")


class PlayerContract(BaseModel):
    """Represents a player's contract"""
    source: ContractSource = Field(description="How the player joined the club (draft, trade, or free agency)")
    contract_id: str = Field(description="Unique identifier for the contract")
    club_id: str = Field(description="Associated club ID")
    start_date: str = Field(description="Contract start date (ISO-8601 or similar)")
    end_date: str = Field(description="Contract end date (ISO-8601 or similar)")
    extra_info: PlayerContractExtraInfo = Field(description="Additional contract details")


class Player(BaseModel):
    """Represents a basketball player"""
    player_id: str = Field(description="Unique identifier for the player")
    name: PlayerName = Field(description="Player's full name")
    address: Address = Field(description="Player's address")
    email: str = Field(description="Player's email address")
    contracts: Dict[str, PlayerContract] = Field(description="Dictionary of contracts indexed by contract ID")
    games: List[str] = Field(description="List of game IDs the player has participated in")


GameStatus = Literal["scheduled", "in_progress", "final", "postponed"]


class GameClubEntry(BaseModel):
    """Represents a participating club in a game with score"""
    name: str = Field(description="Club name")
    club_id: str = Field(description="Unique identifier for the club")
    score: int = Field(description="Total points scored by the club")


class GameClubs(BaseModel):
    """Represents home and away clubs for a game"""
    home: GameClubEntry = Field(description="Home club details")
    away: GameClubEntry = Field(description="Away club details")


class BoxScoreOptions(BaseModel):
    """Represents additional box score statistics"""
    rebounds: int = Field(description="Number of rebounds")
    assists: int = Field(description="Number of assists")
    minutes: str = Field(description="Minutes played (e.g., '34:12')")


class BoxScoreEntry(BaseModel):
    """Represents a single box score entry for a player in a game"""
    name: str = Field(description="Player's name")
    club_id: str = Field(description="Club ID for which the player played")
    player_id: str = Field(description="Player ID")
    points: int = Field(description="Points scored by the player")
    options: BoxScoreOptions = Field(description="Additional player statistics")


class GamePeriod(BaseModel):
    """Represents a period within a game"""
    period_number: int = Field(description="Period number (e.g., 1-4, or more for OT)")
    event_ids: List[str] = Field(description="List of event IDs that occurred in the period")
    player_ids: List[str] = Field(description="List of player IDs that participated in the period")


ScoringEventType = Literal['1PT', "2PT", "3PT", "FT"]


class GameScoringEvent(BaseModel):
    """Represents a scoring event in a game"""
    event_type: ScoringEventType = Field(description="Type of scoring event (1PT, 2PT, 3PT, FT)")
    points: int = Field(description="Points scored for the event")
    club_id: str = Field(description="Club credited with the score")
    player_id: str = Field(description="Player credited with the score")
    timestamp: str = Field(description="Timestamp of the event within the game")


class Game(BaseModel):
    """Represents a basketball game"""
    game_id: str = Field(description="Unique identifier for the game")
    season: str = Field(description="Season identifier (e.g., '2024-25')")
    date: str = Field(description="Date of the game (ISO-8601 or similar)")
    status: GameStatus = Field(description="Current status of the game")
    venue: Address = Field(description="Venue address where the game is played")
    clubs: GameClubs = Field(description="Home and away club details")
    box_score: List[BoxScoreEntry] = Field(description="Box score entries for the game")
    periods: List[GamePeriod] = Field(description="List of periods and associated data")
    scoring_history: List[GameScoringEvent] = Field(description="Chronological list of scoring events")


class BasketballDB(DB):
    """Database containing all basketball-related data including clubs, players and games"""
    clubs: Dict[str, Club] = Field(description="Dictionary of all clubs indexed by club ID")
    players: Dict[str, Player] = Field(description="Dictionary of all players indexed by player ID")
    games: Dict[str, Game] = Field(description="Dictionary of all games indexed by game ID")

    def get_statistics(self) -> dict[str, Any]:
        """Get the statistics of the database."""
        num_clubs = len(self.clubs)
        num_players = len(self.players)
        num_games = len(self.games)
        total_roster_size = sum(len(club.roster) for club in self.clubs.values())
        return {
            "num_clubs": num_clubs,
            "num_players": num_players,
            "num_games": num_games,
            "total_roster_size": total_roster_size,
        }