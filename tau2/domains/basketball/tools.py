"""
Toolkit for the basketball domain, following the style of RetailTools.
"""

import json
from typing import Any, Dict, List, Optional

from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
from tau2.domains.basketball.data_model import (
    BasketballDB,
    BoxScoreEntry,
    Club,
    Game,
    GameScoringEvent,
    GameStatus,
    Player,
    RosterEntry,
    ScoringEventType,
)


class BasketballTools(ToolKitBase):
    """All the tools for the basketball domain."""

    db: BasketballDB

    def __init__(self, db: BasketballDB) -> None:
        super().__init__(db)

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _get_club(self, club_id: str) -> Club:
        """Get a club from the database."""
        if club_id not in self.db.clubs:
            raise ValueError("Club not found")
        return self.db.clubs[club_id]

    def _get_player(self, player_id: str) -> Player:
        """Get a player from the database."""
        if player_id not in self.db.players:
            raise ValueError("Player not found")
        return self.db.players[player_id]

    def _get_game(self, game_id: str) -> Game:
        """Get a game from the database."""
        if game_id not in self.db.games:
            raise ValueError("Game not found")
        return self.db.games[game_id]

    # ----------------------------
    # Generic utilities
    # ----------------------------
    @is_tool(ToolType.GENERIC)
    def calculate(self, expression: str) -> str:
        """
        Calculate the result of a mathematical expression.

        Args:
            expression: The mathematical expression to calculate, such as '2 + 2'. The expression can contain
                        numbers, operators (+, -, *, /), parentheses, and spaces.

        Returns:
            The result of the mathematical expression as a string (rounded to 2 decimals).

        Raises:
            ValueError: If the expression is invalid.
        """
        if not all(char in "0123456789+-*/(). " for char in expression):
            raise ValueError("Invalid characters in expression")
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """
        Transfer the user to a human agent, with a summary of the user's issue.
        Only transfer if
         - the user explicitly asks for a human agent
         - given the policy and the available tools, you cannot solve the user's issue.

        Args:
            summary: A summary of the user's issue.

        Returns:
            A message indicating the user has been transferred to a human agent.
        """
        return "Transfer successful"

    # ----------------------------
    # READ tools
    # ----------------------------
    @is_tool(ToolType.READ)
    def get_club_details(self, club_id: str) -> Club:
        """Get details of a club by club ID.

        Args:
            club_id: The club ID, such as 'club_001'.

        Returns:
            Club: The club details.

        Raises:
            ValueError: If the club is not found.
        """
        return self._get_club(club_id)

    @is_tool(ToolType.READ)
    def get_player_details(self, player_id: str) -> Player:
        """Get details of a player by player ID.

        Args:
            player_id: The player ID, such as 'player_001'.

        Returns:
            Player: The player details.

        Raises:
            ValueError: If the player is not found.
        """
        return self._get_player(player_id)

    @is_tool(ToolType.READ)
    def get_game_details(self, game_id: str) -> Game:
        """Get details of a game by game ID.

        Args:
            game_id: The game ID, such as 'game_001'.

        Returns:
            Game: The game details.

        Raises:
            ValueError: If the game is not found.
        """
        return self._get_game(game_id)

    @is_tool(ToolType.READ)
    def list_all_clubs(self) -> str:
        """List the name and club id of all clubs.

        Returns:
            str: A JSON string mapping club names to their club IDs, sorted alphabetically by name.
        """
        club_dict = {club.name: club.club_id for club in self.db.clubs.values()}
        return json.dumps(club_dict, sort_keys=True)

    @is_tool(ToolType.READ)
    def list_clubs_by_league(self, league: str) -> str:
        """List clubs filtered by league.

        Args:
            league: League name, case-insensitive match.

        Returns:
            str: A JSON string mapping club names to their club IDs within the specified league,
                 sorted alphabetically by club name.

        Raises:
            ValueError: If no clubs are found for the league.
        """
        league_lower = league.lower()
        filtered = {
            club.name: club.club_id
            for club in self.db.clubs.values()
            if club.league.lower() == league_lower
        }
        if not filtered:
            raise ValueError("No clubs found for the specified league")
        return json.dumps(dict(sorted(filtered.items())), sort_keys=False)

    @is_tool(ToolType.READ)
    def find_player_id_by_email(self, email: str) -> str:
        """Find player ID by email.

        Args:
            email: The email of the player, such as 'player@example.com'.

        Returns:
            str: The player ID.

        Raises:
            ValueError: If the player is not found.
        """
        for player_id, player in self.db.players.items():
            if player.email.lower() == email.lower():
                return player_id
        raise ValueError("Player not found")

    @is_tool(ToolType.READ)
    def find_player_id_by_name(self, first_name: str, last_name: str) -> str:
        """Find player ID by first name and last name.

        Args:
            first_name: First name of the player, e.g., 'John'.
            last_name: Last name of the player, e.g., 'Doe'.

        Returns:
            str: The player ID.

        Raises:
            ValueError: If no player or multiple players are found (ambiguous).
        """
        matches: List[str] = []
        fn = first_name.lower()
        ln = last_name.lower()
        for pid, p in self.db.players.items():
            if p.name.first_name.lower() == fn and p.name.last_name.lower() == ln:
                matches.append(pid)
        if not matches:
            raise ValueError("Player not found")
        if len(matches) > 1:
            raise ValueError("Multiple players found, please provide more information")
        return matches[0]

    @is_tool(ToolType.READ)
    def list_club_roster_ids(self, club_id: str, active_only: bool = True) -> List[str]:
        """List player IDs in a club's roster.

        Args:
            club_id: The club ID, such as 'club_001'.
            active_only: If True, include only active players. Defaults to True.

        Returns:
            List[str]: Sorted list of player IDs.

        Raises:
            ValueError: If the club is not found.
        """
        club = self._get_club(club_id)
        ids = [
            pid
            for pid, entry in club.roster.items()
            if (entry.active if active_only else True)
        ]
        return sorted(ids)

    @is_tool(ToolType.READ)
    def get_game_box_score(self, game_id: str) -> List[BoxScoreEntry]:
        """Get the box score entries for a game.

        Args:
            game_id: The game ID, such as 'game_001'.

        Returns:
            List[BoxScoreEntry]: Box score entries.

        Raises:
            ValueError: If the game is not found.
        """
        game = self._get_game(game_id)
        return game.box_score

    @is_tool(ToolType.READ)
    def get_club_schedule(
        self,
        club_id: str,
        season: Optional[str] = None,
        status: Optional[GameStatus] = None,
    ) -> List[str]:
        """Get the schedule (list of game IDs) for a club, optionally filtered by season and status.

        Args:
            club_id: The club ID.
            season: Optional season (e.g., '2024-25').
            status: Optional status filter ('scheduled', 'in_progress', 'final', 'postponed').

        Returns:
            List[str]: List of game IDs.

        Raises:
            ValueError: If the club is not found.
        """
        club = self._get_club(club_id)
        result: List[str] = []
        for gid in club.games:
            g = self._get_game(gid)
            if season is not None and g.season != season:
                continue
            if status is not None and g.status != status:
                continue
            result.append(gid)
        return result

    @is_tool(ToolType.READ)
    def get_game_score(self, game_id: str) -> Dict[str, Any]:
        """Get a game's current score summary.

        Args:
            game_id: The game ID.

        Returns:
            Dict[str, Any]: {
                "status": <status>,
                "home": {"club_id": <id>, "name": <name>, "score": <int>},
                "away": {"club_id": <id>, "name": <name>, "score": <int>}
            }

        Raises:
            ValueError: If the game is not found.
        """
        g = self._get_game(game_id)
        return {
            "status": g.status,
            "home": {
                "club_id": g.clubs.home.club_id,
                "name": g.clubs.home.name,
                "score": g.clubs.home.score,
            },
            "away": {
                "club_id": g.clubs.away.club_id,
                "name": g.clubs.away.name,
                "score": g.clubs.away.score,
            },
        }

    @is_tool(ToolType.READ)
    def get_game_timeline(self, game_id: str) -> List[GameScoringEvent]:
        """Get the chronological scoring history for a game.

        Args:
            game_id: The game ID.

        Returns:
            List[GameScoringEvent]: The list of scoring events.

        Raises:
            ValueError: If the game is not found.
        """
        g = self._get_game(game_id)
        return g.scoring_history

    @is_tool(ToolType.READ)
    def get_player_total_points(
        self, player_id: str, season: Optional[str] = None
    ) -> int:
        """Compute a player's total points across games, optionally filtered by season.

        Args:
            player_id: The player ID.
            season: Optional season identifier (e.g., '2024-25').

        Returns:
            int: Total points.

        Raises:
            ValueError: If the player is not found.
        """
        p = self._get_player(player_id)
        total = 0
        for gid in p.games:
            g = self._get_game(gid)
            if season is not None and g.season != season:
                continue
            for entry in g.box_score:
                if entry.player_id == player_id:
                    total += entry.points
        return total

    @is_tool(ToolType.READ)
    def get_club_record(self, club_id: str, season: Optional[str] = None) -> Dict[str, int]:
        """Compute a club's win-loss record for a season (or overall if season is None).

        Args:
            club_id: The club ID.
            season: Optional season filter.

        Returns:
            Dict[str, int]: {"wins": <int>, "losses": <int>}

        Raises:
            ValueError: If the club is not found.
        """
        club = self._get_club(club_id)
        wins = 0
        losses = 0
        for gid in club.games:
            g = self._get_game(gid)
            if g.status != "final":
                continue
            if season is not None and g.season != season:
                continue
            home = g.clubs.home
            away = g.clubs.away
            if home.club_id == club_id:
                if home.score > away.score:
                    wins += 1
                elif home.score < away.score:
                    losses += 1
            elif away.club_id == club_id:
                if away.score > home.score:
                    wins += 1
                elif away.score < home.score:
                    losses += 1
        return {"wins": wins, "losses": losses}

    # ----------------------------
    # WRITE tools
    # ----------------------------
    @is_tool(ToolType.WRITE)
    def modify_player_address(
        self,
        player_id: str,
        street: str,
        city: str,
        state: str,
        country: str,
        zip: str,
    ) -> Player:
        """Modify a player's address. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            player_id: The player ID, such as 'player_001'.
            street: Street address, e.g., '123 Main St'.
            city: City name.
            state: State or province.
            country: Country name.
            zip: Postal code.

        Returns:
            Player: The player details after the modification.

        Raises:
            ValueError: If the player is not found.
        """
        player = self._get_player(player_id)
        player.address.street = street
        player.address.city = city
        player.address.state = state
        player.address.country = country
        player.address.zip = zip
        return player

    @is_tool(ToolType.WRITE)
    def update_game_status(self, game_id: str, new_status: GameStatus) -> Game:
        """Update the status of a game. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.

        Allowed transitions:
          - scheduled -> in_progress or postponed
          - postponed -> scheduled
          - in_progress -> final
          - final -> (no further transitions)

        Args:
            game_id: The game ID, such as 'game_001'.
            new_status: One of 'scheduled', 'in_progress', 'final', 'postponed'.

        Returns:
            Game: The game details after the update.

        Raises:
            ValueError: If the game is not found or transition is invalid.
        """
        game = self._get_game(game_id)
        current = game.status
        allowed: Dict[GameStatus, List[GameStatus]] = {
            "scheduled": ["in_progress", "postponed"],
            "postponed": ["scheduled"],
            "in_progress": ["final"],
            "final": [],
        }
        if new_status == current:
            return game
        if new_status not in allowed[current]:
            raise ValueError("Invalid status transition")
        game.status = new_status
        return game

    @is_tool(ToolType.WRITE)
    def record_scoring_event(
        self,
        game_id: str,
        event_type: ScoringEventType,
        points: int,
        club_id: str,
        player_id: str,
        timestamp: str,
    ) -> Game:
        """Record a scoring event for an in-progress game, update the team's score, and append to scoring history.
        The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            game_id: The game ID.
            event_type: '2PT', '3PT', or 'FT'.
            points: Points scored (must match the event type: 2 for 2PT, 3 for 3PT, 1 for FT).
            club_id: The club credited with the score (must be either home or away club of the game).
            player_id: The player credited with the score (must exist in the database).
            timestamp: Timestamp string within the game context (e.g., 'Q2 03:21').

        Returns:
            Game: The game details after the scoring event is recorded.

        Raises:
            ValueError: If the game is not found, not in progress, invalid event, player/club not found,
                        or club is not part of the game.
        """
        game = self._get_game(game_id)
        if game.status != "in_progress":
            raise ValueError("Scoring can only be recorded for games in progress")

        # Validate event type and points consistency
        expected_points = {"2PT": 2, "3PT": 3, "FT": 1}[event_type]
        if points != expected_points:
            raise ValueError("Points do not match the event type")

        # Validate club
        if club_id not in {game.clubs.home.club_id, game.clubs.away.club_id}:
            raise ValueError("Club is not part of this game")

        # Validate player exists
        self._get_player(player_id)

        # Append scoring event
        event = GameScoringEvent(
            event_type=event_type,
            points=points,
            club_id=club_id,
            player_id=player_id,
            timestamp=timestamp,
        )
        game.scoring_history.append(event)

        # Update team score
        if club_id == game.clubs.home.club_id:
            game.clubs.home.score += points
        else:
            game.clubs.away.score += points

        return game
