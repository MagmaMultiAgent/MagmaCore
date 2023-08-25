"""Module representing a team in the game"""
from dataclasses import dataclass
from enum import Enum

TERM_COLORS = False
try:
    from termcolor import colored

    TERM_COLORS = True
except ImportError:
    pass


@dataclass
class FactionInfo:
    """Dataclass containing faction information"""
    color: str = "none"
    alt_color: str = "red"
    faction_id: int = -1


class FactionTypes(Enum):
    """Enum class containing all possible factions"""
    NULL = FactionInfo(color="gray", faction_id=0)
    ALPHA_STRIKE = FactionInfo(color="yellow", faction_id=1)
    MOTHER_MARS = FactionInfo(color="green", faction_id=2)
    THE_BUILDERS = FactionInfo(color="blue", faction_id=3)
    FIRST_MARS = FactionInfo(color="red", faction_id=4)


class Team:
    """Class representing a team"""
    def __init__(
        self,
        team_id: int,
        agent: str,
        faction: FactionTypes = None,
        water=0,
        metal=0,
        factories_to_place=0,
        factory_strains=None,
        place_first=False,
    ) -> None:
        self.faction = faction
        self.team_id = team_id
        # the key used to differentiate ownership of things in state
        self.agent = agent

        self.water = water
        self.metal = metal
        self.factories_to_place = factories_to_place
        self.factory_strains = factory_strains
        # whether this team gets to place factories down first or not.
        # The bid winner has this set to True.
        # If tied, player_0's team has this True
        self.place_first = place_first

    def state_dict(self):
        """Function returning state dictionary"""
        return {"team_id": self.team_id,
         "faction": self.faction.name,
         "water": self.water,
         "metal": self.metal,
         "factories_to_place": self.factories_to_place,
         "factory_strains": self.factory_strains,
         "place_first": self.place_first
         }

    def __str__(self) -> str:
        out = f"[Player {self.team_id}]"
        if TERM_COLORS:
            return colored(out, self.faction.value.color)
        return out
