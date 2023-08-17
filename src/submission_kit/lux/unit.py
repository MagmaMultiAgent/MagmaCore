"""Module representing a unit"""
import math
from dataclasses import dataclass
from typing import List

import numpy as np

from lux.cargo import UnitCargo
from lux.config import EnvConfig, UnitConfig

# a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])


@dataclass
class Unit:
    """Dataclass representing unit config"""

    team_id: int
    unit_id: str
    unit_type: str  # "LIGHT" or "HEAVY"
    pos: np.ndarray
    power: int
    cargo: UnitCargo
    env_cfg: EnvConfig
    unit_cfg: UnitConfig
    action_queue: List

    @property
    def agent_id(self):
        """Property containing which player a unit belongs to"""
        if self.team_id == 0:
            return "player_0"
        return "player_1"

    def action_queue_cost(self):
        """Function returning the cost of changing action queue"""
        cost = self.env_cfg.robots[self.unit_type].action_queue_power_cost
        return cost

    def move_cost(self, game_state, direction):
        """Function calculating the cost of moving"""
        board = game_state.board
        target_pos = self.pos + move_deltas[direction]
        if (
            target_pos[0] < 0
            or target_pos[1] < 0
            or target_pos[1] >= len(board.rubble)
            or target_pos[0] >= len(board.rubble[0])
        ):
            # print("Warning, tried to get move cost for going off the map", file=sys.stderr)
            return None
        factory_there = board.factory_occupancy_map[target_pos[0], target_pos[1]]
        if (
            factory_there not in game_state.teams[self.agent_id].factory_strains
            and factory_there != -1
        ):
            return None
        rubble_at_target = board.rubble[target_pos[0]][target_pos[1]]

        return math.floor(
            self.unit_cfg.mo
            + self.unit_cfg.rubble_movement_cost * rubble_at_target
        )

    def move(self, direction, repeat=0, num=1):
        """Function implementing the move action"""
        if not isinstance(direction, int):
            pass
        return np.array([0, direction, 0, 0, repeat, num])

    def transfer(
        self, transfer_direction, transfer_resource, transfer_amount, repeat=0, num=1
    ):
        """Function implementing the transfer action"""
        assert 0 <= transfer_resource < 5
        assert 0 <= transfer_direction < 5
        return np.array(
            [1, transfer_direction, transfer_resource, transfer_amount, repeat, num]
        )

    def pickup(self, pickup_resource, pickup_amount, repeat=0, num=1):
        """Function implementing the pickup action"""
        assert 0 <= pickup_resource < 5
        return np.array([2, 0, pickup_resource, pickup_amount, repeat, num])

    def dig_cost(self):
        """Function returning the cost of digging"""
        return self.unit_cfg.dig_cost

    def dig(self, repeat=0, num=1):
        """Function implementing the dig function"""
        return np.array([3, 0, 0, 0, repeat, num])

    def self_destruct_cost(self):
        """Function returning the cost of self-destruction"""
        return self.unit_cfg.self_destruct_cost

    def self_destruct(self, repeat=0, num=1):
        """Function implementing the self-destruct action"""
        return np.array([4, 0, 0, 0, repeat, num])

    def recharge(self, x_coord, repeat=0, num=1):
        """Function implementing the recharge function"""
        return np.array([5, 0, 0, x_coord, repeat, num])

    def __str__(self) -> str:
        out = f"[{self.team_id}] {self.unit_id} {self.unit_type} at {self.pos}"
        return out
