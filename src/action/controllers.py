"""
Controller Wrapper for the Competition
"""

from typing import Any, Dict
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

import logging
logger = logging.getLogger(__name__)


class Controller:
    """
    A controller is a class that takes in an action space and converts it into a lux action
    """
    def __init__(self, action_space: spaces.Space) -> None:
        logger.info(f"Creating {self.__class__.__name__}")
        self.logger = logging.getLogger(f"{__name__}_{id(self)}")
        self.action_space = action_space

    def action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        """
        Takes as input the current "raw observation" and the parameterized action and returns
        an action formatted for the Lux env
        """
        raise NotImplementedError()
    
    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Generates a boolean action mask indicating in each \
        discrete dimension whether it would be valid or not
        """
        self.logger.debug("Generating action masks")
        raise NotImplementedError()


class SimpleUnitDiscreteController(Controller):
    """
    A simple controller that controls only the robot \
    that will get spawned.
    """

    def __init__(self, env_cfg) -> None:
        """
        A simple controller that controls only the robot that will get spawned.
        Moreover, it will always try to spawn one heavy robot if there are none regardless of action given

        For the robot unit
        - 4 cardinal direction movement (4 dims)
        - transfer action just for transferring in 4 cardinal directions or center (5 * 3)
        - pickup action for power (1 dims)
        - dig action (1 dim)

        It does not include
        - self destruct action
        - planning (via actions executing multiple times or repeating actions)
        - transferring power or resources other than ice

        To help understand how to this controller works to map one \
        action space to the original lux action space,
        see how the lux action space is defined in luxai_s2/spaces/action.py

        """
        logger.info(f"Creating SimpleUnitDiscreteController")
        self.logger = logging.getLogger(f"{__name__}_{id(self)}")

        self.env_cfg = env_cfg
        self.move_act_dims = 4
        self.one_transfer_dim = 5
        self.transfer_act_dims = self.one_transfer_dim * 3
        self.pickup_act_dims = 1
        self.dig_act_dims = 1
        self.recharge_act_dims = 1

        self.move_dim_high = self.move_act_dims
        self.transfer_dim_high = self.move_dim_high + self.transfer_act_dims
        self.pickup_dim_high = self.transfer_dim_high + self.pickup_act_dims
        self.dig_dim_high = self.pickup_dim_high + self.dig_act_dims
        self.recharge_dim_high = self.dig_dim_high + self.recharge_act_dims

        self.total_act_dims = self.recharge_dim_high
        action_space = spaces.Discrete(self.total_act_dims)
        super().__init__(action_space)

    def _is_move_action(self, id):
        """
        Checks if the action id corresponds to a move action
        """
        return id < self.move_dim_high

    def _get_move_action(self, id):
        """
        Converts the action id to a move action
        """
        return np.array([0, id + 1, 0, 0, 0, 1])

    def _is_transfer_action(self, id):
        """
        Checks if the action id corresponds to a transfer action
        """
        return id < self.transfer_dim_high

    def _get_transfer_action(self, id):
        
        id = id - (self.transfer_act_dims/3)
        transfer_dir = id % (self.transfer_act_dims/3)
        transfer_type_mapping = [0, 1, 4]  # Mapping index to desired value
        transfer_type_index = id // (self.transfer_act_dims/3)  # 0 for ice, 1 for ore, 2 for power
        transfer_type = transfer_type_mapping[transfer_type_index.astype(int).item()]
        return np.array([1, transfer_dir, transfer_type, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_pickup_action(self, id):
        """
        Checks if the action id corresponds to a pickup action
        """
        return id < self.pickup_dim_high

    def _get_pickup_action(self, id):
        """
        Converts the action id to a pickup action
        """
        return np.array([2, 0, 4, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_dig_action(self, id):
        """
        Checks if the action id corresponds to a dig action
        """
        return id < self.dig_dim_high

    def _get_dig_action(self, id):
        """
        Converts the action id to a dig action
        """
        return np.array([3, 0, 0, 0, 0, 1])
    
    def _is_recharge_action(self, id):
        """
        Checks if the action id corresponds to a recharge action
        """
        return id < self.recharge_dim_high
    
    def _get_recharge_action(self, id):
        """
        Converts the action id to a recharge action
        """
        return np.array([4, 0, 0, 0, 0, 1])

    def action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        logging.debug(f"Creating lux action for agent {agent} with action\n{action}")
        unit_action = self.unit_action_to_lux_action(agent, obs, action)
        return unit_action

    def unit_action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        """
        Converts the action to a lux action
        """
        self.logger.debug(f"Creating lux action for agent {agent} with action\n{action}")

        shared_obs = obs["player_0"]
        lux_action = {}
        units = shared_obs["units"][agent]

        for unit_id, unit in units.items():
            pos = tuple(unit['pos'])
            filtered_action = action[pos]
            choice = filtered_action
            action_queue = []
            no_op = False
            if self._is_move_action(choice):
                action_queue = [self._get_move_action(choice)]
            elif self._is_transfer_action(choice):
                action_queue = [self._get_transfer_action(choice)]
            elif self._is_pickup_action(choice):
                action_queue = [self._get_pickup_action(choice)]
            elif self._is_dig_action(choice):
                action_queue = [self._get_dig_action(choice)]
            elif self._is_recharge_action(choice):
                action_queue = [self._get_recharge_action(choice)]
            else:
                no_op = True

            if len(unit["action_queue"]) > 0 and len(action_queue) > 0:
                same_actions = (unit["action_queue"][0] == action_queue[0]).all()
                if same_actions:
                    no_op = True
            if not no_op:
                lux_action[unit_id] = action_queue

        return lux_action

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Defines a simplified action mask for this controller's action space

        Doesn't account for whether robot has enough power
        """

        ## Should return a 19x64x64 mask. 19 actions, 64x64 board
        ## It should return all false where units are not there

        self.logger.debug(f"Creating simplified action mask for {agent}")

        shared_obs = obs[agent]
        factory_occupancy_map = (
            np.ones_like(shared_obs["board"]["rubble"], dtype=int) * -1
        )
        player_occupancy_map = (
            np.ones_like(shared_obs["board"]["rubble"], dtype=int) * -1
        )
        for player in shared_obs["units"]:
            for unit_id in shared_obs["units"][player]:
                unit = shared_obs["units"][player][unit_id]
                pos = np.array(unit["pos"])
                player_occupancy_map[pos[0], pos[1]] = int(unit["unit_id"].split("_")[1])
                
        for player in shared_obs["factories"]:
            for unit_id in shared_obs["factories"][player]:
                f_data = shared_obs["factories"][player][unit_id]
                f_pos = f_data["pos"]
                # store in a 3x3 space around the factory position it's strain id.
                factory_occupancy_map[
                    f_pos[0] - 1 : f_pos[0] + 2, f_pos[1] - 1 : f_pos[1] + 2
                ] = f_data["strain_id"]

        units = shared_obs["units"][agent]
        action_mask = np.ones((self.total_act_dims, self.env_cfg.map_size, self.env_cfg.map_size), dtype=bool)
        for unit_id in units.keys():
            
            # get position of unit
            pos = np.array(unit["pos"])

            # move is only valid if there is a unit there
            action_mask[:4, pos[0], pos[1]] = True

            # transferring is valid only if the target exists
            unit = units[unit_id]
            
            # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
            move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
            for i, move_delta in enumerate(move_deltas):
                transfer_pos = np.array(
                    [pos[0] + move_delta[0], pos[1] + move_delta[1]]
                )
                # check if transfer position is valid
                if (
                    transfer_pos[0] < 0
                    or transfer_pos[1] < 0
                    or transfer_pos[0] >= len(factory_occupancy_map)
                    or transfer_pos[1] >= len(factory_occupancy_map[0])
                ):
                    continue

                # check if there is a unit there or factory
                factory_there = factory_occupancy_map[transfer_pos[0], transfer_pos[1]]
                unit_there = player_occupancy_map[transfer_pos[0], transfer_pos[1]]

                if unit_there != -1:
                    for j in range(3):  # we want to allow transfer of any kind of resouce if its possible
                        index = (self.transfer_dim_high - self.transfer_act_dims + i) + j * self.one_transfer_dim
                        action_mask[index, pos[0], pos[1]] = True

                if factory_there in shared_obs["teams"][agent]["factory_strains"]:
                    for j in range(3):  # same for factories
                        index = (self.transfer_dim_high - self.transfer_act_dims + i) + j * self.one_transfer_dim
                        action_mask[index, pos[0], pos[1]] = True

            factory_there = factory_occupancy_map[pos[0], pos[1]]
            on_top_of_factory = (
                factory_there in shared_obs["teams"][agent]["factory_strains"]
            )

            # dig is valid only if on top of tile with rubble or resources or lichen
            board_sum = (
                shared_obs["board"]["ice"][pos[0], pos[1]]
                + shared_obs["board"]["ore"][pos[0], pos[1]]
                + shared_obs["board"]["rubble"][pos[0], pos[1]]
                + shared_obs["board"]["lichen"][pos[0], pos[1]]
            )
            if board_sum > 0 and not on_top_of_factory:
                action_mask[
                    self.dig_dim_high - self.dig_act_dims : self.dig_dim_high, pos[0], pos[1]
                ] = True

            # pickup is valid only if on top of factory tile
            if on_top_of_factory:
                action_mask[
                    self.pickup_dim_high - self.pickup_act_dims : self.pickup_dim_high, pos[0], pos[1]
                ] = True
                action_mask[
                    self.dig_dim_high - self.dig_act_dims : self.dig_dim_high, pos[0], pos[1]
                ] = False

            # recharge should only be valid in the night
            if shared_obs["real_env_steps"] % 40 > 30:
                action_mask[
                    self.recharge_dim_high - self.recharge_act_dims : self.recharge_dim_high, pos[0], pos[1]
                ] = True
    
        return action_mask



class SimpleFactoryController(Controller):
    """
    A simple controller that controls only the robot \
    that will get spawned.
    """

    def __init__(self, env_cfg) -> None:
        """

        """
        logger.info(f"Creating SimpleFactoryController")
        self.logger = logging.getLogger(f"{__name__}_{id(self)}")
        self.env_cfg = env_cfg
        action_space = spaces.Discrete(3)
        super().__init__(action_space)


    def action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        logging.debug(f"Creating lux action for agent {agent} with action\n{action}")
        unit_action = self.factory_action_to_lux_action(agent, obs, action)
        return unit_action

    def factory_action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        self.logger.debug(f"Creating lux action for factory {agent} with action {action}")

        lux_action = {}
        shared_obs = obs["player_0"]
        factories = shared_obs["factories"][agent]
        
        for i, factory_id in enumerate(factories.keys()):
            lux_action[factory_id] = np.argmax(action[i])

        return lux_action
    

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        return np.ones((3,), dtype=bool)