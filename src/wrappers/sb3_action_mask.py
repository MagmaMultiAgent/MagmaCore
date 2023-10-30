"""
This file contains a wrapper that adds action masks to the environment \
for use with stable-baselines3
"""
import copy
from typing import Dict, Callable
import gymnasium as gym
from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict, StatsStateDict
from luxai_s2.unit import BidActionType, FactoryPlacementActionType
from luxai_s2.wrappers.controllers import Controller
from wrappers.sb3 import SB3Wrapper
from action.controllers import SimpleUnitDiscreteController, SimpleFactoryController
from reward.early_reward_parser import EarlyRewardParser

import logging
logger = logging.getLogger(__name__)

class SB3InvalidActionWrapper(SB3Wrapper):
    """
    This wrapper adds action masks to the environment for use with stable-baselines3
    """

    def __init__(
        self,
        env: LuxAI_S2,
        bid_policy: Callable[
            [str, ObservationStateDict], Dict[str, BidActionType]
        ] = None,
        factory_placement_policy: Callable[
            [str, ObservationStateDict], Dict[str, FactoryPlacementActionType]
        ] = None,
        unit_controller: SimpleUnitDiscreteController = None,
        factory_controller: SimpleFactoryController = None,
    ) -> None:
        """
        This wrapper adds action masks to the environment for use with stable-baselines3
        """

        logger.info(f"Creating {self.__class__.__name__}")
        self.logger = logging.getLogger(f"{__name__}_{id(self)}")
        super().__init__(env, bid_policy, factory_placement_policy, unit_controller, factory_controller)

    def action_masks(self, agent_type: str):
        """
        Generates a boolean action mask indicating in each \
        discrete dimension whether it would be valid or not
        """
        self.logger.debug("Generating action mask")
        if agent_type == 'factory':
            mask = self.factory_controller.action_masks('player_0', self.prev_obs)
        else:
            mask = self.unit_controller.action_masks('player_0', self.prev_obs)
        self.logger.debug(mask.shape)
        return mask
