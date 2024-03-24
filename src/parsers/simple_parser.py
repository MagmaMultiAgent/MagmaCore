from .dense_reward_parser import DenseRewardParser

import numpy as np

from copy import deepcopy
import sys


class SimpleRewardParser(DenseRewardParser):
    def parse(self, dones, game_state, env_stats, global_info):
        global_reward = [0.0, 0.0]

        for team in [0, 1]:
            player = f"player_{team}"
            own_global_info = global_info[player]
            own_unit_info = own_global_info["units"]
            own_factory_info = own_global_info["factories"]

            last_count = self.last_count[player]
            last_count_units = last_count["units"]
            last_count_factories = last_count["factories"]

            unit_count = own_global_info["unit_count"]

            step_weight = game_state[0].real_env_steps / 1000

            for unit_name, unit in own_unit_info.items():
                unit_reward = 0

                if unit_name not in last_count_units:
                    continue

                # ice transfered
                ice_decrement = max(last_count_units[unit_name]["cargo_ice"] - unit["cargo_ice"], 0)
                ice_transfer_reward = ice_decrement / 4   # 1 water = 4 ice
                unit_reward += ice_transfer_reward

                global_reward[team] += unit_reward


            for factory_name, factory in own_factory_info.items():
                factory_reward = 0

                if factory_name not in last_count_factories:
                    continue

                # lichen count
                lichen_count = factory["lichen_count"]
                lichen_reward = lichen_count / 20  # 20 lichen can be on a tile
                lichen_reward *= step_weight
                factory_reward += lichen_reward

                global_reward[team] += factory_reward

        _, sub_rewards = super(SimpleRewardParser, self).parse(dones, game_state, env_stats, global_info)

        self.update_last_count(global_info)

        final_reward = global_reward
        return final_reward, sub_rewards
    
    def reset(self, game_state, global_info, env_stats):
        self.update_last_count(global_info)

    def update_last_count(self, global_info):
        self.last_count = deepcopy(global_info)