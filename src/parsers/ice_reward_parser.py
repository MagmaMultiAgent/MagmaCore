from .dense_reward_parser import DenseRewardParser

import numpy as np

from copy import deepcopy
import sys


class IceRewardParser(DenseRewardParser):
    def __init__(self, max_entity_number: int = 1000):
        super(IceRewardParser, self).__init__()
        self.max_entity_number = max_entity_number

    def parse(self, dones, game_state, env_stats, global_info):
        global_reward = [0.0, 0.0]

        final_reward = [np.zeros((self.max_entity_number,), dtype=np.float32) for _ in range(2)]

        step_weight_later = 1 + (game_state[0].real_env_steps / 1000) * 0.1
        step_weight_early = 1 + ((1000 - game_state[0].real_env_steps) / 1000) * 0.1

        # factory counts
        factory_count = [global_info[f'player_{pid}']["factory_count"] for pid in range(2)]

        reward_scale = 0.01

        ice_norm = 1

        for team in [0, 1]:
            player = f"player_{team}"
            own_global_info = global_info[player]
            own_unit_info = own_global_info["units"]
            own_factory_info = own_global_info["factories"]

            last_count = self.last_count[player]
            last_count_units = last_count["units"]
            last_count_factories = last_count["factories"]

            unit_count = own_global_info["unit_count"]

            own_reward_weight = 1.0
            unit_groups = {}
            for unit_name, unit in own_unit_info.items():
                unit_reward = 0

                if unit_name not in last_count_units:
                    continue

                cargo_ice = unit["cargo_ice"]
                last_cargo_ice = last_count_units[unit_name]['cargo_ice']

                ice_increment = max(cargo_ice - last_cargo_ice, 0) / 4
                ice_decrement = max(last_cargo_ice - cargo_ice, 0) / 4  # transfer to factory, 4 ice = 1 water

                ice_increment_reward = ice_increment * 0.1 / ice_norm * step_weight_early
                ice_decrement_reward = ice_decrement / ice_norm * step_weight_early

                unit_reward += ice_increment_reward
                unit_reward += ice_decrement_reward
                unit_reward /= 2  # don't count it twice (onece with gent, once with factory)

                if True:
                    # clearing away rubble from next to lichen and factories
                    # check if unit in same pos as last time
                    # check if unit is next to lichen or factory
                    if unit["x"] == last_count_units[unit_name]["x"] and unit["y"] == last_count_units[unit_name]["y"] and unit["next_to_lichen_or_factory"]:
                        rubble_under = unit["rubble_under"]
                        last_rubble_under = last_count_units[unit_name]["rubble_under"]
                        rubber_decrease = max(last_rubble_under - rubble_under, 0) / 100 * 0.1
                        unit_reward += rubber_decrease

                unit_reward *= reward_scale

                group_id = unit["group_id"]
                if group_id not in unit_groups:
                    unit_groups[group_id] = 0
                unit_groups[group_id] += unit_reward

            for factory_name, factory in own_factory_info.items():
                factory_reward = 0

                if factory_name not in last_count_factories:
                    continue

                cargo_ice = factory["cargo_ice"]
                last_cargo_ice = last_count_factories[factory_name]['cargo_ice']
                ice_increment = max(cargo_ice - last_cargo_ice, 0) / 4  # 4 ice = 1 water

                ice_increment_reward = ice_increment / ice_norm * step_weight_early

                factory_reward += ice_increment_reward

                factory_reward /= 2  # don't count it twice (onece with unit, once with factory)

                # if game is over for both and both have factories and more than 900 steps -> truncation after 1000 steps
                if dones["player_0"] and dones["player_1"] and factory_count[0] > 0 and factory_count[1] > 0 and game_state[0].real_env_steps > 900:
                    lichen_count = factory["lichen_count"]
                    lichen_reward = lichen_count / 100
                    lichen_reward *= 10
                    factory_reward += lichen_reward

                factory_reward *= reward_scale

                group_id = factory["group_id"]
                if group_id not in unit_groups:
                    unit_groups[group_id] = 0
                unit_groups[group_id] += factory_reward

            global_rev = 0
            if len(unit_groups) > 0:
                total_reward = sum(unit_groups.values())
            else:
                total_reward = 0

            for group_id, group_reward in unit_groups.items():
                final_reward[team][group_id] += (group_reward * own_reward_weight) + (total_reward * (1 - own_reward_weight)) + global_rev
            
        _, sub_rewards = super(IceRewardParser, self).parse(dones, game_state, env_stats, global_info)

        self.update_last_count(global_info)

        return final_reward, sub_rewards
    
    def reset(self, game_state, global_info, env_stats):
        self.update_last_count(global_info)

    def update_last_count(self, global_info):
        self.last_count = deepcopy(global_info)
