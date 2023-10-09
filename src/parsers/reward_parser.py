from dataclasses import asdict, dataclass, field
from functools import reduce
from copy import deepcopy
from scipy.stats import gamma
from lux.kit import EnvConfig, GameState
from typing import List
import numpy as np
import tree
import gymnasium as gym

stats_reward_params = dict(
    action_queue_updates_total=0,
    action_queue_updates_success=0,
    consumption={
        "power": {
            "LIGHT": 0,
            "HEAVY": 0,
            "FACTORY": 0,
        },
        "water": 0,
        "metal": 0,
        "ore": {
            "LIGHT": 0,
            "HEAVY": 0,
        },
        "ice": {
            "LIGHT": 0,
            "HEAVY": 0,
        },
    },
    destroyed={
        'FACTORY': 0,
        'LIGHT': {
            'own': -RewardParam.light_reward_weight,
            'enm': RewardParam.light_reward_weight,
        },
        'HEAVY': {
            'own': -RewardParam.heavy_reward_weight,
            'enm': RewardParam.heavy_reward_weight,
        },
        'rubble': {
            'LIGHT': 0,
            'HEAVY': 0,
        },
        'lichen': {
            'LIGHT': {
                'own': 0,
                'enm': 0,
            },
            'HEAVY': {
                'own': 0,
                'enm': 0,
            },
        },
    },
    generation={
        'power': {
            'LIGHT': 0,
            'HEAVY': 0,
            'FACTORY': 0,
        },
        'water': 0,
        'metal': 0,
        'ore': {
            'LIGHT': 0,
            'HEAVY': 0,
        },
        'ice': {
            'LIGHT': 0.0,
            'HEAVY': 0.0,
        },
        'lichen': 0,
        'built': {
            'LIGHT': 0,
            'HEAVY': 0,
        },
    },
    pickup={
        'power': 0,
        'water': 0,
        'metal': 0,
        'ice': 0,
        'ore': 0,
    },
    transfer={
        'power': {
            "LIGHT": 0,
            "HEAVY": 0,
            "FACTORY": 0,
        },
        'water': {
            "LIGHT": 0,
            "HEAVY": 0,
            "FACTORY": 0,
        },
        'metal': {
            "LIGHT": 0,
            "HEAVY": 0,
            "FACTORY": 0,
        },
        'ice': {
            "LIGHT": 0,
            "HEAVY": 0,
            "FACTORY": 0,
        },
        'ore': {
            "LIGHT": 0,
            "HEAVY": 0,
            "FACTORY": 0,
        },
    },
)

@dataclass
class RewardParam ():
    use_gamma_coe: bool = False
    zero_sum: bool = True
    global_reward_weight = 0
    win_reward_weight: float = 0. * global_reward_weight
    light_reward_weight: float = 0.4 * global_reward_weight
    heavy_reward_weight: float = 4 * global_reward_weight
    ice_reward_weight: float = 0.005 * global_reward_weight
    ore_reward_weight: float = 0.01 * global_reward_weight
    water_reward_weight: float = 0.01 * global_reward_weight
    metal_reward_weight: float = 0.02 * global_reward_weight
    power_reward_weight: float = 0.0005 * global_reward_weight
    lichen_reward_weight: float = 0.002
    factory_penalty_weight: float = 5 * global_reward_weight
    lose_penalty_coe: float = 0.
    survive_reward_weight: float = 0.01

class DenseRewardParser(gym.Wrapper):

    def __init__(self, ):
        pass

    def reset(self, game_state, global_info, env_stats):
        self.update_last_count(global_info)
        self.update_env_stats(env_stats)


    def step(self, dones, game_state: List[GameState], env_stats, global_info):
        sub_rewards_keys = [
            "reward_light",
            "reward_heavy",
            "reward_ice",
            "reward_ore",
            "reward_water",
            "reward_metal",
            "reward_lichen",
            "reward_factory",
            "reward_survival",
            "reward_win_lose",
        ]
        sub_rewards = [
            {k: 0
             for k in sub_rewards_keys},
            {k: 0
             for k in sub_rewards_keys},
        ]

        env_stats_rewards = {}
        for player in ["player_0", "player_1"]:
            env_stats_rewards[player] = tree.map_structure(
                lambda cur, last, param: (cur - last) * param,
                env_stats[player],
                self.last_env_stats[player],
                stats_reward_params,
            )

        for team in [0, 1]:
            player = f"player_{team}"
            env_stats_rewards[player] = tree.flatten_with_path(env_stats_rewards[player])
            env_stats_rewards[player] = list(
                map(
                    lambda item: {"reward_" + "_".join(item[0]).lower(): item[1]},
                    env_stats_rewards[player],
                ))
            env_stats_rewards[player] = reduce(lambda cat1, cat2: dict(cat1, **cat2), env_stats_rewards[player])
            sub_rewards[team].update(env_stats_rewards[player])

        if RewardParam.use_gamma_coe:
            gamma_coe = GammaTransform.gamma_(game_state[0].real_env_steps)
            gamma_flipped_coe = GammaTransform.gamma_flipped(game_state[0].real_env_steps)
        else:
            gamma_coe, gamma_flipped_coe = 1, 1
        for team in [0, 1]:
            player = f"player_{team}"
            own_global_info = global_info[player]
            enm_global_info = global_info[f"player_{1 - team}"]
            last_count = self.last_count[player]
            own_sub_rewards = sub_rewards[team]

            factories_increment = own_global_info["factory_count"] - last_count['factory_count']
            light_increment = own_global_info["light_count"] - last_count['light_count']
            heavy_increment = own_global_info["heavy_count"] - last_count['heavy_count']
            ice_increment = own_global_info["total_ice"] - last_count['total_ice']
            ore_increment = own_global_info["total_ore"] - last_count['total_ore']
            water_increment = own_global_info["total_water"] - last_count['total_water']
            metal_increment = own_global_info["total_metal"] - last_count['total_metal']
            power_increment = own_global_info["total_power"] - last_count['total_power']
            lichen_increment = own_global_info["lichen_count"] - last_count['lichen_count']

            own_sub_rewards["reward_light"] = light_increment * RewardParam.light_reward_weight * gamma_flipped_coe
            own_sub_rewards["reward_heavy"] = heavy_increment * RewardParam.heavy_reward_weight
            own_sub_rewards["reward_ice"] = max(ice_increment, 0) * RewardParam.ice_reward_weight
            own_sub_rewards["reward_ore"] = max(ore_increment, 0) * RewardParam.ore_reward_weight * gamma_coe
            own_sub_rewards["reward_water"] = water_increment * RewardParam.water_reward_weight * gamma_flipped_coe
            own_sub_rewards["reward_metal"] = metal_increment * RewardParam.metal_reward_weight * gamma_coe
            own_sub_rewards["reward_power"] = power_increment * RewardParam.power_reward_weight
            own_sub_rewards["reward_lichen"] = lichen_increment * RewardParam.lichen_reward_weight * gamma_flipped_coe
            own_sub_rewards["reward_factory"] = factories_increment * RewardParam.factory_penalty_weight
            own_sub_rewards["reward_survival"] = RewardParam.survive_reward_weight * gamma_flipped_coe

            if dones[f'player_{team}']:
                own_lichen = own_global_info["lichen_count"]
                enm_lichen = enm_global_info["lichen_count"]
                if enm_global_info["factory_count"] == 0:
                    win = True
                elif own_lichen > enm_lichen:
                    win = True
                else:
                    win = False

                if win:
                    own_sub_rewards["reward_win_lose"] = RewardParam.win_reward_weight * (own_lichen - enm_lichen)**0.5
                    own_sub_rewards["reward_win_lose"] += (
                        game_state[team].env_cfg.max_episode_length -
                        game_state[team].real_env_steps) * RewardParam.survive_reward_weight * 2
                else:
                    all_past_reward = 0
                    all_past_reward += own_global_info["light_count"] * RewardParam.light_reward_weight
                    all_past_reward += own_global_info["heavy_count"] * RewardParam.heavy_reward_weight
                    all_past_reward += own_global_info["total_ice"] * RewardParam.ice_reward_weight
                    all_past_reward += own_global_info["total_ore"] * RewardParam.ore_reward_weight
                    all_past_reward += own_global_info["total_water"] * RewardParam.water_reward_weight
                    all_past_reward += own_global_info["total_metal"] * RewardParam.metal_reward_weight
                    all_past_reward += own_global_info["total_power"] * RewardParam.power_reward_weight
                    all_past_reward += own_global_info["lichen_count"] * RewardParam.lichen_reward_weight
                    all_past_reward += own_global_info["factory_count"] * RewardParam.factory_penalty_weight
                    all_past_reward += game_state[team].env_steps * RewardParam.survive_reward_weight
                    own_sub_rewards["reward_win_lose"] = RewardParam.lose_penalty_coe * all_past_reward

        rewards = [sum(sub_rewards[0].values()), sum(sub_rewards[1].values())]

        # record total reward for logging
        # it is not added to the training rewards
        sub_rewards[0]["reward_total"] = sum(sub_rewards[0].values())
        sub_rewards[1]["reward_total"] = sum(sub_rewards[1].values())

        # ensure it is a zero-sum game
        if RewardParam.zero_sum:
            rewards_mean = sum(rewards) / 2
            rewards[0] -= rewards_mean
            rewards[1] -= rewards_mean

            # survival reward is not zero-sum
            rewards[0] += sub_rewards[0]["reward_survival"]
            rewards[1] += sub_rewards[1]["reward_survival"]

        self.update_last_count(global_info)
        self.update_env_stats(env_stats)

        return rewards, sub_rewards

    def update_last_count(self, global_info):
        self.last_count = deepcopy(global_info)

    def update_env_stats(self, env_stats: dict):
        self.last_env_stats = deepcopy(env_stats)