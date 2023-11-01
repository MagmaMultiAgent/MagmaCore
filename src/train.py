"""
Implementation of RL agent. Note that luxai_s2 and stable_baselines3 \
are packages not available during the competition running (ATM)
"""

# pylint: disable=E0401
import copy
import argparse
import os.path as osp
import gymnasium as gym
import torch as th
from gymnasium.wrappers import TimeLimit
from luxai_s2.state import StatsStateDict
from luxai_s2.utils.heuristics.factory_placement import place_near_random_ice
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.ppo import PPO
from sb3_contrib.ppo_mask import MaskablePPO
from action.controllers import SimpleUnitDiscreteController, SimpleFactoryController
from wrappers.obs_wrappers import SimpleUnitObservationWrapper
from wrappers.sb3_action_mask import SB3InvalidActionWrapper
from net.net import UNetWithResnet50Encoder
from reward.early_reward_parser import EarlyRewardParser
from net.factory_net import FactoryNet

import sys
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger(__name__)
logger.info('Creating logger')

logging.setLoggerClass

class EarlyRewardParserWrapper(gym.Wrapper):
    """
    Custom wrapper for the LuxAI_S2 environment
    """

    def __init__(self, env: gym.Env) -> None:
        """
        Adds a custom reward and turns the LuxAI_S2 environment \
        into a single-agent environment for easy training
        """
        logger.info(f"Adding early reward parser wrapper to environment {env}")
        self.logger = logging.getLogger(f"{__name__}_{id(self)}")
        super().__init__(env)
        self.prev_step_metrics = None
        self.reward_parser = EarlyRewardParser()

    def step(self, action):
        self.logger.debug(f"Stepping environment with action\n{action}")
        
        agent = "player_0"
        opp_agent = "player_1"

        opp_factories = self.env.state.factories[opp_agent]
        for k in opp_factories.keys():
            factory = opp_factories[k]
            factory.cargo.water = 1000

        action = {agent: action}
        obs, _, termination, truncation, info = self.env.step(action)
        done = dict()
        for k in termination:
            done[k] = termination[k] | truncation[k]
        obs = obs[agent]

        stats: StatsStateDict = self.env.state.stats[agent]

        global_info_own = self.reward_parser.get_global_info(agent, self.env.state)
        self.reward_parser.reset(global_info_own, stats)
        reward = self.reward_parser.parse(self.env.state, stats, global_info_own)

        info = dict()
        metrics = dict()
        metrics["ice_dug"] = (
            stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"]
        )
        metrics["ore_dug"] = (
            stats["generation"]["ore"]["HEAVY"] + stats["generation"]["ore"]["LIGHT"]
        )
        metrics["power_consumed"] = (
            stats["consumption"]["power"]["HEAVY"] + stats["consumption"]["power"]["LIGHT"] + stats["consumption"]["power"]["FACTORY"]
        )
        metrics["water_consumed"] = stats["consumption"]["water"]
        metrics["metal_consumed"] = stats["consumption"]["metal"]
        metrics["rubble_destroyed"] = (
            stats["destroyed"]["rubble"]["LIGHT"] + stats["destroyed"]["rubble"]["HEAVY"]
        )
        metrics["ore_transferred"] = stats["transfer"]["ore"]
        metrics["water_transferred"] = stats["transfer"]["water"]
        metrics["energy_transferred"] = stats["transfer"]["power"]
        metrics["energy_pickup"] = stats["pickup"]["power"]
        metrics["light_robots_built"] = stats["generation"]["built"]["LIGHT"]
        metrics["heavy_robots_built"] = stats["generation"]["built"]["HEAVY"]
        metrics["light_power"] = stats["generation"]["power"]["LIGHT"]
        metrics["heavy_power"] = stats["generation"]["power"]["HEAVY"]
        metrics["factory_power"] = stats["generation"]["power"]["FACTORY"]
        metrics["metal_produced"] = stats["generation"]["metal"]
        metrics["water_produced"] = stats["generation"]["water"]
        metrics["action_queue_updates_success"] = stats["action_queue_updates_success"]
        metrics["action_queue_updates_total"] = stats["action_queue_updates_total"]

        info["metrics"] = metrics

        self.prev_step_metrics = copy.deepcopy(metrics)
        return obs, reward, termination[agent], truncation[agent], info

    def reset(self, **kwargs):
        """
        Resets the environment
        """
        obs, reset_info = self.env.reset(**kwargs)
        self.prev_step_metrics = None
        return obs["player_0"], reset_info


def parse_args():
    """
    Parses the arguments
    """

    parser = argparse.ArgumentParser(
        description="Simple script that simplifies Lux AI Season 2 as a single-agent \
        environment with a reduced observation and action space. It trains a policy \
        that can succesfully control a heavy unit to dig ice and transfer it back to \
        a factory to keep it alive"
    )
    parser.add_argument("-s", "--seed", type=int, default=666, help="seed for training")
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel envs to run. Note that the rollout \
        size is configured separately and invariant to this value",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=1000,
        help="Max steps per episode before truncating them",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=5_000_000,
        help="Total timesteps for training",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="If set, will only evaluate a given policy. \
            Otherwise enters training mode",
    )
    parser.add_argument(
        "--model-path", type=str, help="Path to SB3 model \
            weights to use for evaluation"
    )
    parser.add_argument(
        "-l",
        "--log-path",
        type=str,
        default="logs",
        help="Logging path",
    )
    args = parser.parse_args()
    return args


def make_env(env_id: str, rank: int, seed: int = 0, max_episode_steps=100):
    """
    Creates the environment
    """

    def _init() -> gym.Env:
        """
        Initializes the environment
        """
        logger.debug(f"Initializing environment {env_id}")

        env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=4, disable_env_checker=True)

        env = SB3InvalidActionWrapper(
            env,
            factory_placement_policy=place_near_random_ice,
            unit_controller=SimpleUnitDiscreteController(env.env_cfg),
            factory_controller=SimpleFactoryController(env.env_cfg),
        )

        env = SimpleUnitObservationWrapper(
            env
        )
        env = EarlyRewardParserWrapper(env)
        env = TimeLimit(
            env, max_episode_steps=max_episode_steps
        )
        env = Monitor(env)
        logger.debug(f"Resetting env {env}")
        env.reset(seed=seed + rank)
        set_random_seed(seed)

        logger.debug(f"Environment {env} ready")
        return env

    return _init


class TensorboardCallback(BaseCallback):
    """
    Callback for logging metrics to tensorboard
    """

    def __init__(self, tag: str, verbose=0):
        """
        Initializes the callback
        """

        super().__init__(verbose)
        self.tag = tag

    def _on_step(self) -> bool:
        """
        Called on every step
        """
        count = 0

        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                count += 1
                for k in info["metrics"]:
                    stat = info["metrics"][k]
                    self.logger.record_mean(f"{self.tag}/{k}", stat)
        return True


def save_model_state_dict(save_path, model):
    """
    Saves the model state dict
    """

    state_dict = model.policy.to("cpu").state_dict()
    th.save(state_dict, save_path)


def evaluate(args, env_id, model):
    """
    Evaluates the model
    """

    logger.info("Eval mode")
    model = model.load(args.model_path)
    video_length = 1000
    eval_env = SubprocVecEnv(
        [make_env(env_id, i, max_episode_steps=1000) for i in range(args.n_envs)]
    )
    eval_env = VecVideoRecorder(
        eval_env,
        osp.join(args.log_path, "eval_videos"),
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix="evaluation_video",
    )
    eval_env.reset()
    out = evaluate_policy(model, eval_env, render=False, deterministic=False)
    print(out)


def train(args, env_id, model: MaskablePPO, factory_model: MaskablePPO, invalid_action_masking):
    """
    Trains the model
    """

    logger.info("Training mode")
    eval_environments = [make_env(env_id, i, max_episode_steps=1000) for i in range(4)]
    eval_env = DummyVecEnv(eval_environments) if invalid_action_masking \
        else SubprocVecEnv(eval_environments)
    eval_env.reset()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=osp.join(args.log_path, "models"),
        log_path=osp.join(args.log_path, "eval_logs"),
        eval_freq=24_000,
        deterministic=False,
        render=False,
        n_eval_episodes=5,
    )

    logger.info("Starting learning")

    for i in range(args.total_timesteps):
        if i % 2:
            model.learn(1, callback=[TensorboardCallback(tag="unit_train_metrics"), eval_callback])
        else:
            factory_model.learn(1, callback=[TensorboardCallback(tag="factory_train_metrics"), eval_callback])
    
    logger.info("Saving model")
    model.save(osp.join(args.log_path, "models/latest_model"))


def main(args):
    """
    Main function
    """

    logger.debug("Starting main")
    
    logger.info(f"Training with args {args}")
    if args.seed is not None:
        set_random_seed(args.seed)
    env_id = "LuxAI_S2-v0"
    invalid_action_masking = True

    environments = [make_env(env_id, i, max_episode_steps=args.max_episode_steps) \
                    for i in range(args.n_envs)]
    logger.debug(f"Creating {len(environments)} environment(s)")
    env = DummyVecEnv(environments) if invalid_action_masking \
        else SubprocVecEnv(environments)
    logger.debug("Resetting env")
    env.reset()
    logger.debug(f"Env: {env}")
    logger.debug(f"Env action space: {env.action_space}")

    policy_kwargs_unit = {
        "features_extractor_class": UNetWithResnet50Encoder,
        "features_extractor_kwargs": {
            "output_channels": 22,
            }
        }
    policy_kwargs_factory = {
        "features_extractor_class": FactoryNet,
        "features_extractor_kwargs": {
            "num_actions": 3
        }
    }
    rollout_steps = 1000
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        agent_type="unit",
        n_steps=rollout_steps // args.n_envs,
        batch_size=16,
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs_unit,
        verbose=1,
        target_kl=0.05,
        gamma=0.99,
        tensorboard_log=osp.join(args.log_path),
    )

    model_factory = MaskablePPO(
        "MultiInputPolicy",
        env,
        agent_type='factory',
        n_steps=rollout_steps // args.n_envs,
        batch_size=800,
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs_factory,
        verbose=1,
        n_epochs=2,
        target_kl=0.05,
        gamma=0.99,
        tensorboard_log=osp.join(args.log_path),
    )

    logger.debug(f"Model: {model}")
    if args.eval:
        evaluate(args, env_id, model)
    else:
        train(args, env_id, model, model_factory, invalid_action_masking)


if __name__ == "__main__":
    main(parse_args())
