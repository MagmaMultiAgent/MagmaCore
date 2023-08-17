"""Module containing the main loop"""
import json
from argparse import Namespace
from submission_kit.lux.config import EnvConfig
from submission_kit.lux.kit import (
    process_action,
    process_obs,
)
from agent import Agent


### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict = (
    {}
)  # store potentially multiple dictionaries as kaggle imports code directly
agent_prev_obs = {}


def agent_fn(observation, configurations):
    """
    agent definition for kaggle submission.
    """
    global agent_dict
    step = observation.step

    player = observation.player
    remaining_overage_time = observation.remaining_overage_time
    if step == 0:
        env_cfg = EnvConfig.from_dict(configurations["env_cfg"])
        agent_dict[player] = Agent(player, env_cfg)
        agent_prev_obs[player] = {}
        agent = agent_dict[player]
    agent = agent_dict[player]
    obs = process_obs(player, agent_prev_obs[player], step, json.loads(observation.obs))
    agent_prev_obs[player] = obs
    agent.step = step
    if step == 0:
        actions = agent.bid_policy(step, obs, remaining_overage_time)
    elif obs["real_env_steps"] < 0:
        actions = agent.factory_placement_policy(step, obs, remaining_overage_time)
    else:
        actions = agent.act(step, obs, remaining_overage_time)

    return process_action(actions)


if __name__ == "__main__":

    def read_input():
        """
        Reads input from stdin
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof) from eof

    i = 0
    while True:
        inputs = read_input()
        obs = json.loads(inputs)

        observation = Namespace(
            {"step": obs['step'], "obs": json.dumps(obs['obs']),
             "remainingOverageTime": obs['remainingOverageTime'],
             "player": obs['player'], "info":obs['info']}
        )
        if i == 0:
            configurations = obs["info"]["env_cfg"]
        i += 1
        actions = agent_fn(observation, {"env_cfg": configurations})
        # send actions to engine
        print(json.dumps(actions))
