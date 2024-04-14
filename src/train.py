# %%writefile src/train.py

import argparse
import os
import random
import time
from distutils.util import strtobool
import sys
from typing import Union

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from policy.net import Net
from policy.simple_net import SimpleNet
from luxenv import LuxSyncVectorEnv
import tree
from utils import save_args, save_model, cal_mean_return, make_env
import gc
from pprint import pprint

import seeding

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
                    handlers=[logging.StreamHandler()])
stream_handler = [h for h in logging.root.handlers if isinstance(h , logging.StreamHandler)][0]
stream_handler.setLevel(logging.INFO)
stream_handler.setStream(sys.stderr)
logger = logging.getLogger("train")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

# Types
TensorPerKey = dict[str, torch.Tensor]
TensorPerPlayer = dict[str, dict[str, torch.Tensor]]

LOG = True
log_from_global_info = [
    'factory_count',
    'unit_count',
    'light_count',
    'heavy_count',
    'unit_ice',
    'unit_ore',
    'unit_water',
    'unit_metal',
    'unit_power',
    'factory_ice',
    'factory_ore',
    'factory_water',
    'factory_metal',
    'factory_power',
    'total_ice',
    'total_ore',
    'total_water',
    'total_metal',
    'total_power',
    'lichen_count',
    'units_on_ice',
    'avg_distance_from_ice',
    'rubble_on_ice',

    'ice_transfered',
    'ore_transfered',
    'ice_mined',
    'ore_mined',
    'lichen_grown',
    'unit_created',
    'light_created',
    'heavy_created',
    'unit_destroyed',
    'light_destroyed',
    'heavy_destroyed',
]

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="LuxAI_S2-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=32,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=512,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--train-num-collect", type=int, default=16384,
        help="the number of data collections in training process")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.001,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--save-interval", type=int, default=65536,
        help="global step interval to save model")
    parser.add_argument("--load-model-path", type=str, default=None,
        help="path for pretrained model loading")
    parser.add_argument("--replay-dir", type=str, default=None,
        help="replay dirs to reset state")
    parser.add_argument("--evaluate-interval", type=int, default=65536,
        help="evaluation steps")
    parser.add_argument("--evaluate-num", type=int, default=16,
        help="evaluation numbers")

    args = parser.parse_args()

    if args.seed is None:
        args.seed = 42

    # Test arguments
    if True:
        args.num_steps = 200
        args.num_envs = 4
        args.train_num_collect = args.num_envs*args.num_steps
        args.evaluate_interval = None
        args.save_interval = None
        args.evaluate_num = 2

    # Reward per entity
    args.max_entity_number = 500

    # size of a batch
    args.batch_size = int(args.num_envs * args.num_steps)
    # number of steps to train on from all envs
    args.train_num_collect = args.minibatch_size if args.train_num_collect is None else args.train_num_collect
    # size of a minibatch
    args.minibatch_size = int(args.train_num_collect // args.num_minibatches)
    # how many steps to stop at when collecting data
    args.max_train_step = int(args.train_num_collect // args.num_envs)

    logger.info(args)
    return args


def layer_init(layer, std: float = np.sqrt(2), bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def put_into_store(data: dict, ind: int, store: dict, max_train_step: int, num_envs: int, device: Union[torch.device, str]):
    for key, value in data.items():
        if isinstance(value, dict):
            if key not in store:
                store[key] = {}
            put_into_store(value, ind, store[key], max_train_step, num_envs, device)
        else:
            if key not in store:
                store[key] = torch.zeros((max_train_step, num_envs) + value.shape[1:], device=device, dtype=value.dtype)
            store[key][ind] = value


def reset_store(store: dict):
    for key, value in store.items():
        if isinstance(value, dict):
            reset_store(store[key])
        else:
            if store[key].dtype in {torch.float32, torch.float64, torch.int32, torch.int64}:
                store[key][:] = 0
            elif store[key].dtype in {torch.bool}:
                store[key][:] = False
            else:
                raise NotImplementedError(f"store[key].dtype={store[key].dtype}")


def create_model(device: Union[torch.device, str], load_model_path: Union[str, None], learning_rate: float, max_entity_number: int, seed: int):
    """
    Create the model
    """
    agent = SimpleNet(max_entity_number, seed).to(device)
    if load_model_path is not None:
        agent.load_state_dict(torch.load(load_model_path))
        print('load successfully')

    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    return agent, optimizer


def create_traced_model(agent: Net, obs: TensorPerPlayer, envs: LuxSyncVectorEnv, device: Union[torch.device, str], forced_action: Union[TensorPerKey, None] = None):
    valid_action = envs.get_valid_actions(0)
    valid_action = tree.map_structure(lambda x: np2torch(x, torch.bool), valid_action)
    obs = obs['player_0']

    traced_model = torch.jit.trace(agent, (
        obs['global_feature'].to(device),
        obs['map_feature'].to(device),
        obs['factory_feature'].to(device),
        obs['unit_feature'].to(device),
        obs['location_feature'].to(device),
        tree.map_structure(lambda x: x.to(device), valid_action),
        None if forced_action is None else tree.map_structure(lambda x: x.to(device), forced_action)
    ))
    return traced_model


def sample_action_for_player(agent: Net, obs: TensorPerKey, valid_action: TensorPerKey, device: Union[torch.device, str], forced_action: Union[TensorPerKey, None] = None):
    """
    Sample action and value from the agent
    """
    logprob, value, action, entropy = agent(
        obs['global_feature'].to(device),
        obs['map_feature'].to(device),
        obs['factory_feature'].to(device),
        obs['unit_feature'].to(device),
        obs['location_feature'].to(device),
        tree.map_structure(lambda x: x.to(device), valid_action),
        None if forced_action is None else tree.map_structure(lambda x: x.to(device), forced_action)
    )

    return logprob, value, action, entropy


def sample_actions_for_players(envs: LuxSyncVectorEnv,
                               agent: Net,
                               next_obs: TensorPerPlayer,
                               model_device: Union[torch.device, str],
                               store_device: Union[torch.device, str]
                               ) -> tuple[TensorPerPlayer, TensorPerPlayer, TensorPerPlayer, TensorPerPlayer]:
    """
    Sample action and value for both players
    """
    action = dict()
    valid_action = dict()
    logprob = dict()
    value = dict()

    for player_id, player in enumerate(['player_0', 'player_1']):
        with torch.no_grad():
            _valid_action = envs.get_valid_actions(player_id)
            _valid_action = tree.map_structure(lambda x: np2torch(x, torch.bool), _valid_action)

            _logprob, _value, _action, _ = sample_action_for_player(agent, next_obs[player], _valid_action, model_device, None)

            action[player] = tree.map_structure(lambda x: x.detach().to(store_device), _action)
            valid_action[player] = _valid_action
            logprob[player] = _logprob.detach().to(store_device)
            value[player] = _value.detach().to(store_device)

    return action, valid_action, logprob, value


def calculate_returns(envs: LuxSyncVectorEnv,
                      agent: Net,
                      next_obs: TensorPerKey,
                      next_done: torch.Tensor,
                      dones: torch.Tensor,
                      rewards: TensorPerKey,
                      values: TensorPerKey,
                      max_train_step: int,
                      num_envs: int,
                      max_entity_number: int,
                      gamma: float,
                      gae_lambda: float,
                      model_device: Union[torch.device, str],
                      store_device: Union[torch.device, str]
                      ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Calculate GAE returns
    """
    returns = dict(player_0=torch.zeros((max_train_step, num_envs, max_entity_number)).to("cpu"),player_1=torch.zeros((max_train_step, num_envs, max_entity_number)).to("cpu"))
    advantages = dict(player_0=torch.zeros((max_train_step, num_envs, max_entity_number)).to("cpu"),player_1=torch.zeros((max_train_step, num_envs, max_entity_number)).to("cpu"))
    with torch.no_grad():
        _, _, _, value = sample_actions_for_players(envs, agent, next_obs, model_device, store_device)

        for player in ['player_0', 'player_1']:
            next_value = value[player]
            next_value = next_value.reshape(1,-1)
            lastgaelam = 0
            for t in reversed(range(max_train_step-1)):
                if t == max_train_step - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[player][t + 1]
                    nextvalues = values[player][t + 1]
                delta = rewards[player][t] + gamma * nextvalues * nextnonterminal - values[player][t]
                advantages[player][t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns[player] = advantages[player] + values[player]
    return returns, advantages


def calculate_loss(advantages: torch.Tensor,
                   returns: torch.Tensor,
                   values: torch.Tensor,
                   newvalue: torch.Tensor,
                   entropy: torch.Tensor,
                   ratio: torch.Tensor,
                   max_entity_number: int,
                   clip_vloss: bool,
                   clip_coef: float,
                   ent_coef: float,
                   vf_coef: float,
                   valid_samples: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the loss
    """
    newvalue = newvalue.view(-1, max_entity_number)

    advantages = advantages[valid_samples]
    returns = returns[valid_samples]
    values = values[valid_samples]
    newvalue = newvalue[valid_samples]
    entropy = entropy[valid_samples]
    ratio = ratio[valid_samples]

    # Policy loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    if clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = values + torch.clamp(
            newvalue - values,
            -clip_coef,
            clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

    # Entropy loss
    entropy_loss = entropy.mean()

    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

    return loss, pg_loss, entropy_loss, v_loss


def optimize_for_player(player: str,
                        agent: Net,
                        optimizer: optim.Optimizer,
                        b_inds: torch.Tensor,
                        b_obs: dict[str, list[torch.Tensor]],
                        b_va: dict[str, list[torch.Tensor]],
                        b_actions: dict[str, list[torch.Tensor]],
                        b_logprobs: dict[str, list[torch.Tensor]],
                        b_advantages: dict[str, list[torch.Tensor]],
                        b_returns: dict[str, list[torch.Tensor]],
                        b_values: dict[str, list[torch.Tensor]],
                        max_entity_number: int,
                        train_num_collect: int,
                        minibatch_size: int,
                        clip_vloss: bool,
                        clip_coef: float,
                        norm_adv: bool,
                        ent_coef: float,
                        vf_coef: float,
                        max_grad_norm: float,
                        device: Union[torch.device, str]
                        ) -> tuple[float, float, float, float, float, list[float]]:
    """
    Update weights for a player with PPO
    """
    clipfracs = []
    for start in range(0, train_num_collect, minibatch_size):
        end = start + minibatch_size
        mb_inds = b_inds[start:end]

        mb_obs = tree.map_structure(lambda x: (x.view(-1, *x.shape[2:])[mb_inds]).to(device), b_obs[player])
        mb_va = tree.map_structure(lambda x: (x.view(-1, *x.shape[2:])[mb_inds]).to(device), b_va[player])
        mb_actions = tree.map_structure(lambda x: (x.view(-1, *x.shape[2:])[mb_inds]).to(device), b_actions[player])
        mb_logprobs = (b_logprobs[player][mb_inds]).to(device)
        mb_returns = (b_returns[player][mb_inds]).to(device)
        mb_values = (b_values[player][mb_inds]).to(device)

        valid_samples = torch.where(mb_logprobs != 0)

        newlogprob, newvalue, _, entropy = sample_action_for_player(agent, mb_obs, mb_va, device, mb_actions)

        logratio = newlogprob - mb_logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

        mb_advantages = (b_advantages[player][mb_inds]).to(device)
        if norm_adv:
            if len(mb_inds)==1:
                mb_advantages = mb_advantages
            else:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        loss, pg_loss, entropy_loss, v_loss = calculate_loss(mb_advantages, mb_returns, mb_values, newvalue, entropy, ratio, max_entity_number, clip_vloss, clip_coef, ent_coef, vf_coef, valid_samples)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()

        return v_loss, pg_loss, entropy_loss, approx_kl, old_approx_kl, clipfracs


def write(writer, prefix, results, step):
    for key, value in results.items():
        new_prefix = f"{prefix}/{key}"
        if isinstance(value, dict):
            write(writer, new_prefix, value, step)
        else:
            writer.add_scalar(new_prefix, value, step)


def eval2(agent: torch.nn.Module, writer, seed: int = 0, num_envs: int = 8, device: Union[torch.device, str] = "cpu", global_step: int = 0) -> dict:
    print("Evaluating")
    with torch.no_grad():
        eval_seed = seed
        envs = LuxSyncVectorEnv(
            [make_env(i, eval_seed + i, None, device=device) for i in range(num_envs)],
            device=device
        )

        own = 0
        enemy = 1 - own

        # Reset envs, get obs
        next_obs, _ = envs.reset(seed=eval_seed)
        next_obs = tree.map_structure(lambda x: np2torch(x, torch.float32), next_obs)

        # Init stats
        episode_return = np.zeros((num_envs, 2))
        step_counts = np.zeros(num_envs)
        global_info_save_own = {
            i: {} for i in range(num_envs)
        }
        global_info_save_enemy = {
            i: {} for i in range(num_envs)
        }
        first_episode = np.ones(num_envs, dtype=bool)

        max_step = 1024
        for step in range(0, max_step):

            if (step+1) % (max_step / 8) == 0:
                logger.info(f"Eval Step {step + 1} / {max_step}")

            # Sample actions
            action, _, _, _ = sample_actions_for_players(envs, agent, next_obs, device, device)

            # Step environment
            _action = {}
            for player_id, player in enumerate(['player_0', 'player_1']):
                _action[player_id] = action[player]
            action = tree.map_structure(lambda x: torch2np(x, np.int32), _action)
            del _action
            next_obs, reward, terminated, truncation, info = envs.step(action)
            next_obs = tree.map_structure(lambda x: np2torch(x, torch.float32), next_obs)

            # reward is shape (env, player, group)
            episode_return[np.where(first_episode), :] += np.sum(reward, axis=-1)[np.where(first_episode), :]
            step_counts[np.where(first_episode)] += 1

            # Save info
            for key in log_from_global_info:
                save_key = f"sum_{key}"
                
                for env_id in range(num_envs):
                    if not first_episode[env_id]:
                        continue

                    if save_key not in global_info_save_own[env_id]:
                        global_info_save_own[env_id][save_key] = 0
                    if save_key not in global_info_save_enemy[env_id]:
                        global_info_save_enemy[env_id][save_key] = 0
                    
                    global_info_save_own[env_id][save_key] += info[f"player_{own}"][env_id][key]
                    global_info_save_enemy[env_id][save_key] += info[f"player_{enemy}"][env_id][key]

            done = terminated | truncation
            # all entities done for a player, at least one player is done
            _done = done.all(axis=-1).any(-1)
            first_episode = first_episode & ~_done

            if (~first_episode).all():
                break

        # Need a list of tuples where each tuple contains info about environment
        # - episode_length
        # - return_own
        # - return_enemy
        # - info_sum_own
        # - info_sum_enemy
        return_own = {
            i: episode_return[i, 0] for i in range(num_envs)
        }
        return_enemy = {
            i: episode_return[i, 1] for i in range(num_envs)
        }
        episode_length = {
            i: step_counts[i] for i in range(num_envs)
        }
        info_sum_own = {
            i: global_info_save_own[i] for i in range(num_envs)
        }
        info_sum_enemy = {
            i: global_info_save_enemy[i] for i in range(num_envs)
        }
        results = [
            (episode_length[i], return_own[i], return_enemy[i], info_sum_own[i], info_sum_enemy[i]) for i in range(num_envs)
        ]
        eval_results = envs.process_eval_results(results)
        if writer:
            write(writer, "eval", eval_results, global_step)
        print(f"Finished evaluating for step {global_step}")
        pprint({
            "avg_return_total": eval_results["avg_return_total"],
            "avg_episode_length": eval_results["avg_episode_length"],
            "total_ice_transfered": eval_results["avg_info_total"]["sum_ice_transfered"],
            "total_ice_mined": eval_results["avg_info_total"]["sum_ice_mined"],
        })
        envs.close()
        del envs


def main(args, model_device, store_device):
    player_id = 0
    enemy_id = 1 - player_id
    player = f'player_{player_id}'
    run_name = 'standard_icein_allemb16x2_small1x1_large5x5d2_agr4_comb16_val41_noun_perunit_rub_8k_fp_e001_am_test'
    save_path = f'/content/drive/MyDrive/Lux/MA/results/{run_name}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if LOG:
        writer = SummaryWriter(f"/content/drive/MyDrive/Lux/MA/results/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    else:
        writer = None

    save_args(args, save_path+'args.json')

    # TRY NOT TO MODIFY: seeding
    seeding.set_seed(args.seed)

    # Create model
    agent, optimizer = create_model(model_device, args.load_model_path, args.learning_rate, args.max_entity_number, args.seed)
    traced_model = None

    # reset seed after model creation
    seeding.set_seed(args.seed)

    # env setup
    envs = LuxSyncVectorEnv(
        [make_env(i, args.seed + i, args.replay_dir, device=model_device, max_entity_number=args.max_entity_number) for i in range(args.num_envs)],
        device=model_device
    )

    # Start the game
    global_step = 0
    last_eval_step = 0
    last_save_model_step = 0
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size

    # Init value stores for PPO
    # Store the value on 'store_device' (cpu)
    obs = {}
    actions = {}
    valid_actions = {}
    logprobs = dict(player_0=torch.zeros((args.max_train_step, args.num_envs, args.max_entity_number), device=store_device), player_1=torch.zeros((args.max_train_step, args.num_envs, args.max_entity_number), device=store_device))
    rewards = dict(player_0=torch.zeros((args.max_train_step, args.num_envs, args.max_entity_number), device=store_device), player_1=torch.zeros((args.max_train_step, args.num_envs, args.max_entity_number), device=store_device))
    dones = dict(player_0=torch.zeros((args.max_train_step, args.num_envs, args.max_entity_number), device=store_device), player_1=torch.zeros((args.max_train_step, args.num_envs, args.max_entity_number), device=store_device))
    values = dict(player_0=torch.zeros((args.max_train_step, args.num_envs, args.max_entity_number), device=store_device), player_1=torch.zeros((args.max_train_step, args.num_envs, args.max_entity_number), device=store_device))

    logger.info("Starting train")
    last_seed = args.seed
    for update in range(1, num_updates + 1):
        
        logger.info(f"Update {update} / {num_updates}")
        new_seed = np.random.SeedSequence(last_seed).generate_state(2)
        last_seed = new_seed[0].item()
        new_seed = new_seed[1].item()
        seeding.set_seed(new_seed)

        # Reset envs, get obs
        next_obs, _ = envs.reset(seed=new_seed)
        next_obs = tree.map_structure(lambda x: np2torch(x, torch.float32), next_obs)
        next_done = torch.zeros((args.num_envs, 2, args.max_entity_number), device=store_device, dtype=torch.bool)

        # Evaluate at the beggining
        if args.evaluate_interval is not None and last_eval_step == 0:
            traced_model = create_traced_model(agent, next_obs, envs, model_device)
            eval2(traced_model, writer, seed=0, num_envs=args.evaluate_num, device=model_device, global_step=global_step)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Init stats
        total_return = 0.0
        episode_return = np.zeros(args.num_envs)
        episode_return_list = []
        step_counts = np.zeros(args.num_envs)
        episode_lengths = []
        episode_sub_return = {}
        episode_sub_return_list = []
        train_step = -1
        global_info_save = {}
        first_episode = np.ones(args.num_envs, dtype=bool)

        for step in range(0, args.num_steps):

            if (step+1) % (args.num_steps / 8) == 0:
                logger.info(f"Step {step + 1} / {args.num_steps}")

            train_step += 1
            global_step += 1 * args.num_envs

            # Save obervations for PPO
            for player_id, player in enumerate(['player_0', 'player_1']):
                for env_id in range(0, args.num_envs):
                    # insert tensor [env, player, entity] into [player, step, env, entity]
                    dones[player][train_step, env_id] = next_done[env_id, player_id]
            put_into_store(next_obs, train_step, obs, args.max_train_step, args.num_envs, store_device)

            # Sample actions
            action, valid_action, logprob, value = sample_actions_for_players(envs, traced_model, next_obs, model_device, store_device)

            # Save actions for PPO
            put_into_store(action, train_step, actions, args.max_train_step, args.num_envs, store_device)
            put_into_store(valid_action, train_step, valid_actions, args.max_train_step, args.num_envs, store_device)
            for player in ['player_0', 'player_1']:
                logprobs[player][train_step] = logprob[player]
                values[player][train_step] = value[player]

            # Step environment
            _action = {}
            for player_id, player in enumerate(['player_0', 'player_1']):
                _action[player_id] = action[player]
            action = tree.map_structure(lambda x: torch2np(x, np.int32), _action)
            del _action
            next_obs, reward, terminated, truncation, info = envs.step(action)
            next_obs = tree.map_structure(lambda x: np2torch(x, torch.float32), next_obs)

            # reward is shape (env, player, group)
            episode_return += np.mean(np.sum(reward, axis=-1), axis=-1)

            step_counts += 1

            reward = np2torch(reward, torch.float32)

            done = terminated | truncation
            # all entities done for a player, at least one player is done
            _done = done.all(axis=-1).any(-1)
            if step == args.num_steps-1:
                _done[:] = 1
            next_done = np2torch(done, torch.bool)

            # Save rewards for PPO
            for player_id, player in enumerate(['player_0', 'player_1']):
                rewards[player][train_step] = reward[:, player_id]

            # Save global info
            for key in log_from_global_info:
                for env_id in range(args.num_envs):

                    if not first_episode[env_id]:
                        continue

                    for player in ["player_0", "player_1"]:
                        if player not in global_info_save:
                            global_info_save[player] = {}
                        if "total" not in global_info_save:
                            global_info_save["total"] = {}
                        if key not in global_info_save[player]:
                            global_info_save[player][key] = 0
                        if key not in global_info_save["total"]:
                            global_info_save["total"][key] = 0

                        global_info_save[player][key] += info[player][env_id][key]
                        global_info_save["total"][key] += info[player][env_id][key]

            # Save stats
            if _done.any():
                done_envs_all = [d.item() for d in np.where(_done==True)[0]]
                done_envs = [d.item() for d in np.where((_done==True) & (first_episode==True))[0]]
                for env_ind in done_envs:
                    episode_return_list.append(episode_return[env_ind])
                    episode_lengths.append(step_counts[env_ind])
                episode_return[done_envs_all] = 0
                step_counts[done_envs_all] = 0
                first_episode[done_envs_all] = False

            total_return += cal_mean_return(info['agents'], player_id=0)
            total_return += cal_mean_return(info['agents'], player_id=1)

            if (step == args.num_steps-1):
                return_mean = np.mean(episode_return_list)
                return_median = np.median(episode_return_list)
                length_mean = np.mean(episode_lengths)
                length_median = np.median(episode_lengths)
                logger.info(f"global_step={global_step}, total_return={return_mean.round(8)} ({return_median.round(8)}), episode_length={length_mean.round(2)} ({length_median.round(2)})")
                if LOG:
                    writer.add_scalar("charts/episodic_total_return", return_mean, global_step)
                    writer.add_scalar("charts/episodic_length", length_mean, global_step)
                    mean_episode_sub_return = {}
                    for key in episode_sub_return.keys():
                        mean_episode_sub_return[key] = np.mean(list(map(lambda sub: sub[key], episode_sub_return_list)))
                        writer.add_scalar(f"sub_reward/{key}", mean_episode_sub_return[key], global_step)

                    for groupname, group in global_info_save.items():
                        for key, value in group.items():
                            multiplier = (1 / args.num_envs) if groupname != "total" else (1 / (args.num_envs * 2))
                            writer.add_scalar(f"global_info/sum_{groupname}_{key}", value * multiplier, global_step)
                            global_info_save[groupname][key] = 0
                    global_info_save = {}


            # Train with PPO
            if train_step >= args.max_train_step-1 or step == args.num_steps-1:
                logger.info("Training with PPO")
                returns, advantages = calculate_returns(envs, traced_model, next_obs, next_done, dones, rewards, values, args.max_train_step, args.num_envs, args.max_entity_number, args.gamma, args.gae_lambda, model_device, store_device)

                # flatten the batch
                b_obs = obs
                b_actions = actions
                b_va = valid_actions

                b_logprobs = tree.map_structure(lambda x: x.view(-1, args.max_entity_number), logprobs)
                b_advantages = tree.map_structure(lambda x: x.view(-1, args.max_entity_number), advantages)
                b_returns = tree.map_structure(lambda x: x.view(-1, args.max_entity_number), returns)
                b_values = tree.map_structure(lambda x: x.view(-1, args.max_entity_number), values)

                # Optimizing the policy and value network
                b_inds = np.arange(args.train_num_collect)
                clipfracs = []
                for _ in range(args.update_epochs):
                    np.random.shuffle(b_inds)
                    _b_inds = np2torch(b_inds, torch.long)
                    v_loss_total = 0
                    pg_loss_total = 0
                    entropy_loss_total = 0
                    approx_kl_total = 0
                    old_approx_kl_total = 0
                    explained_var_total = 0
                    advantage_total = 0
                    reward_total = 0
                    logprob_total = 0
                    valid_sample_count_total = 0
                    clipfracs_total = []
                    total_weight_norm_total = 0
                    total_bias_norm_total = 0

                    for player_id, player in enumerate(['player_0', 'player_1']):
                        v_loss, pg_loss, entropy_loss, approx_kl, old_approx_kl, clipfracs = optimize_for_player(player, agent, optimizer, _b_inds, b_obs, b_va, b_actions, b_logprobs, b_advantages, b_returns, b_values, args.max_entity_number, args.train_num_collect, args.minibatch_size, args.clip_vloss, args.clip_coef, args.norm_adv, args.ent_coef, args.vf_coef, args.max_grad_norm, model_device)
                        clipfracs += clipfracs
                        clipfracs_total += clipfracs

                        v_loss_total += v_loss
                        pg_loss_total += pg_loss
                        entropy_loss_total += entropy_loss
                        approx_kl_total += approx_kl
                        old_approx_kl_total += old_approx_kl

                        if args.target_kl is not None:
                            if approx_kl > args.target_kl:
                                print(f"Approx KL {approx_kl} > Target KL {args.target_kl}")
                                break

                        y_pred, y_true = b_values[player].cpu().numpy(), b_returns[player].cpu().numpy()
                        var_y = np.var(y_true)
                        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                        explained_var_total += explained_var

                        valid_samples = torch.where(b_logprobs[player] != 0)
                        valid_sample_count = len(valid_samples[0])
                        advantage = b_advantages[player][valid_samples].mean().item()
                        reward = b_returns[player][valid_samples].mean().item()
                        logprob = b_logprobs[player][valid_samples].mean().item()

                        advantage_total += advantage
                        reward_total += reward
                        logprob_total += logprob
                        valid_sample_count_total += valid_sample_count

                        # TRY NOT TO MODIFY: record rewards for plotting purposes
                        if LOG:
                            writer.add_scalar(f"losses/value_loss_{player_id}", v_loss.item(), global_step)
                            writer.add_scalar(f"losses/policy_loss_{player_id}", pg_loss.item(), global_step)
                            writer.add_scalar(f"losses/entropy_{player_id}", entropy_loss.item(), global_step)
                            writer.add_scalar(f"losses/old_approx_kl_{player_id}", old_approx_kl.item(), global_step)
                            writer.add_scalar(f"losses/approx_kl_{player_id}", approx_kl.item(), global_step)
                            writer.add_scalar(f"losses/clipfrac_{player_id}", np.mean(clipfracs), global_step)
                            writer.add_scalar(f"losses/explained_variance_{player_id}", explained_var, global_step)
                            writer.add_scalar(f"losses/advantage_{player_id}", advantage, global_step)
                            writer.add_scalar(f"losses/return_{player_id}", reward, global_step)
                            writer.add_scalar(f"losses/logprob_{player_id}", logprob, global_step)
                            writer.add_scalar(f"losses/num_agents_{player_id}", valid_sample_count, global_step)
                            # norm of all weights as a single number
                            total_weight_norm = 0
                            for param in agent.parameters():
                                if param.requires_grad:
                                    total_weight_norm += torch.norm(param.data)
                            total_bias_norm = 0
                            for param in agent.parameters():
                                if param.requires_grad:
                                    total_bias_norm += torch.norm(param.data)
                            writer.add_scalar(f"losses/weight_norm_{player_id}", total_weight_norm, global_step)
                            writer.add_scalar(f"losses/bias_norm_{player_id}", total_bias_norm, global_step)
                            total_weight_norm_total += total_weight_norm
                            total_bias_norm_total += total_bias_norm
                    
                    if LOG:
                        v_loss_total /= 2
                        pg_loss_total /= 2
                        entropy_loss_total /= 2
                        approx_kl_total /= 2
                        old_approx_kl_total /= 2
                        explained_var_total /= 2
                        advantage_total /= 2
                        reward_total /= 2
                        logprob_total /= 2
                        valid_sample_count_total /= 2
                        total_weight_norm_total /= 2
                        total_bias_norm_total /= 2
                        writer.add_scalar("losses/value_loss_total", v_loss_total, global_step)
                        writer.add_scalar("losses/policy_loss_total", pg_loss_total, global_step)
                        writer.add_scalar("losses/entropy_total", entropy_loss_total, global_step)
                        writer.add_scalar("losses/old_approx_kl_total", old_approx_kl_total, global_step)
                        writer.add_scalar("losses/approx_kl_total", approx_kl_total, global_step)
                        writer.add_scalar("losses/clipfrac_total", np.mean(clipfracs_total), global_step)
                        writer.add_scalar("losses/explained_variance_total", explained_var_total, global_step)
                        writer.add_scalar("losses/advantage_total", advantage_total, global_step)
                        writer.add_scalar("losses/return_total", reward_total, global_step)
                        writer.add_scalar("losses/logprob_total", logprob_total, global_step)
                        writer.add_scalar("losses/num_agents_total", valid_sample_count_total, global_step)
                        writer.add_scalar("losses/weight_norm_total", total_weight_norm_total, global_step)
                        writer.add_scalar("losses/bias_norm_total", total_bias_norm_total, global_step)

                # free up memory
                del b_obs
                del b_actions
                del b_va
                del b_logprobs
                del b_advantages
                del b_returns
                del b_values
                gc.collect()
                torch.cuda.empty_cache()

                if LOG:
                    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar("charts/SPS", round(global_step / (time.time() - start_time), 2), global_step)
                    writer.add_scalar("charts/SPR", round((time.time() - start_time) / update, 2), global_step)

                logger.info(f"SPS: {round(global_step / (time.time() - start_time), 2)}")
                logger.info(f"SPR: {round((time.time() - start_time) / update, 2)}")
                logger.info(f"global step: {global_step}")

                reset_store(obs)
                reset_store(actions)
                reset_store(valid_actions)

                for player_id, player in enumerate(['player_0', 'player_1']):
                    logprobs[player][:] = 0
                    rewards[player][:] = 0
                    dones[player][:] = 0
                    values[player][:] = 0

                train_step = -1

                traced_model = create_traced_model(agent, next_obs, envs, model_device)

            # Evaluate initially
            if args.evaluate_interval and (global_step - last_eval_step) >= args.evaluate_interval:
                eval2(traced_model, writer, seed=0, num_envs=args.evaluate_num, device=model_device, global_step=global_step)
                last_eval_step = global_step

            # Save model
            if args.save_interval and (global_step - last_save_model_step) >= args.save_interval:
                save_model(agent, save_path+f'model_{global_step}.pth')
                last_save_model_step = global_step
    
    envs.close()
    if LOG:
        writer.close()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    args = parse_args()

    # Get device
    device1 = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    device2 = torch.device("cpu")

    logger.info(f"Device: {device1}")

    np2torch = lambda x, dtype: torch.tensor(x, device=device2, dtype=dtype)
    cpu2device = lambda x: x.to(device1)
    device2cpu = lambda x: x.detach().to(device2)
    torch2np = lambda x, dtype: x.detach().cpu().numpy().astype(dtype)

    main(args, device1, device2)
