import argparse
import os
import random
import time
from distutils.util import strtobool
from pprint import pprint
import sys
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from policy.net import Net
from luxenv import LuxSyncVectorEnv,LuxEnv
import tree
import json
import gzip
from kit.load_from_replay import replay_to_state_action, get_obs_action_from_json
from utils import save_args, save_model, load_model, eval_model, _process_eval_resluts, cal_mean_return, make_env

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


LOG = True

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
    parser.add_argument("--total-timesteps", type=int, default=100000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=64,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--train-num-collect", type=int, default=64,
        help="the number of data collections in training process")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--save-interval", type=int, default=100000, 
        help="global step interval to save model")
    parser.add_argument("--load-model-path", type=str, default=None,
        help="path for pretrained model loading")
    parser.add_argument("--evaluate-interval", type=int, default=10000,
        help="evaluation steps")
    parser.add_argument("--evaluate-num", type=int, default=5,
        help="evaluation numbers")
    parser.add_argument("--replay-dir", type=str, default=None,
        help="replay dirs to reset state")
    parser.add_argument("--eval", type=bool, default=False,
        help="is eval model")
    
    args = parser.parse_args()
    # size of a batch
    args.batch_size = int(args.num_envs * args.num_steps)
    # number of steps from all envs
    args.num_steps = args.num_steps*args.num_envs
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


def create_model(device: torch.device, eval: bool, load_model_path: Union[str, None], evaluate_num: int, learning_rate: float):
    """
    Create the model
    """
    agent = Net().to(device)
    if load_model_path is not None:
        agent.load_state_dict(torch.load(load_model_path))
        print('load successfully')
        if eval:
            import sys
            for i in range(10):
                eval_results = []
                for _ in range(evaluate_num):
                    eval_results.append(eval_model(agent))
                eval_results = _process_eval_resluts(eval_results)
                if LOG:
                    for key, value in eval_results.items():
                        writer.add_scalar(f"eval/{key}", value, i)
                pprint(eval_results)
            sys.exit()
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    return agent, optimizer


np2torch = lambda x, dtype: torch.tensor(x).type(dtype).to(device)
torch2np = lambda x, dtype: x.cpu().numpy().astype(dtype)


def sample_action_for_player(agent: Net, obs: dict[str, np.ndarray], valid_action: dict[str, np.ndarray], forced_action: Union[dict[str, np.ndarray], None] = None):
    """
    Sample action and value from the agent
    """
    logprob, value, action, entropy = agent(
        np2torch(obs['global_feature'], torch.float32),
        np2torch(obs['map_feature'], torch.float32), 
        tree.map_structure(lambda x: np2torch(x, torch.int16), obs['action_feature']),
        tree.map_structure(lambda x: np2torch(x, torch.bool), valid_action),
        None if forced_action is None else tree.map_structure(lambda x: np2torch(x, torch.float32), forced_action)
    )

    return logprob, value, action, entropy


def sample_actions_for_players(envs: LuxSyncVectorEnv, agent: Net, next_obs: dict[str, np.ndarray]):
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
            
            _logprob, _value, _action, _ = sample_action_for_player(agent, next_obs[player], _valid_action, None)

            action[player_id] = _action
            valid_action[player_id] = _valid_action
            logprob[player_id] = _logprob
            value[player_id] = _value

    return action, valid_action, logprob, value


def calculate_returns(envs: LuxSyncVectorEnv,
                      agent: Net,
                      next_obs: dict[str, np.ndarray],
                      values,
                      device: torch.device,
                      max_train_step: int,
                      num_envs: int,
                      gamma: float,
                      gae_lambda: float
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate GAE returns
    """
    returns = dict(player_0=torch.zeros((max_train_step, num_envs)).to(device),player_1=torch.zeros((max_train_step, num_envs)).to(device))
    advantages = dict(player_0=torch.zeros((max_train_step, num_envs)).to(device),player_1=torch.zeros((max_train_step, num_envs)).to(device))
    with torch.no_grad():
        _, _, _, value = sample_actions_for_players(envs, agent, next_obs)

        for player_id, player in enumerate(['player_0', 'player_1']):
            next_value = value[player_id]
            next_value = next_value.reshape(1,-1)
            lastgaelam = 0
            for t in reversed(range(max_train_step-1)):
                if t == max_train_step - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
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
                   clip_vloss: bool,
                   clip_coef: float,
                   ent_coef: float,
                   vf_coef: float
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the loss
    """
    # Policy loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = newvalue.view(-1)
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
                        b_obs: dict[str, np.ndarray],
                        b_va: dict[str, np.ndarray],
                        b_actions: dict[str, np.ndarray],
                        b_logprobs: dict[str, np.ndarray],
                        b_advantages: dict[str, np.ndarray],
                        b_returns: dict[str, np.ndarray],
                        b_values: dict[str, np.ndarray],
                        train_num_collect: int,
                        minibatch_size: int,
                        clip_vloss: bool,
                        clip_coef: float,
                        norm_adv: bool,
                        ent_coef: float,
                        vf_coef: float,
                        max_grad_norm: float):
    """
    Update weights for a player with PPO
    """
    clipfracs = []
    for start in range(0, train_num_collect, minibatch_size):
        end = start + minibatch_size
        mb_inds = b_inds[start:end]

        mb_obs = envs.concatenate_obs(list(map(lambda i: b_obs[player][i], mb_inds)))
        mb_va = envs.concatenate_va(list(map(lambda i: b_va[player][i], mb_inds)))
        mb_actions = envs.concatenate_action(list(map(lambda i: b_actions[player][i], mb_inds)))

        newlogprob, newvalue, _, entropy = sample_action_for_player(agent, mb_obs, mb_va, mb_actions)

        logratio = newlogprob - b_logprobs[player][mb_inds]
        ratio = logratio.exp()

        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

        mb_advantages = b_advantages[player][mb_inds]
        if norm_adv:
            if len(mb_inds)==1:
                mb_advantages = mb_advantages
            else:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
        mb_returns = b_returns[player][mb_inds]
        mb_values = b_values[player][mb_inds]

        loss, pg_loss, entropy_loss, v_loss = calculate_loss(mb_advantages, mb_returns, mb_values, newvalue, entropy, ratio, clip_vloss, clip_coef, ent_coef, vf_coef)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()

        return v_loss, pg_loss, entropy_loss, approx_kl, old_approx_kl, clipfracs



if __name__ == "__main__":
    args = parse_args()
    player_id = 0
    enemy_id = 1 - player_id
    player = f'player_{player_id}'
    enemy = f'player_{enemy_id}'
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    save_path = f'../results/{run_name}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if LOG:
        writer = SummaryWriter(f"../results/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    save_args(args, save_path+'args.json')

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logger.info(f"Device: {device}")

    # env setup
    envs = LuxSyncVectorEnv(
        [make_env(i, args.seed + i, args.replay_dir) for i in range(args.num_envs)]
    )
    
    # Create model
    agent, optimizer = create_model(device, args.eval, args.load_model_path, args.evaluate_num, args.learning_rate)

    # Start the game
    global_step = 0
    last_save_model_step = 0
    last_eval_step = 0
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size
    
    logger.info("Starting train")
    for update in range(1, num_updates + 1):

        logger.info(f"Update {update} / {num_updates}")
            
        # Init value stores for PPO
        obs = dict(player_0=list(),player_1=list())
        actions = dict(player_0=list(),player_1=list())
        valid_actions = dict(player_0=list(),player_1=list())
        logprobs = dict(player_0=torch.zeros((args.max_train_step, args.num_envs)).to(device), player_1=torch.zeros((args.max_train_step, args.num_envs)).to(device))
        rewards = dict(player_0=torch.zeros((args.max_train_step, args.num_envs)).to(device),player_1=torch.zeros((args.max_train_step, args.num_envs)).to(device))
        dones = torch.zeros((args.max_train_step, args.num_envs)).to(device)
        values = dict(player_0=torch.zeros((args.max_train_step, args.num_envs)).to(device),player_1=torch.zeros((args.max_train_step, args.num_envs)).to(device))

        # Reset envs, get obs
        next_obs, infos = envs.reset()
        next_done = torch.zeros(args.num_envs).to(device)

        total_done = 0
        
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Init stats
        total_return = 0.0
        episode_return = torch.zeros(args.num_envs)
        episode_return_list = []
        episode_sub_return = {}
        episode_sub_return_list = []
        train_step = -1
        
        for step in range(0, args.num_steps):

            if (step+1) % (2**4) == 0:
                logger.info(f"Step {step + 1} / {args.num_steps}")

            train_step += 1 
            global_step += 1 * args.num_envs

            # Save obervations for PPO
            for player_id, player in enumerate(['player_0', 'player_1']):
                obs[player] += envs.split(next_obs[player])
                dones[train_step] = next_done

            # Sample actions
            action, valid_action, logprob, value = sample_actions_for_players(envs, agent, next_obs)

            # Save actions for PPO
            for player_id, player in enumerate(['player_0', 'player_1']):
                values[player][train_step] = value[player_id]
                valid_actions[player] += envs.split(valid_action[player_id])
                actions[player] += envs.split(action[player_id])
                logprobs[player][train_step] = logprob[player_id]
            
            action = tree.map_structure(lambda x: torch2np(x, np.int16), action)

            # Step environment
            next_obs, reward, terminated, truncation, info = envs.step(action)
            done = terminated | truncation
            next_done = torch.tensor(done, dtype=torch.long).to(device)

            # Save rewards for PPO
            for player_id, player in enumerate(['player_0', 'player_1']):
                rewards[player][train_step] = torch.tensor(reward[:, player_id]).to(device).view(-1)

            # Save stats
            episode_return += torch.mean(torch.tensor(reward), dim=-1).to(episode_return.device)
            if True in next_done:
                episode_return_list.append(np.mean(episode_return[torch.where(next_done.cpu()==True)].cpu().numpy()))
                episode_return[torch.where(next_done==True)] = 0
                tmp_sub_return_dict = {}
                for key in episode_sub_return:
                    tmp_sub_return_dict.update({key: np.mean(episode_sub_return[key][torch.where(next_done.cpu()==True)].cpu().numpy())})
                    episode_sub_return[key][torch.where(next_done.cpu()==True)] = 0
                episode_sub_return_list.append(tmp_sub_return_dict)
                total_done_tmp = torch.sum(next_done).cpu().numpy()
                total_done += total_done_tmp
            total_return += cal_mean_return(info['agents'], player_id=0)
            total_return += cal_mean_return(info['agents'], player_id=1)
            if (step== args.num_steps-1):
                logger.info(f"global_step={global_step}, total_return={np.mean(episode_return_list)}")
                if LOG:
                    writer.add_scalar("charts/avg_steps", (step*args.num_envs)/total_done, global_step)
                    writer.add_scalar("charts/episodic_total_return", np.mean(episode_return_list), global_step)
                    mean_episode_sub_return = {}
                    for key in episode_sub_return.keys():
                        mean_episode_sub_return[key] = np.mean(list(map(lambda sub: sub[key], episode_sub_return_list)))
                        writer.add_scalar(f"sub_reward/{key}", mean_episode_sub_return[key], global_step)

            # Train with PPO
            if train_step >= args.max_train_step-1 or step == args.num_steps-1:  
                logger.info("Training with PPO")
                returns, advantages = calculate_returns(envs, agent, next_obs, values, device, args.max_train_step, args.num_envs, args.gamma, args.gae_lambda)

                # flatten the batch
                b_obs = obs   
                b_logprobs = tree.map_structure(lambda x: x.reshape(-1), logprobs)
                b_actions = actions
                b_advantages = tree.map_structure(lambda x: x.reshape(-1), advantages)
                b_returns = tree.map_structure(lambda x: x.reshape(-1), returns)
                b_values = tree.map_structure(lambda x: x.reshape(-1), values)
                b_va = valid_actions

                # Optimizing the policy and value network
                b_inds = np.arange(args.train_num_collect)
                clipfracs = []
                for epoch in range(args.update_epochs):
                    np.random.shuffle(b_inds)
                    for player_id, player in enumerate(['player_0', 'player_1']):
                        v_loss, pg_loss, entropy_loss, approx_kl, old_approx_kl, clipfracs = optimize_for_player(player, agent, optimizer, b_obs, b_va, b_actions, b_logprobs, b_advantages, b_returns, b_values, args.train_num_collect, args.minibatch_size, args.clip_vloss, args.clip_coef, args.norm_adv, args.ent_coef, args.vf_coef, args.max_grad_norm)
                        clipfracs += clipfracs

                        if args.target_kl is not None:
                            if approx_kl > args.target_kl:
                                break
                        
                        y_pred, y_true = b_values[player].cpu().numpy(), b_returns[player].cpu().numpy()
                        var_y = np.var(y_true)
                        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                        # TRY NOT TO MODIFY: record rewards for plotting purposes
                        if LOG:
                            writer.add_scalar(f"losses/value_loss_{player_id}", v_loss.item(), global_step)
                            writer.add_scalar(f"losses/policy_loss_{player_id}", pg_loss.item(), global_step)
                            writer.add_scalar(f"losses/entropy_{player_id}", entropy_loss.item(), global_step)
                            writer.add_scalar(f"losses/old_approx_kl_{player_id}", old_approx_kl.item(), global_step)
                            writer.add_scalar(f"losses/approx_kl_{player_id}", approx_kl.item(), global_step)
                            writer.add_scalar(f"losses/clipfrac_{player_id}", np.mean(clipfracs), global_step)
                            writer.add_scalar(f"losses/explained_variance_{player_id}", explained_var, global_step)
                
                if LOG:
                    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar("charts/SPS", round(global_step / (time.time() - start_time), 2), global_step)
                
                logger.info(f"SPS: {round(global_step / (time.time() - start_time), 2)}")
                logger.info(f"global step: {global_step}")

                total_done += dones.sum().cpu().numpy()
                obs = dict(player_0=list(),player_1=list())
                actions = dict(player_0=list(),player_1=list())
                valid_actions = dict(player_0=list(),player_1=list())
                logprobs = dict(player_0=torch.zeros((args.max_train_step, args.num_envs)).to(device), player_1=torch.zeros((args.max_train_step, args.num_envs)).to(device))
                rewards = dict(player_0=torch.zeros((args.max_train_step, args.num_envs)).to(device),player_1=torch.zeros((args.max_train_step, args.num_envs)).to(device))
                dones = torch.zeros((args.max_train_step, args.num_envs)).to(device)
                values = dict(player_0=torch.zeros((args.max_train_step, args.num_envs)).to(device),player_1=torch.zeros((args.max_train_step, args.num_envs)).to(device))
                train_step = -1
            
            # Evaluate
            if (global_step - last_eval_step) >= args.evaluate_interval:
                eval_results = []
                for _ in range(args.evaluate_num):
                    eval_results.append(eval_model(agent))
                eval_results = _process_eval_resluts(eval_results)
                if LOG:
                    for key, value in eval_results.items():
                        writer.add_scalar(f"eval/{key}", value, global_step)
                pprint(eval_results)
                last_eval_step = global_step
            
            # Save model
            if (global_step - last_save_model_step) >= args.save_interval:
                save_model(agent, save_path+f'model_{global_step}.pth')
                last_save_model_step = global_step
    envs.close()
    if LOG:
        writer.close()