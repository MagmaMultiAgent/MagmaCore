import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    CustomExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from sb3_contrib.common.maskable.distributions import MaskableDistribution, make_masked_proba_distribution

import logging
logger = logging.getLogger(__name__)


class MaskableActorCriticPolicy(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
            squash_output=False,
        )
<<<<<<< HEAD
=======

>>>>>>> df4383346aeb574afc020781a778898a56dc5875
        if isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            warnings.warn(
                (
                    "As shared layers in the mlp_extractor are removed since SB3 v1.8.0, "
                    "you should now pass directly a dictionary and not a list "
                    "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
                ),
            )
            net_arch = net_arch[0]

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = dict(pi=[64, 64], vf=[64, 64])

        self.hidden_value_layers = [1024, 512, 128, 1]
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.share_features_extractor = share_features_extractor
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        if self.share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.make_features_extractor()

        # Action distribution
        self.action_dist = make_masked_proba_distribution(action_space)

        self._build(lr_schedule)

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :param action_masks: Action masks to apply to the action distribution
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi: th.Tensor = self.mlp_extractor.forward_actor(pi_features)
            latent_vf: th.Tensor = self.mlp_extractor.forward_critic(vf_features)

        values = self.value_net(latent_vf)

        latent_shape = latent_pi.shape
        assert len(latent_shape) == 4

        if action_masks is not None:
            assert latent_shape == action_masks.shape

        batch_size, action_channels, height, width = latent_shape

        latent_pi = latent_pi.permute(0, 2, 3, 1)
        latent_pi = latent_pi.reshape(-1, action_channels)
        assert latent_pi.shape[0] == batch_size * height * width
        
        if action_masks is not None:
            action_masks = np.transpose(action_masks, (0, 2, 3, 1))

            action_masks = action_masks.reshape(-1, action_channels)

            assert latent_pi.shape == action_masks.shape

        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        #log_prob = log_prob * (actions != 15)
        actions = actions.reshape(batch_size, height, width)
        actions = actions.permute(0, 1, 2)
        log_prob = log_prob.view(batch_size, height, width)
        log_prob = log_prob.permute(0, 1, 2)
        log_prob = log_prob.sum(axis=[1,2])
        return actions, values, log_prob

    def extract_features(self, obs: th.Tensor) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.
        :param obs: Observation
        :return: the output of the features extractor(s)
        """
        if self.share_features_extractor:
            return super().extract_features(obs, self.features_extractor)
        else:
            pi_features = super().extract_features(obs, self.pi_features_extractor)
            vf_features = super().extract_features(obs, self.vf_features_extractor)
            return pi_features, vf_features

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = CustomExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            obs_shape = self.observation_space['map'].shape[1],
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        self.action_net = self.action_dist.proba_distribution_net(latent_dim=self.mlp_extractor.latent_dim_pi)
        layers = []

        layers.append(nn.Linear(self.mlp_extractor.latent_dim_pi, self.hidden_value_layers[0]))
        layers.append(self.activation_fn())
        for i in range(1, len(self.hidden_value_layers)):
            layers.append(nn.Linear(self.hidden_value_layers[i - 1], self.hidden_value_layers[i]))
            layers.append(self.activation_fn())
        self.value_net = nn.Sequential(*layers)

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> MaskableDistribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        action_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def _predict(
        self,
        observation: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :param action_masks: Action masks to apply to the action distribution
        :return: Taken action according to the policy
        """
        distribution, batch_size, height, width = self.get_distribution(observation, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        actions = actions.reshape(batch_size, height, width)
        actions = actions.permute(0, 1, 2)
        return actions

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :param action_masks: Action masks to apply to the action distribution
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic, action_masks=action_masks)
            # Convert to numpy
            actions = actions.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions = actions.squeeze(axis=0)

        return actions, None

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        latent_shape = latent_pi.shape
        assert len(latent_shape) == 4
        
        batch_size, action_channels, height, width = latent_shape

        if action_masks is not None:
            assert action_masks.shape == (batch_size * height * width, action_channels)

        latent_pi = latent_pi.permute(0, 2, 3, 1)
        latent_pi = latent_pi.reshape(-1, action_channels)

        assert latent_pi.shape[0] == batch_size * height * width

        if action_masks is not None:
            assert latent_pi.shape == action_masks.shape

        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks, eval = True)

        log_prob = distribution.log_prob(actions)

        #log_prob = log_prob * (actions != 15)
        entropy = distribution.entropy()
        #entropy = entropy * (actions != 15)
        
        entropy = entropy.view(batch_size, height, width)
        actions = actions.reshape(batch_size, height, width)
        actions = actions.permute(0, 1, 2)
        log_prob = log_prob.view(batch_size, height, width)
        log_prob = log_prob.permute(0, 1, 2)
        entropy = entropy.permute(0, 1, 2)
        log_prob = log_prob.sum(axis=[1,2])
        values = self.value_net(latent_vf)
        return values, log_prob, entropy

    def get_distribution(self, obs: th.Tensor, action_masks: Optional[np.ndarray] = None) -> MaskableDistribution:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation
        :param action_masks: Actions' mask
        :return: the action distribution.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, _ = self.mlp_extractor(features)
        else:
            pi_features, _ = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
        batch_size, action_channels, height, width = latent_pi.shape


        latent_pi = latent_pi.permute(0, 2, 3, 1)
        latent_pi = latent_pi.reshape(-1, action_channels)
        assert latent_pi.shape[0] == batch_size * height * width
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)

        return distribution, batch_size, height, width

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)


class MaskableActorCriticCnnPolicy(MaskableActorCriticPolicy):
    """
    CNN policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


class MaskableMultiInputActorCriticPolicy(MaskableActorCriticPolicy):
    """
    MultiInputActorClass policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space (Tuple)
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param features_extractor_class: Uses the CombinedExtractor
    :param features_extractor_kwargs: Keyword arguments
        to pass to the feature extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
