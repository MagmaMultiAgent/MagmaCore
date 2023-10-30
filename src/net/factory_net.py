import IPython
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import sys
from net import *
import logging
logger = logging.getLogger(__name__)


class FactoryNet(BaseFeaturesExtractor):
    def __init__(self, observation_space, num_actions):
        
        logger.info(f"Creating {self.__class__.__name__}")
        self.logger = logging.getLogger(f"{__name__}_{id(self)}")
        logger.debug(f"Observation space AAAA: {observation_space}")
        self.input_dim = observation_space['factory'].shape[0]
        super(FactoryNet, self).__init__(observation_space, num_actions)

        self.linear = nn.Linear(self.input_dim, num_actions)

    def forward(self, x):
        x = x['factory']
        self.logger.debug(f"Forward call input {x.shape}")
        x = self.linear(x)
        self.logger.debug(f"Factory net output {x.shape}")
        return x


