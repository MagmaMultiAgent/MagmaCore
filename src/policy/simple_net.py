import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.ops as T
import numpy as np
from impl_config import ActDims, UnitActChannel, UnitActType, EnvParam
from .actor_head import sample_from_categorical
import sys


class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()


        # EMBEDDINGS
        """
        Embeddings are used to convert the input features into a lower-dimensional space, and to extract relevant information from the input features.
        """

        self.embedding_dims = 16
        self.embedding_layers = 1
        self.embedding_is_residual = False
        self.embedding_use_se = False

        self.embedding_feature_counts = {
            "global": 2,
            "factory": 6,
            "unit": 4,
            "map": 6,
        }
        self.embedding_feature_count = sum(self.embedding_feature_counts.values())

        self.embedding = nn.Sequential(
            nn.Conv2d(self.embedding_feature_count, self.embedding_dims, kernel_size=1, stride=1, padding="same", bias=True),
            nn.BatchNorm2d(self.embedding_dims),
            nn.GELU()
        )
        self.embedding2 = nn.Sequential(
            nn.Conv2d(self.embedding_dims, self.embedding_dims, kernel_size=1, stride=1, padding="same", bias=True),
            nn.BatchNorm2d(self.embedding_dims),
            nn.GELU(),
        )

        # self.embedding_residual = nn.Sequential(
        #     nn.Conv2d(self.embedding_dims, self.embedding_dims, kernel_size=3, stride=1, padding="same", bias=True),
        #     nn.BatchNorm2d(self.embedding_dims),
        #     nn.GELU(),

        #     nn.Conv2d(self.embedding_dims, self.embedding_dims, kernel_size=3, stride=1, padding="same", bias=True),
        #     nn.BatchNorm2d(self.embedding_dims),
        #     nn.GELU()
        # )
            

        # SPATIAL INFORMATION

        self.spatial_embedding_feature_count = self.embedding_dims
        self.spatial_embedding_dim = 4

        # close distance
        """
        Units should be able to see close objects.
        """
        self.small_distance_feature_count = self.spatial_embedding_feature_count
        self.small_distance_dim = self.spatial_embedding_dim
        self.small_distance_net = nn.Sequential(
            # can see 1 distance away
            nn.Conv2d(self.small_distance_feature_count, self.small_distance_dim, kernel_size=3, stride=1, padding="same", bias=True),
            nn.BatchNorm2d(self.small_distance_dim),
            nn.GELU(),
        )

        # large distance
        """
        Units should be able to see far away objects.
        """

        self.large_distance_feature_count = self.spatial_embedding_feature_count
        self.large_distance_dim = self.spatial_embedding_dim
        self.large_distance_net = nn.Sequential(
            # can see 5 distance away
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),  # +1 distance
            nn.Conv2d(self.large_distance_feature_count, self.large_distance_dim, kernel_size=5, stride=1, padding="same", bias=True, dilation=2),  # +2*(5//2) distance
            nn.BatchNorm2d(self.large_distance_dim),
            nn.GELU(),
        )


        # COMBINED

        self.combined_feature_count = self.embedding_dims + self.spatial_embedding_dim
        self.combined_feature_dim = 16
        self.combined_net = nn.Sequential(
            nn.Conv2d(self.combined_feature_count, self.combined_feature_dim, kernel_size=1, stride=1, padding="same", bias=True),
            nn.BatchNorm2d(self.combined_feature_dim),
            nn.GELU(),
        )
        self.combined_net2 = nn.Sequential(
            # nn.Conv2d(self.combined_feature_dim, self.combined_feature_dim, kernel_size=1, stride=1, padding="same", bias=True),
            # nn.BatchNorm2d(self.combined_feature_dim),
            # nn.GELU(),

            nn.Identity(),
        )


        # FINAL

        # critic
        self.critic_feature_count = self.combined_feature_dim
        # self.critic_dim = 4
        self.critic_head = nn.Sequential(
            nn.Conv2d(self.critic_feature_count, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )

        # factory
        self.factory_feature_count = self.combined_feature_dim
        self.factory_head = nn.Sequential(
            nn.Linear(self.factory_feature_count, ActDims.factory_act, bias=True),
        )

        # unit
        self.unit_feature_count = self.combined_feature_dim
        self.unit_dim = self.combined_feature_dim
        self.unit_net = nn.Sequential(
            nn.Identity()
        )

        # act type
        self.act_type_feature_count = self.unit_dim
        self.unit_act_type_net = nn.Sequential(
            nn.Linear(self.act_type_feature_count, len(UnitActType), bias=True),
        )

        # params
        self.param_heads = nn.ModuleDict({
            unit_act_type.name: nn.ModuleDict({
                "direction": nn.Linear(self.unit_dim, ActDims.direction, bias=True),
                "resource": nn.Linear(self.unit_dim, ActDims.resource, bias=True),
                "amount": nn.Linear(self.unit_dim, ActDims.amount, bias=True),
                "repeat": nn.Linear(self.unit_dim, ActDims.repeat, bias=True),
            }) for unit_act_type in UnitActType
        })


    def forward(self, global_feature, map_feature, factory_feature, unit_feature, location_feature, va, action=None):
        B, _, H, W = map_feature.shape
        max_group_count = 1000

        # Embeddings
        global_feature = global_feature[..., None, None].expand(-1, -1, H, W)
        all_features = torch.cat([global_feature, factory_feature, unit_feature, map_feature], dim=1)
        features_embedded = self.embedding(all_features)
        features_embedded = self.embedding2(features_embedded)
        
        # _features_embedded = self.embedding_residual(features_embedded)
        # features_embedded = features_embedded + _features_embedded

        small_distance = self.small_distance_net(features_embedded)
        large_distance = self.large_distance_net(features_embedded)
        aggregated_distance = (small_distance + large_distance) / 2

        # Combined
        combined_feature = torch.cat([features_embedded, aggregated_distance], dim=1)
        combined_feature = self.combined_net(combined_feature)
        combined_feature = self.combined_net2(combined_feature)

        # Valid actions
        unit_act_type_va = torch.stack(
            [
                va['move'].flatten(1, 2).any(1),
                va['transfer'].flatten(1, 3).any(1),
                va['pickup'].flatten(1, 2).any(1),
                va['dig'].any(1),
                va['self_destruct'].any(1),
                va['recharge'].any(1),
                va['do_nothing'],
            ],
            axis=1,
        )

        # Locations
        factory_pos = torch.where(va['factory_act'].any(1))
        unit_pos = torch.where(unit_act_type_va.any(1))
        def _gather_from_map(x, pos):
            return x[pos[0], ..., pos[1], pos[2]]
        factory_ids = _gather_from_map(location_feature[:, 0], factory_pos).int()
        unit_ids = (_gather_from_map(location_feature[:, 1], unit_pos)).int()
        unit_indices = unit_pos[0] * max_group_count + unit_ids
        if len(unit_indices) > 0:
            assert unit_indices.max(dim=-1)[0] < (B * max_group_count)
            assert unit_indices.min(dim=-1)[0] >= 0
        factory_indices = factory_pos[0] * max_group_count + factory_ids

        # Critic
        critic_value = torch.zeros((B, max_group_count), device=combined_feature.device)
        critic_value = critic_value.view(-1)

        _critic_value = self.critic(combined_feature)
        _critic_value_unit = _gather_from_map(_critic_value[:, 0], unit_pos)
        if len(unit_indices) > 0:
            critic_value.scatter_add_(0, unit_indices, _critic_value_unit)
        _critic_value_factory = _gather_from_map(_critic_value[:, 0], factory_pos)
        if len(factory_indices) > 0:
            critic_value.scatter_add_(0, factory_indices, _critic_value_factory)

        critic_value = critic_value.view(B, max_group_count)

        # Actor
        logp, action, entropy = self.actor(combined_feature, va, factory_pos, unit_act_type_va, unit_pos, factory_ids, max_group_count, unit_indices, factory_indices, action)

        return logp, critic_value, action, entropy

    def critic(self, combined_feature):
        critic_value = self.critic_head(combined_feature)
        return critic_value

    def actor(self, combined_feature, va, factory_pos, unit_act_type_va, unit_pos, factory_ids, max_group_count, unit_indices, factory_indices, action=None):
        B, _, H, W = combined_feature.shape

        logp = torch.zeros((B, max_group_count), device=combined_feature.device)
        logp = logp.view(-1)
        entropy = torch.zeros((B, max_group_count), device=combined_feature.device)
        entropy = entropy.view(-1)
        output_action = {}

        def _gather_from_map(x, pos):
            return x[pos[0], ..., pos[1], pos[2]]

        def _put_into_map(emb, pos):
            shape = (B, ) + emb.shape[1:] + (H, W)
            map = torch.zeros(shape, dtype=emb.dtype, device=emb.device)
            map[pos[0], ..., pos[1], pos[2]] = emb
            return map

        # factory actor
        factory_emb = _gather_from_map(combined_feature, factory_pos)

        factory_va = _gather_from_map(va['factory_act'], factory_pos)
        factory_action = action and _gather_from_map(action['factory_act'], factory_pos)
        factory_logp, factory_action, factory_entropy = self.factory_actor(
            factory_emb,
            factory_va,
            factory_action,
        )

        if len(factory_indices) > 0:
            logp.scatter_add_(0, factory_indices, factory_logp)
            entropy.scatter_add_(0, factory_indices, factory_entropy)
        
        output_action['factory_act'] = _put_into_map(factory_action, factory_pos)

        # unit actor
        unit_emb = _gather_from_map(combined_feature, unit_pos)
        unit_emb = self.unit_net(unit_emb)
        
        unit_va = {
            'act_type': _gather_from_map(unit_act_type_va, unit_pos),
            'move': _gather_from_map(va['move'], unit_pos),
            'transfer': _gather_from_map(va['transfer'], unit_pos),
            'pickup': _gather_from_map(va['pickup'], unit_pos),
            'dig': _gather_from_map(va['dig'], unit_pos),
            'self_destruct': _gather_from_map(va['self_destruct'], unit_pos),
            'recharge': _gather_from_map(va['recharge'], unit_pos),
            'do_nothing': _gather_from_map(va['do_nothing'], unit_pos),
        }

        unit_action = action and _gather_from_map(action['unit_act'], unit_pos)
        unit_logp, unit_action, unit_entropy = self.unit_actor(
            unit_emb,
            unit_emb,
            unit_va,
            unit_action,
        )

        if len(unit_indices) > 0:
            logp.scatter_add_(0, unit_indices, unit_logp)
            entropy.scatter_add_(0, unit_indices, unit_entropy)

        output_action['unit_act'] = _put_into_map(unit_action, unit_pos)

        logp = logp.view(B, max_group_count)
        entropy = entropy.view(B, max_group_count)

        return logp, output_action, entropy


    def factory_actor(self, x, va, action=None):
        logits = self.factory_head(x)
        logp, output_action, entropy = sample_from_categorical(logits, va, action)
        return logp, output_action, entropy


    def unit_actor(self, x_act, x_param,  va, action=None):
        act_type_logp, act_type, act_type_entropy = sample_from_categorical(
            self.unit_act_type_net(x_act),
            va['act_type'],
            action[:, UnitActChannel.TYPE] if action is not None else None
        )
        logp = act_type_logp
        entropy = act_type_entropy
        output_action = torch.zeros((x_act.shape[0], len(UnitActChannel)), device=x_act.device)

        for type in UnitActType:
            mask = (act_type == type)
            move_logp, move_action, move_entropy = self.get_unit_action(
                x_param[mask],
                va[type.name.lower()][mask],
                type,
                action[mask] if action is not None else None,
            )
            logp[mask] += move_logp
            entropy[mask] += move_entropy
            output_action[mask] = move_action

        return logp, output_action, entropy


    def get_unit_action(self, x, va, unit_act_type, action=None):
        n_units = x.shape[0]
        unit_idx = torch.arange(n_units, device=x.device)

        output_action = torch.zeros((n_units, len(UnitActChannel)), device=x.device)

        params, param_logp, param_entropy = self.get_params(unit_idx, x, unit_act_type, va, action)

        output_action[:, UnitActChannel.TYPE] = unit_act_type
        output_action[:, UnitActChannel.N] = 1
        for param_name in ["direction", "resource", "amount", "repeat"]:
            if param_name in params and params[param_name] is not None:
                output_action[:, UnitActChannel[param_name.upper()]] = params[param_name]

        return param_logp, output_action, param_entropy


    def get_params(self, unit_idx, x, action_type, va, action=None):
        n_units = x.shape[0]
        
        direction, resource, amount, repeat = None, None, None, None
        direction_logp = torch.zeros(n_units, device=x.device)
        resource_logp = torch.zeros(n_units, device=x.device)
        amount_logp = torch.zeros(n_units, device=x.device)
        repeat_logp = torch.zeros(n_units, device=x.device)
        direction_entropy = torch.zeros(n_units, device=x.device)
        resource_entropy = torch.zeros(n_units, device=x.device)
        amount_entropy = torch.zeros(n_units, device=x.device)
        repeat_entropy = torch.zeros(n_units, device=x.device)

        # direction
        if action_type in [UnitActType.MOVE, UnitActType.TRANSFER]:
            direction, direction_logp, direction_entropy = self.get_direction_param(x, va, action_type, action)

        # resource
        if action_type in [UnitActType.TRANSFER, UnitActType.PICKUP]:
            resource, resource_logp, resource_entropy = self.get_resource_param(x, va, action_type, unit_idx, direction, action)

        # amount
        if action_type in [UnitActType.TRANSFER, UnitActType.PICKUP, UnitActType.RECHARGE]:
            amount, amount_logp, amount_entropy = self.get_amount_param(x, action_type, action)

        # repeat
        if action_type in [UnitActType.MOVE, UnitActType.DIG]:
            repeat, repeat_logp, repeat_entropy = self.get_repeat_param(x, va, action_type, unit_idx, direction, resource, action)

        return {"direction": direction, "resource": resource, "amount": amount, "repeat": repeat}, \
                direction_logp + resource_logp + amount_logp + repeat_logp, \
                direction_entropy + resource_entropy + amount_entropy + repeat_entropy


    def get_direction_param(self, x, va, action_type, action=None):
        direction_va = va.flatten(2).any(dim=-1)
        direction_head = self.param_heads[action_type.name]['direction']
        direction_logp, direction, direction_entropy = sample_from_categorical(
            direction_head(x),
            direction_va,
            action[:, UnitActChannel.DIRECTION] if action is not None else None,
        )
        return direction, direction_logp, direction_entropy


    def get_resource_param(self, x, va, action_type, unit_idx, direction, action=None):
        if action_type in [UnitActType.TRANSFER]:
            resource_va = va[unit_idx, direction].flatten(2).any(-1)
        elif action_type in [UnitActType.PICKUP]:
            resource_va = va.flatten(2).any(-1)
        else:
            resource_va = va
        resource_head = self.param_heads[action_type.name]['resource']
        resource_logp, resource, resource_entropy = sample_from_categorical(
            resource_head(x),
            resource_va,
            action[:, UnitActChannel.RESOURCE] if action is not None else None,
        )
        return resource, resource_logp, resource_entropy


    def get_amount_param(self, x, action_type, action=None):
        amount_head = self.param_heads[action_type.name]['amount']
        amount_logp, amount, amount_entropy = sample_from_categorical(
            amount_head(x),
            torch.tensor(True, device=x.device),
            action[:, UnitActChannel.AMOUNT] if action is not None else None,
        )
        return amount, amount_logp, amount_entropy


    def get_repeat_param(self, x, va, action_type, unit_idx, direction, resource, action=None):
        if action_type in [UnitActType.MOVE]:
            repeat_va = va[unit_idx, direction]
        elif action_type in [UnitActType.PICKUP]:
            repeat_va = va[unit_idx, resource]
        elif action_type in [UnitActType.TRANSFER]:
            repeat_va = va[unit_idx, direction, resource]
        else:
            repeat_va = va

        repeat_head = self.param_heads[action_type.name]['repeat']
        repeat_logp, repeat, repeat_entropy = sample_from_categorical(
            repeat_head(x),
            repeat_va,
            action[:, UnitActChannel.REPEAT] if action is not None else None,
        )

        return repeat, repeat_logp, repeat_entropy
