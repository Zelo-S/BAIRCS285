import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO DONE?: return the action that the policy prescribes
        observation = ptu.from_numpy(observation.astype(np.float32))
        # print("Pre get_action in network")
        if self.discrete:
            action = self.logits_na(observation)
        elif not self.discrete:
            action = self.mean_net(observation) # action is of dim self.ac_dim
        # print("In get_action", action, "ok then: ", ptu.to_numpy(action))
        return ptu.to_numpy(action)

    # update/train this policy
    # Technically shouldn't be implemented.... ???
    # since we leave it to the MLPPolicySL instead???
    # I guess since the child overwrites it never actually calls this function
    # so we're good : )
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # normalize it? dunno
        # print("At start of forward, obsertation of type", type(observation))
        # observation = ptu.to_numpy(observation)
        output = self.get_action(observation)
        # print("in forward, output is: ", output)
        # assert output.shape == self.ac_dim
        return ptu.from_numpy(output) # a 1-way tensor(vector)

#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        # print("Before update success")
        # TODO DONE?: update the policy and return the loss
        predicted_actions = []
        for obs in observations:
            predicted_action = self.forward(obs)
            predicted_actions.append(ptu.to_numpy(predicted_action))

        predicted_actions = np.array(predicted_actions)
        predicted_actions = ptu.from_numpy(predicted_actions)
        predicted_actions.requires_grad = True

        actions = ptu.from_numpy(np.array(actions))
        
        predicted_actions = torch.squeeze(predicted_actions, dim=1)
        # print(f"Predicted actions shape: {predicted_actions.shape} and actions: {actions.shape}")

        loss = self.loss(predicted_actions, actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # You can add extra logging information here, but keep this line
        return {
            'Training Loss': ptu.to_numpy(loss),
        }
