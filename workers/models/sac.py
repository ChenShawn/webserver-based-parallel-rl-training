import os
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .utils import soft_update, hard_update
from .model import GaussianPolicy, QNetwork, DeterministicPolicy


class SAC(object):
    def __init__(self, num_inputs, action_space, args):
        self.alpha = args.alpha
        self.policy_type = args.policy
        self.device = torch.device('cpu')
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)


    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]


    def get_value(self, state_batch):
        state_batch = torch.FloatTensor(state_batch).to(self.device).unsqueeze(0)
        action, log_pi, _ = self.policy.sample(state_batch)
        qf1, qf2 = self.critic_target(state_batch, action)
        min_qf_target = torch.min(qf1, qf2) - self.alpha * log_pi
        return min_qf_target.detach().cpu().numpy()[0]

    
    def load_model(self, actor_path=None, critic_path=None):
        """load_model
            Load model parameters
            Make sure the actor_path and critic_path are valid before loading
        """
        if actor_path is None:
            actor_path = './train/actor.pth'
        if critic_path is None:
            critic_path = './train/critic.pth'
        self.policy.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))