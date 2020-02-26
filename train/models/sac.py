import os
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .utils import soft_update, hard_update
from .model import GaussianPolicy, QNetwork, DeterministicPolicy


class SAC(object):
    def __init__(self, num_inputs, action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval

        tmp_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(tmp_device) 

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            # if self.automatic_entropy_tuning == True:
            #     self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            #     self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            #     self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            # self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)


    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]


    def update_parameters(self, dataset, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch = dataset.get_batch_data()

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        with torch.no_grad():
            next_q_value = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        negative_entropy = self.alpha * log_pi
        policy_loss = (negative_entropy - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), next_q_value.mean().item(), negative_entropy.mean()


    # Save model parameters
    def save_model(self, env_name):
        if not os.path.exists('./train/'):
            os.makedirs('./train/')
        if not os.path.exists('./train/{}/'.format(env_name)):
            os.makedirs('./train/{}/'.format(env_name))

        actor_path = "./train/{}/sac_actor.pth".format(env_name)
        critic_path = "./train/{}/sac_critic.pth".format(env_name)
        critic_target_path = "./train/{}/sac_critic_target.pth".format(env_name)

        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        # NOTE: critic_target must be saved!
        torch.save(self.critic_target.state_dict(), critic_target_path)


    # Load model parameters
    def load_model(self, env_name):
        actor_path = "./train/{}/sac_actor.pth".format(env_name)
        critic_path = "./train/{}/sac_critic.pth".format(env_name)
        critic_target_path = "./train/{}/sac_critic_target.pth".format(env_name)
        self.policy.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        self.critic_target.load_state_dict(torch.load(critic_target_path))

