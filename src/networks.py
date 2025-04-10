import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.distributions import DiagGaussianDistribution

class Policy(th.nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device, sigma=0.1):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = 1
        self.sigma = sigma
        self.noise = th.zeros(output_dim, device = device)
        self.timestep_counter = 0
        self.resample_threshold = np.random(16, 24)
        self.gru = nn.GRU(input_dim, hidden_dim, 1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

        # Apply custom initialization
        for name, param in self.named_parameters():
            
            if "gru" in name:
                if "weight_ih" in name:
                    nn.init.orthogonal_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias_ih" in name:
                    nn.init.zeros_(param)
                elif "bias_hh" in name:
                    nn.init.zeros_(param)
            elif "fc" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, -10.0)
            else:
                raise ValueError(f"Unexpected parameter: {name}")
        
        self.to(device)

    def forward(self, x, h0):
        y, h = self.gru(x.unsqueeze(1), h0)  
        u = self.sigmoid(self.fc(y.squeeze(1))) 

        # Apply periodic Gaussian noise
        self.timestep_counter += 1
        if self.timestep_counter >= self.resample_threshold:
            self.resample_noise()

        return u + self.noise, h
    
    def resample_noise(self):

        self.noise = th.randn_like(self.noise) * self.sigma
        self.timestep_counter = 0
        self.resample_threshold = np.random.uniform(16, 24)
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden
    
class Critic(th.nn.Module):
    def __init__(self, input_size, device):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Tanh(100, 64)
        self.fc3 = nn.Tanh(64, 64)
        self.fc4 = nn.Linear(64, 1)
        self.device = device

        # Apply orthogonal initialization
        self._initialize_weights()

        self.to(device)

    def _initialize_weights(self):
        """Apply orthogonal initialization with gain=1 and bias=0."""
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.orthogonal_(layer.weight, gain=1)  # Orthogonal init with gain=1
            nn.init.zeros_(layer.bias)  # Bias = 0

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.fc4(x)

class ACNetwork(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self._build_network()
        self.hidden_states = None  # Stores hidden states during rollout

    def _build_network(self):
        input_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        self.hidden_dim = 64  # Match your GRU hidden size

        # Actor (GRU-based) and Critic
        self.actor = Policy(input_dim, self.hidden_dim, self.action_dim, self.device)
        self.critic = Critic(input_dim, self.device)

        # Initialize action distribution (adjust based on your action space)
        self.action_dist = DiagGaussianDistribution(self.action_dim)

    def forward(self, obs, deterministic=False):
        # Handle hidden states (simplified for example)
        batch_size = obs.shape[0]
        if self.hidden_states is None:
            self.hidden_states = self.actor.init_hidden(batch_size)
        
        # Forward through actor and critic
        actions, new_hidden = self.actor(obs, self.hidden_states)
        values = self.critic(obs)
        self.hidden_states = new_hidden.detach()  # Detach to prevent BPTT

        # Create action distribution (example for continuous actions)
        log_std = th.ones_like(actions) * self.log_std
        self.action_dist = DiagGaussianDistribution(actions, log_std)
        return actions, values, self.action_dist.log_prob(actions)

    def evaluate_actions(self, obs, actions, hidden_states):
        # Evaluate actions for training
        action_pred, _ = self.actor(obs, hidden_states)
        values = self.critic(obs)
        log_std = th.ones_like(action_pred) * self.log_std
        dist = DiagGaussianDistribution(action_pred, log_std)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs):
        return self.critic(obs)

default_ppo_kwargs = {
    "policy": ACNetwork,  # Use the custom network
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 128,  # Batch size set to 128
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "policy_kwargs": {"enable_critic_lstm": True},  # Enable GRU in policy
}
