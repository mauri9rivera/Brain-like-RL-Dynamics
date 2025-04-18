import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.callbacks import BaseCallback


class Policy(th.nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device, sigma=0.1):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = 1
        self.sigma = sigma
        self.noise = th.zeros(output_dim, device = device)
        self.timestep_counter = 0
        self.resample_threshold = np.random.randint(16, 25)
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
        self.resample_threshold = np.random.randint(16, 25)
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden
    
class Critic(th.nn.Module):
    def __init__(self, input_size, device):
        super().__init__()
        self.device = device
        
        # Define network layers properly
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 64)
        self.fc3 = nn.Linear(64, 1)
        
        # Activation functions
        self.tanh = nn.Tanh()

        # Apply orthogonal initialization
        self._initialize_weights()

        self.to(self.device)

    def _initialize_weights(self):
        """Apply orthogonal initialization with gain=1 and bias=0."""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(layer.weight, gain=1)  # Orthogonal init with gain=1
            nn.init.zeros_(layer.bias)  # Bias = 0

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return self.fc3(x)

class ACNetwork(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, device, **kwargs):
        self._device = device
        kwargs.pop("device", None)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self._build_network()
        self.hidden_states = None  # Stores hidden states during rollout
        self.log_std = nn.Parameter(th.zeros(self.action_dim), requires_grad=True)
        self.name = 'DefaultPPO'
        self.to(self._device)

    def _build_network(self):
        input_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        self.hidden_dim = 64  # Match your GRU hidden size

        # Actor (GRU-based) and Critic
        self.actor = Policy(input_dim, self.hidden_dim, self.action_dim, self._device)
        self.critic = Critic(input_dim, self._device)

        # Initialize action distribution (adjust based on your action space)
        self.action_dist = DiagGaussianDistribution(self.action_dim)

    def forward(self, obs, deterministic=False):
        # Handle hidden states (simplified for example)
        batch_size = obs.shape[0]
        if self.hidden_states is None:
            self.hidden_states = self.actor.init_hidden(batch_size)
        
        # Forward through actor and critic
        mean_actions, new_hidden = self.actor(obs, self.hidden_states)
        values = self.critic(obs)
        self.hidden_states = new_hidden.detach()  # Detach to prevent BPTT

        self.action_dist.proba_distribution(mean_actions, self.log_std)

        # Sample from distribution
        actions = self.action_dist.mode() if deterministic else self.action_dist.sample()
        log_prob = self.action_dist.log_prob(actions)

        # Create action distribution (example for continuous actions)
        #log_std = th.ones_like(actions) * self.log_std
        #self.action_dist = DiagGaussianDistribution(actions, log_std)
        #return actions, values, self.action_dist.log_prob(actions)

        return actions, values, log_prob

    def evaluate_actions(self, obs, actions, hidden_states):
        # Evaluate actions for training
        action_pred, _ = self.actor(obs, hidden_states)
        values = self.critic(obs)
        '''
        log_std = th.ones_like(action_pred) * self.log_std
        dist = DiagGaussianDistribution(action_pred)
        dist.log_std = log_std
        print(f'This is dist: {dist}')
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy
        '''
        self.action_dist.proba_distribution(action_pred, self.log_std)
        log_prob = self.action_dist.log_prob(actions)
        entropy = self.action_dist.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs):
        return self.critic(obs)

class DummyExtractor(nn.Module):
    def __init__(self, latent_dim_pi: int, latent_dim_vf: int):
        super().__init__()
        # SB3 only needs these for sizing:
        self.latent_dim_pi = latent_dim_pi
        self.latent_dim_vf = latent_dim_vf

    def forward(self, features):
        # SB3 never actually calls it during inference for MLP policies,
        # but it expects two outputs.
        return features, features

class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    SB3 policy that uses your custom Policy (GRU + noise) as actor
    and your custom Critic as the value network.
    Tracks GRU hidden states and respects a 'device' argument.
    """
    def __init__(self, observation_space, action_space, lr_schedule, device="cpu", name="DefaultPPO", **kwargs):
        # 0) Store device & name
        self._device = th.device(device)
        self.name    = name

        # 1) Define dims *before* super().__init__
        obs_dim    = observation_space.shape[0]
        act_dim    = action_space.shape[0]
        self.hidden_dim = 64             # <-- must exist for _build_mlp_extractor
        self.input_dim  = obs_dim
        self.action_dim = act_dim
        self.hidden_states = None        # also ready if callbacks reference it

        # 2) Pop any kwarg collisions, then build base
        kwargs.pop("device", None)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # 3) Now build your actual actor & critic
        self.actor = Policy(self.input_dim,
                            self.hidden_dim,
                            self.action_dim,
                            self._device)

        self.critic = Critic(self.input_dim, self._device)

        # 4) Log‐std & move to device
        self.log_std = nn.Parameter(th.zeros(self.action_dim,
                                             device=self._device))
        self.to(self._device)

    def _build_mlp_extractor(self):
        # Replace the old Dummy with our Module‐based one:
        self.mlp_extractor = DummyExtractor(
            latent_dim_pi=self.hidden_dim,
            latent_dim_vf=self.hidden_dim,
        )
    def forward(self, obs, deterministic=False):
        """
        Given observations, returns:
          - actions  (batch, act_dim)
          - values   (batch, 1)
          - log_prob (batch,)
        """
        # ensure obs on correct device
        obs = obs.to(self.device)
        
        # init hidden if first call
        batch_size = obs.shape[0]
        if self.hidden_states is None:
            self.hidden_states = self.actor.init_hidden(batch_size)
        
        # actor forward
        mean_actions, new_hidden = self.actor(obs, self.hidden_states)
        self.hidden_states = new_hidden.detach()  # detach BPTT
        
        # critic forward
        values = self.critic(obs)
        
        # build distribution
        log_std = self.log_std.unsqueeze(0).expand_as(mean_actions)
        dist = DiagGaussianDistribution(self.action_dim)
        dist = dist.proba_distribution(mean_actions, log_std)
        
        # sample or mode
        actions = dist.mode() if deterministic else dist.sample()
        log_prob = dist.log_prob(actions)
        
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions, hidden_states=None):
        """
        Used by SB3 to compute loss:
          - returns values, log_prob(actions), entropy
        """
        obs = obs.to(self.device)
        batch_size = obs.shape[0]

         # Handle GRU hidden state properly
        if hidden_states is not None:
            h = hidden_states
        else:
            # fallback: make sure self.hidden_states has correct batch size
            if self.hidden_states is None or self.hidden_states.shape[1] != batch_size:
                h = th.zeros(1, batch_size, self.actor.hidden_dim, device=self.device)
            else:
                h = self.hidden_states

        mean_actions, _ = self.actor(obs, h)
        values = self.critic(obs)
        
        log_std = self.log_std.unsqueeze(0).expand_as(mean_actions)
        dist = DiagGaussianDistribution(self.action_dim)
        dist = dist.proba_distribution(mean_actions, log_std)
                
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs):
        obs = obs.to(self.device)
        return self.critic(obs)

class LossTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.losses = []

    def _on_step(self) -> bool:
        # SB3 doesn't expose loss directly, so we approximate it
        # You could modify PPO class to store actual loss if needed
        return True

    def _on_rollout_end(self) -> None:
        # This function is called after each policy update
        # No real "loss", so you could simulate with mean reward or value loss
        ep_rewards = self.locals["rollout_buffer"].rewards
        avg_reward = sum(ep_rewards) / len(ep_rewards)
        self.losses.append(-avg_reward)  # Just an indicator proxy for loss


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

# Example usage:
if __name__ == "__main__":

    DEVICE = "cuda:0" if th.cuda.is_available() else "cpu"
    n_batch = 1000
    batch_size = 128
    total_timesteps = n_batch * batch_size
    policy_kwargs = {"device": DEVICE, "name": "PPO2"}

    from environment import *
    from utils import *
    
    arm = Arm('arm26')
    env = RandomTargetReach2(
        effector=arm.effector, 
        obs_noise=0.0,
        proprioception_noise=0.0,
        vision_noise=0.0,
        action_noise=0.0
        )
    
    model = PPO(
        policy=CustomActorCriticPolicy,
        env=env,
        policy_kwargs=policy_kwargs,
        n_steps=2048,
        batch_size=128,
        learning_rate=3e-4,
        n_epochs=10,
        clip_range=0.2,
        max_grad_norm=0.5,
        verbose=1,
    )

    # Create callback to track loss
    callback = LossTrackingCallback()

    # 5) Train
    model.learn(total_timesteps=total_timesteps, callback=callback)
    plot_loss(callback.losses)
    evaluate_pretrained(model.po, env, batch_size)

    # -----------------------
    # Evaluate & Save
    # -----------------------
    policy = model.policy
    try:
        save_model(policy, callback.losses, env)
        evaluate_pretrained(policy, env, batch_size)
    except Exception as e:
        print("Evaluation failed:", e)
        save_model(policy, callback.losses, env)
    