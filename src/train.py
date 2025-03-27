import gym
import torch
import motornet as mn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from rnn_policy import RNNPolicy
from critic import CriticNetwork

# Initialize environment
env = MotorNetEnv()

# Define PPO model
model = PPO(
    ActorCriticPolicy,
    env,
    batch_size=128,
    policy_kwargs={"net_arch": {"pi": [64], "vf": [100, 64, 64]}},  # Policy & Critic sizes
    verbose=1
)

# Train the model
model.learn(total_timesteps=100000)

# Save trained model
model.save("ppo_motor_control")
