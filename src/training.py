import torch as th
import torch.nn.functional as F
import motornet as mn
import matplotlib.pyplot as plt
import numpy as np

from environment import create_defaultReachTask, RandomTargetReach
from networks import ACNetwork, Policy, Critic, default_ppo_kwargs
from utils import *

def pretrain(env, device="cpu", n_batch=6000, batch_size=128):
    """
    Pretrain the custom Actor-Critic network on a random reach task
    using your CustomReachEnv and ACNetwork objects.
    
    The training follows MotorNet's tutorial:
      - At the beginning of each batch, the hidden state is initialized.
      - The environment is reset with a batch_size, which sets the starting joint state
        to be 1cm from a randomly drawn target in Cartesian space.
      - A full episode is simulated: at each step, the policy produces an action,
        which is passed to the env.step() method. The fingertip positions (xy) and
        target positions (tg) are stored.
      - Once the episode ends, an L1 loss is computed between the concatenated trajectory
        of fingertip positions and the target positions.
      - The gradients are clipped (max norm=1.0) and the optimizer steps.
    
    Returns:
        policy: the trained ACNetwork object.
        losses: a list of loss values (one per batch).
    """

    device = th.device(device)

    # PPO hyperparameters
    gamma = 0.99
    clip_epsilon = 0.2
    update_epochs = 10

    # Use parameters from default_ppo_kwargs for consistent hyperparameters.
    learning_rate = 3e-4  # example value
    lr_schedule = lambda epoch: learning_rate

    # Instantiate the custom ACNetwork using environment properties.
    policy = ACNetwork(env.observation_space, env.action_space, lr_schedule, device)
    policy.train()  # set to training mode
    optimizer = th.optim.Adam(policy.parameters(), lr=1e-3)
    
    losses = []
    interval = 250  # Logging interval

    for batch in range(n_batch):

        # Initialize hidden state (for GRU) for current batch.
        h = policy.actor.init_hidden(batch_size=batch_size)
        
        obs, info = env.reset(options={"batch_size": batch_size})
        
        terminated = False
        
        # Record initial fingertip positions and target positions.
        xy_list = [info["states"]["fingertip"][:, None, :]]
        tg_list = [info["goal"][:, None, :]]
        # Lists to store trajectory data.
        obs_list = []
        actions_list = []
        logp_list = []
        rewards_list = []
        values_list = []
        
        # Simulate full episode until termination.
        while not terminated:
            
          obs_list.append(obs.detach())

          # Forward pass through ACNetwork:
          action, value, logp = policy(obs, deterministic=False)
          actions_list.append(action.detach())
          logp_list.append(logp.detach())
          values_list.append(value.detach())

          #Step in environment
          obs, reward, terminated, truncated, info = env.step(action=action)
          
          # Here reward is assumed to be an instantaneous scalar per sample.
          rewards_list.append(reward.detach())
          # Record current fingertip position and target.
          xy_list.append(info["states"]["fingertip"][:, None, :])
          tg_list.append(info["goal"][:, None, :])
        
        # Concatenate the list over timesteps.
        xy = th.cat(xy_list, axis=1)
        tg = th.cat(tg_list, axis=1)
        
        # Convert trajectory lists to tensors
        obs_tensor = th.cat([o.unsqueeze(0) for o in obs_list], dim=0)        # shape: (T, batch_size, obs_dim)
        actions_tensor = th.cat([a.unsqueeze(0) for a in actions_list], dim=0)    # shape: (T, batch_size, act_dim)
        logp_tensor = th.cat([lp.unsqueeze(0) for lp in logp_list], dim=0)        # shape: (T, batch_size, act_dim) or (T, batch_size)
        rewards_tensor = th.cat([r.unsqueeze(0) for r in rewards_list], dim=0)    # shape: (T, batch_size, 1)
        values_tensor = th.cat([v.unsqueeze(0) for v in values_list], dim=0)      # shape: (T, batch_size, 1)
        
        T = rewards_tensor.shape[0]

        # Compute discounted returns, simple implementation over trajectory:
        returns = th.zeros_like(rewards_tensor)
        future_return = th.zeros((batch_size, 1), device=device)
        for t in reversed(range(T)):
            future_return = rewards_tensor[t] + gamma * future_return
            returns[t] = future_return
        
        # Compute advantages
        advantages = returns - values_tensor
        
        # Flatten trajectory dimensions for training.
        flat_obs = obs_tensor.reshape(-1, obs_tensor.shape[-1])
        flat_actions = actions_tensor.reshape(-1, actions_tensor.shape[-1])
        flat_logp_old = logp_tensor.reshape(-1)
        flat_returns = returns.reshape(-1, 1)
        flat_advantages = advantages.reshape(-1, 1)
        
        # Perform PPO updates over several epochs.
        ppo_loss = 0.0
        value_loss = 0.0
        for epoch in range(update_epochs):
          # Re-evaluate actions on the entire rollout:
          new_value, new_logp, entropy = policy.evaluate_actions(flat_obs, flat_actions, None)
          new_value = new_value.reshape(-1, 1)
          new_logp = new_logp.reshape(-1)
          
          ratio = th.exp(new_logp - flat_logp_old)
          surr1 = ratio * flat_advantages.squeeze(-1)
          surr2 = th.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * flat_advantages.squeeze(-1)
          policy_loss = -th.mean(th.min(surr1, surr2))
          
          # Critic loss as MSE between predicted value and returns.
          critic_loss = F.mse_loss(new_value, flat_returns)
          
          total_loss = policy_loss + critic_loss
          
          optimizer.zero_grad()
          total_loss.backward()
          # Clip gradients for stability.
          th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
          optimizer.step()
          
          ppo_loss += policy_loss.item()
          value_loss += critic_loss.item()
        
        avg_loss = (ppo_loss + value_loss) / update_epochs
        losses.append(avg_loss)
        if batch % 50 == 0 and batch != 0:
          print(f"Batch {batch}/{n_batch} done, avg loss: {avg_loss:.4f}")
    
    try:

        # Final operations
        save_model(policy, losses, env)  # Save model
        evaluate_pretrained(policy, env, 100)  # Evaluation
        plot_loss(losses)  # Plotting

    except:
        save_model(policy, losses, env)
        return policy, losses

    
    
    return policy, losses

def center_out_task(env, model, n_trials=1000, device="cpu"):
   
  neural_activities = []
  trajectories = []

  for _ in range(n_trials):
     
     obs = env.reset()
     model.policy.hidden_states = None #change this
     terminated = False
     trial_activity = []
     trial_traj = []

     while not terminated:
        
      actions, _, _ = model(obs, deterministic=True)
      current_hidden = model.policy.hidden_states.squeeze().detach.cpu().numpy() #change this... ACNetwork needs policy attr and Policy needs hidden_states attr
      
      #Store neural activity
      trial_activity.append(current_hidden)

      #Step in environment
      obs, _, terminated, info = env.step(actions)

      #Store effector position
      trial_traj.append(env.get_state()) #change this

      neural_activities.append(np.array(trial_activity))
      trajectories.append(np.array(trial_traj))


# Example usage:
if __name__ == "__main__":

    # Create your effector (for instance, using RigidTendonArm26 with MujocoHillMuscle)
    arm = mn.effector.RigidTendonArm26(muscle=mn.muscle.MujocoHillMuscle())

    # Instantiate the custom reach environment.
    #env = create_defaultReachTask(arm)
    #env = CustomReachEnv(effector=arm)
    env = RandomTargetReach(
    effector=arm,
    obs_noise=0.0,
    proprioception_noise=0.0,
    vision_noise=0.0,
    action_noise=0.0
    )
    
    # Pretrain the network.
    #pretrain(env)
    trained_policy, training_losses = pretrain(env, n_batch=500)
  