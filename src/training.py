import os
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt

from networks import ACNetwork
from environment import CustomReachEnv

# Define the L1 loss between trajectories.
def l1_loss(x, y):
    """Compute L1 loss across time and features."""
    return th.mean(th.sum(th.abs(x - y), dim=-1))

def pretrain(env, device="cpu", n_batch=6000, batch_size=32):
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
    
    # Instantiate the custom ACNetwork using environment properties.
    # (lr_schedule is a dummy lambda for now.)
    lr_schedule = lambda epoch: 3e-4  
    policy = ACNetwork(env.observation_space, env.action_space, lr_schedule, device=device)
    optimizer = th.optim.Adam(policy.parameters(), lr=1e-3)
    
    losses = []
    interval = 250  # Logging interval
    
    for batch in range(n_batch):
        # Initialize hidden state (for GRU) for current batch.
        h = policy.actor.init_hidden(batch_size=batch_size)
        
        # Reset the custom environment with specified batch size.
        obs, info = env.reset(options={"batch_size": batch_size})
        terminated = False
        
        # Record initial fingertip positions and target positions.
        xy_list = [info["states"]["fingertip"][:, None, :]]
        tg_list = [info["goal"][:, None, :]]
        
        # Simulate full episode until termination.
        while not terminated:
            # Get action, value, and log probability from the network.
            # (Only the actor part is needed for producing actions during pretraining.)
            action, _, _ = policy(obs, deterministic=False)
            # Step the environment.
            obs, reward, terminated, truncated, info = env.step(action=action)
            
            # Record current fingertip position and target.
            xy_list.append(info["states"]["fingertip"][:, None, :])
            tg_list.append(info["goal"][:, None, :])
        
        # Concatenate the list over timesteps.
        xy = th.cat(xy_list, axis=1)
        tg = th.cat(tg_list, axis=1)
        
        # Compute L1 loss between the fingertip trajectory and target trajectory.
        loss = l1_loss(xy, tg)
        
        # Perform backward pass and optimization.
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients with max norm = 1.0 (as per training procedure).
        th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        if (batch % interval == 0) and (batch != 0):
            mean_loss = sum(losses[-interval:]) / interval
            print(f"Batch {batch}/{n_batch} done, mean loss: {mean_loss:.4f}")
    
    # Plot the training loss log.
    plt.figure(figsize=(8, 3))
    plt.semilogy(losses)
    plt.xlabel("Batch #")
    plt.ylabel("L1 Loss")
    plt.title("Pretraining Loss")
    plt.tight_layout()
    plt.show()
    
    return policy, losses

# Example usage:
if __name__ == "__main__":
    import motornet as mn
    # Create your effector (for instance, using RigidTendonArm26 with MujocoHillMuscle)
    arm = mn.effector.RigidTendonArm26(muscle=mn.muscle.MujocoHillMuscle())
    # Instantiate the custom reach environment.
    env = CustomReachEnv(effector=arm)
    
    # Pretrain the network.
    trained_policy, training_losses = pretrain(env, device="cpu", n_batch=6000, batch_size=32)
