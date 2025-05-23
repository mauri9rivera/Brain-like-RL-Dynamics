import os
import torch as th
import json 
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import motornet as mn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA

def save_model(model, losses, env):

    #Create model directory
    base_dir = os.path.join("../outputs", "savedmodels")
    today = datetime.now().strftime("%Y-%m-%d")
    date_dir = os.path.join(base_dir, today)
    os.makedirs(date_dir, exist_ok=True)
    model_dir = os.path.join(date_dir, model.name)
    os.makedirs(model_dir, exist_ok=True)

    weight_file = os.path.join(model_dir, "weights")
    log_file = os.path.join(model_dir, "log.json")
    cfg_file = os.path.join(model_dir, "cfg.json")

    #Saving model weights, training history and environment configuration dictionary
    th.save(model.state_dict(), weight_file)
    with open(log_file, 'w') as file:
        json.dump(losses, file)
    cfg = env.get_save_config()
    with open(cfg_file, 'w') as file:
        json.dump(cfg, file)

    print(f"Done saving model's weights, training history and env in {model_dir}")

def load_environment(cfg_file, verbose=False):

    with open(cfg_file, 'r') as file:
        cfg = json.load(file)

    if verbose:
        for k1, v1 in cfg.items():
            if isinstance(v1, dict):
                print(k1 + ":")
                for k2, v2 in v1.items():
                    if type(v2) is dict:
                        print("\t\t" + k2 + ":")
                        for k3, v3 in v2.items():
                            print("\t\t\t\t" + k3 + ": ", v3)
                    else:
                        print("\t\t" + k2 + ": ", v2)
            else:
                print(k1 + ": ", v1)

    return cfg

def load_model(env, model_class, weight_file, device='cpu'):

    model = model_class(env.observation_space, env.action_space, lambda epoch: 3e-5,device)
    state_dict = th.load(weight_file)
    model.load_state_dict(state_dict)  # Don't overwrite model here!
    model.eval()
    return model

def plot_simulations(xy, target_xy):
    
    plotor = mn.plotor.plot_pos_over_time    
    target_x = target_xy[:, -1, 0]
    target_y = target_xy[:, -1, 1]

    init_x = xy[:, 0, 0]
    init_y = xy[:, 0, 1]

    plt.figure(figsize=(10,3))

    plt.subplot(1,2,1)
    plt.ylim([-1.1, 1.1])
    plt.xlim([-1.1, 1.1])
    plotor(axis=plt.gca(), cart_results=xy)
    plt.scatter(target_x, target_y)
    plt.scatter(target_x, target_y, c="yellow", label="Target", marker="X")
    plt.scatter(init_x, init_y, c="blue", label="Initial", marker="o")

    plt.subplot(1,2,2)
    plt.ylim([-2, 2])
    plt.xlim([-2, 2])
    plotor(axis=plt.gca(), cart_results=xy - target_xy[:, :, :2])
    plt.axhline(0, c="grey")
    plt.axvline(0, c="grey")
    plt.xlabel("X distance to target")
    plt.ylabel("Y distance to target")
    plt.show()

def evaluate_pretrained(policy, env, batch_size):
    """Evaluation function with hidden state management"""
    # Reset hidden states
    policy.hidden_states = policy.actor.init_hidden(batch_size)
    
    # Initialize environment
    obs, info = env.reset(options={"batch_size": batch_size})
    terminated = np.zeros(batch_size, dtype=bool)
    xy = [info["states"]["fingertip"][:, None, :]]
    tg = [info["goal"][:, None, :]]

    # Run evaluation episode
    while not terminated.all():
        action, _, _, _ = policy(obs, deterministic=True)  # Deterministic actions
        obs, _, terminated, _, info = env.step(action)
        xy.append(info["states"]["fingertip"][:, None, :])
        tg.append(info["goal"][:, None, :])

    # Plot results
    xy = th.cat(xy, axis=1).detach().numpy()
    tg = th.cat(tg, axis=1).detach().numpy()
    plot_simulations(xy=xy, target_xy=tg)

def plot_loss(losses):
    """Plot training loss"""
    plt.figure(figsize=(8, 3))
    plt.semilogy(losses)
    plt.xlabel("Batch #")
    plt.ylabel("L1 Loss")
    plt.title("Pretraining Loss")
    plt.tight_layout()
    plt.show()

def visualize_center_out(trajectories):
    """Plot end-effector trajectories for all trials."""
    plt.figure(figsize=(10, 6))
    for traj in trajectories:
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.3, linewidth=0.5)
    plt.xlabel('X Position', fontsize=12)
    plt.ylabel('Y Position', fontsize=12)
    plt.title('Center-Out Reaching Trajectories', fontsize=14)
    plt.grid(True)
    plt.show()

def visualize_task(env, n_batch=1, is_obstacle=False):
    """Visualizes all start positions and targets with proper labels and sizing."""
    # Reset environment and get positions
    _, info = env.reset(options={"batch_size": n_batch})
    starts = info["states"]["fingertip"].cpu().numpy()[:, :2]
    targets = info["goal"].cpu().numpy()[:, :2]

    # Create plot with locked axes
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.5, 1)
    ax.set_aspect('equal', adjustable='box')

    # Add obstacle rectangle if requested
    if is_obstacle:
        rect_params = (-0.3, 0.3, 0.2, 0.4)  # xmin, ymin, xmax, ymax
        width = rect_params[2] - rect_params[0]
        height = rect_params[3] - rect_params[1]
        obstacle = Rectangle(
            (rect_params[0], rect_params[1]),
            width,
            height,
            facecolor='#CCCCCC',
            edgecolor='none',
            alpha=1.0,
            zorder=1  # Ensure rectangle is behind other elements
        )
        ax.add_patch(obstacle)

    # Plot ALL starts and targets first with labels
    ax.scatter(starts[:, 0], starts[:, 1], 
               color='blue', alpha=0.7, s=100, label='Start', zorder=3)
    ax.scatter(targets[:, 0], targets[:, 1], 
               color='red', alpha=0.7, s=50, label='Target', zorder=3)

    # Add connection lines
    for i in range(n_batch):
        ax.plot([starts[i, 0], targets[i, 0]],
                [starts[i, 1], targets[i, 1]], 
                'k--', linewidth=1, alpha=0.5, zorder=2)

    # Add labels and grid
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_title(f'Start/Target Positions (n={n_batch})')
    ax.legend()
    ax.grid(True)
    plt.show()

def visualize_trajectories(trajectories, start_pos=None, targets=None):
    """
    Visualize trajectories for any type of reaching task
    
    Args:
        trajectories: List of numpy arrays containing end effector positions
        start_pos: (Optional) Initial position(s) as numpy array (single or per-trial)
        targets: (Optional) Target positions as numpy array (single or per-trial)
    """
    plt.figure(figsize=(10, 8))
    
    # Plot all trajectories
    for i, traj in enumerate(trajectories):
        # Main trajectory path
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.4, linewidth=0.8, c='blue')
        
        # Plot start and end markers
        start = traj[0] if start_pos is None else start_pos[i] if len(start_pos) > 1 else start_pos
        end = traj[-1] if targets is None else targets[i] if len(targets) > 1 else targets
        
        plt.scatter(start[0], start[1], c='green', s=60, marker='o', edgecolors='k')
        plt.scatter(end[0], end[1], c='red', s=60, marker='s', edgecolors='k')

    # Add labels and decorations
    plt.title("Movement Trajectories", fontsize=14)
    plt.xlabel("X Position (m)", fontsize=12)
    plt.ylabel("Y Position (m)", fontsize=12)
    plt.grid(True)
    plt.axis('equal')
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Start', 
                  markersize=10, markerfacecolor='g', markeredgecolor='k'),
        plt.Line2D([0], [0], marker='s', color='w', label='Target', 
                  markersize=10, markerfacecolor='r', markeredgecolor='k')
    ]
    plt.legend(handles=legend_elements)
    
    plt.show()

    
    