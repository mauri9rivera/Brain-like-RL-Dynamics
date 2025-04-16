import os
import torch as th
import json 
from datetime import datetime
import matplotlib.pyplot as plt
import motornet as mn


def save_model(model, losses, env):

    #Create model directory
    base_dir = os.path.join("outputs", "savedmodels")
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

    plt.figure(figsize=(10,3))

    plt.subplot(1,2,1)
    plt.ylim([-1.1, 1.1])
    plt.xlim([-1.1, 1.1])
    plotor(axis=plt.gca(), cart_results=xy)
    plt.scatter(target_x, target_y)

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
    terminated = False
    xy = [info["states"]["fingertip"][:, None, :]]
    tg = [info["goal"][:, None, :]]

    # Run evaluation episode
    while not terminated:
        action, _, _ = policy(obs, deterministic=True)  # Deterministic actions
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

# Example usage:
if __name__ == "__main__":

    # Create your effector (for instance, using RigidTendonArm26 with MujocoHillMuscle)
    arm = mn.effector.RigidTendonArm26(muscle=mn.muscle.MujocoHillMuscle())

    from environment import RandomTargetReach
    from networks import *
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

    net = load_model(env, ACNetwork, './outputs/savedmodels/2025-04-15/DefaultPPO/weights')
    evaluate_pretrained(net, env, 100)
    