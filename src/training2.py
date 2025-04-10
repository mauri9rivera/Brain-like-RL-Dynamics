import torch as th
import motornet as mn
import matplotlib as plt

from environment import CustomReachEnv
from networks import ACNetwork, Policy, Critic

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
    plotor(axis=plt.gca(), cart_results=xy - target_xy)
    plt.axhline(0, c="grey")
    plt.axvline(0, c="grey")
    plt.xlabel("X distance to target")
    plt.ylabel("Y distance to target")
    plt.show()

def plot_training_log(log):
    fig, axs = plt.subplots(1, 1)
    fig.set_tight_layout(True)
    fig.set_size_inches((8, 3))

    axs.semilogy(log)

    axs.set_ylabel("Loss")
    axs.set_xlabel("Batch #")
    plt.show()

def pretrain(ac_network, env, n_batches=6000, batch_size=32, lr=1e-3, interval=250):
    """
    Pretrains the ACNetwork's actor using L1 loss on target reaching tasks.
    
    Args:
        ac_network: Your ACNetwork instance
        env: CustomReachEnv instance
        n_batches: Number of training batches
        batch_size: Samples per batch
        lr: Learning rate
        interval: Loss logging interval
    """
    device = env.device
    optimizer = th.optim.Adam(ac_network.actor.parameters(), lr=lr)
    losses = []

    def l1(x, y):
        """L1 loss between trajectories and targets"""
        return th.mean(th.sum(th.abs(x - y), dim=-1))

    for batch in range(n_batches):
        # Initialize hidden states and reset environment
        h = ac_network.actor.init_hidden(batch_size)
        obs, info = env.reset(options={"batch_size": batch_size})
        terminated = False

        # Store trajectories and targets
        xy = [info["states"]["fingertip"][:, None, :]]
        tg = [info["goal"][:, None, :]]

        # Run episode
        while not terminated:
            with th.no_grad():  # Disable gradient tracking during rollout
                action, h = ac_network.actor(obs.to(device), h)
                obs, _, terminated, _, info = env.step(action=action)

            xy.append(info["states"]["fingertip"][:, None, :])
            tg.append(info["goal"][:, None, :])

        # Calculate loss
        xy_tensor = th.cat(xy, dim=1).to(device)
        tg_tensor = th.cat(tg, dim=1).to(device)
        loss = l1(xy_tensor, tg_tensor)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(ac_network.actor.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

        # Logging
        if (batch % interval == 0) and (batch != 0):
            avg_loss = sum(losses[-interval:])/interval
            print(f"Batch {batch}/{n_batches} | Avg Loss: {avg_loss:.4f}")

    return losses

# Usage example
if __name__ == "__main__":
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    # Create environment
    effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())
    env = CustomReachEnv(effector=effector)
    
    # Initialize ACNetwork
    ac_network = ACNetwork(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 3e-4  # Dummy schedule
    ).to(device)
    
    # Run pretraining
    losses = pretrain(
        ac_network=ac_network,
        env=env,
        n_batches=6000,
        batch_size=32,
        lr=1e-3
    )
    
    # Plot results (using your existing plotting functions)
    plot_training_log(losses)
    #plot_simulations(xy=th.detach(xy_tensor), target_xy=th.detach(tg_tensor))