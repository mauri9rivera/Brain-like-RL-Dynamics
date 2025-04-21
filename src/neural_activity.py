from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
import numpy as np
import matplotlib.pyplot as plt

def get_neural_activity(env, policy, n_trials=1000):
    """
    Collect the neural activities and trajectories of a network during an experimental task.

    - env: The environment doing the task
    - model: The network from which to extract the neural activity and trajectories
    - n_trials: The number of trials to perform
    """

    neural_activities = []
    trajectories = []

    for _ in range(n_trials):

        # Reset hidden states
        policy.hidden_states = policy.actor.init_hidden(1)

        # Initialize environment
        obs, info = env.reset(seed= 1, options={"batch_size": 1})
        xy = [info["states"]["fingertip"][:, None, :]]
        tg = [info["goal"][:, None, :]]

        done = False
        trial_activity = []
        trial_traj = []

        while not done:

            action, _, _ = policy(obs, deterministic=True)
            
            # Capture hidden state and trajectory
            trial_activity.append(policy.hidden_states)
            
            # Step environment
            next_obs, _, done, _, info = env.step(action)
            obs_tensor = next_obs[0] 
            trial_traj.append(info["states"]["fingertip"][:, None, :])
            
        # Store trial data
        neural_activities.append(np.array([x.detach().numpy() for x in trial_activity]))
        trajectories.append(np.array([x.detach().numpy() for x in trial_traj]))
    
    return neural_activities, trajectories

def extract_leading_components(neural_activities, n_components=20):
    """Extract top PCA components from neural activity."""
    # Concatenate all trials
    data = np.vstack([trial for trial in neural_activities])[:, 0, 0, :]

    # Normalize data (z-score)
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(normalized_data)
    
    return {
        'components': pca.components_,
        'explained_variance': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'transformed_data': pca.transform(normalized_data)
    }

def plot_3d_components(pca_results, figsize=(10, 8), title='First 3 Principal Components'):
    """
    Visualize the first 3 principal components in 3D space
    
    Args:
        pca_results: Dictionary output from extract_leading_components()
        figsize: Tuple specifying figure dimensions
        title: Plot title
    """
    # Extract transformed data
    transformed_data = pca_results['transformed_data']
    
    # Verify we have enough components
    if transformed_data.shape[1] < 3:
        raise ValueError("PCA results must contain at least 3 components")
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each timepoint with temporal coloring
    x = transformed_data[:, 0]
    y = transformed_data[:, 1]
    z = transformed_data[:, 2]
    
    # Create color map based on temporal progression
    colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
    
    # Plot points with color progression
    scatter = ax.scatter(x, y, z, c=colors, s=20, alpha=0.8, depthshade=True)
    
    # Add labels and title
    ax.set_xlabel('PC1 ({:.1f}%)'.format(pca_results['explained_variance'][0]*100))
    ax.set_ylabel('PC2 ({:.1f}%)'.format(pca_results['explained_variance'][1]*100))
    ax.set_zlabel('PC3 ({:.1f}%)'.format(pca_results['explained_variance'][2]*100))
    plt.title(title)
    
    # Add colorbar for temporal progression
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=len(x)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Time Progression')
    
    plt.show()
    return fig, ax

