import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA

def get_neural_activity(env, model, n_trials=1000):
    """
    Collect the neural activities and trajectories of a network during an experimental task.

    - env: The environment doing the task
    - model: The network from which to extract the neural activity and trajectories
    - n_trials: The number of trials to perform
    """

    neural_activities = []
    trajectories = []

    for _ in range(n_trials):

        obs = env.reset()
        model.policy.hidden_states = None
        done = False
        trial_activity = []
        trial_traj = []
        
        while not done:

            # Get action and hidden state BEFORE environment step
            action, _ = model.predict(obs['observation'], deterministic=True)
            
            # Capture hidden state and trajectory
            trial_activity.append(model.policy.hidden_states.squeeze().detach().cpu().numpy())
            trial_traj.append(env.effector.end_effector_position.copy())
            
            # Step environment
            obs, _, done, _ = env.step(action)
            
            
        # Store trial data
        neural_activities.append(np.array(trial_activity))
        trajectories.append(np.array(trial_traj))
    
    return neural_activities, trajectories

def extract_leading_components(neural_activities, n_components=20):
    """Extract top PCA components from neural activity."""
    # Concatenate all trials
    data = np.vstack([trial for trial in neural_activities])

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

def compute_cca(activity1, activity2):
    """Computes Canonical Correlation Analysis (CCA) between two datasets."""
    cca = CCA(n_components=min(activity1.shape[1], activity2.shape[1]))
    return cca.fit(activity1, activity2).score(activity1, activity2)
