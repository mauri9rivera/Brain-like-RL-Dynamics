import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

def extract_leading_components(neural_activities, n_components=20):
    """Extract top PCA components from neural activity."""
    # Concatenate all trials
    data = np.vstack([trial for trial in neural_activities])
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca.components_, pca.explained_variance_ratio_

def compute_cca(activity1, activity2):
    """Computes Canonical Correlation Analysis (CCA) between two datasets."""
    cca = CCA(n_components=min(activity1.shape[1], activity2.shape[1]))
    return cca.fit(activity1, activity2).score(activity1, activity2)
