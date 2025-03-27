import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

def compute_pca(activity_data, n_components=20):
    """Computes PCA on neural activity."""
    pca = PCA(n_components=n_components)
    return pca.fit_transform(activity_data)

def compute_cca(activity1, activity2):
    """Computes Canonical Correlation Analysis (CCA) between two datasets."""
    cca = CCA(n_components=min(activity1.shape[1], activity2.shape[1]))
    return cca.fit(activity1, activity2).score(activity1, activity2)
