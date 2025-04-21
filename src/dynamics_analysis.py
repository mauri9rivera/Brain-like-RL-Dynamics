from dsa_repo.DSA.dmd import DMD
from dsa_repo.DSA.simdist import SimilarityTransformDist
import numpy as np
import matplotlib.pyplot as plt

def dynamic_analysis(neural_activity1, neural_activity2):
    # 1. Set up DMD embedding with 10 time delays
    delays = 10
    dmd = DMD(k=delays)

    # 2. Embed each full 20-dimensional trajectory
    emb1 = dmd.embed(neural_activity1)
    emb2 = dmd.embed(neural_activity2)

    # 3. Instantiate the Procrustes-style distance (SimilarityTransformDist)
    sim = SimilarityTransformDist(
        iters=2000,
        score_method='angular',   # or 'euclidean' / 'wasserstein'
        lr=1e-3,
        device='cpu',
        group='O(n)'             # can be 'SO(n)' or 'GL(n)'
    )

    # 4. Compute self- and cross-model dissimilarities
    diss11 = sim.fit_score(emb1, emb1)
    diss22 = sim.fit_score(emb2, emb2)
    diss12 = sim.fit_score(emb1, emb2)

    # 5. Component-wise dissimilarities across the 20 PCA ranks
    n_components = neural_activity1.shape[1]
    component_scores = np.zeros(n_components)
    for i in range(n_components):
        comp1 = neural_activity1[:, i:i+1]
        comp2 = neural_activity2[:, i:i+1]
        e1 = dmd.embed(comp1)
        e2 = dmd.embed(comp2)
        component_scores[i] = sim.fit_score(e1, e2)

    # 6. Heatmap of all component-vs-component dissimilarities
    heatmap = np.zeros((n_components, n_components))
    for i in range(n_components):
        for j in range(n_components):
            e1 = dmd.embed(neural_activity1[:, i:i+1])
            e2 = dmd.embed(neural_activity2[:, j:j+1])
            heatmap[i, j] = sim.fit_score(e1, e2)

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, aspect='auto', origin='lower')
    plt.colorbar(label='Procrustes-style dissimilarity')
    plt.xlabel('Model 2 PCA component')
    plt.ylabel('Model 1 PCA component')
    plt.title('Heatmap of Procrustes-style Dissimilarity')
    plt.tight_layout()
    plt.show()

    # 7. Barplot of self vs cross-model dissimilarities
    labels = ['Model1 vs Model1', 'Model2 vs Model2', 'Model1 vs Model2']
    values = [diss11, diss22, diss12]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylabel('Procrustes-style dissimilarity')
    plt.title('Self vs Cross-Model Dissimilarity')
    plt.tight_layout()
    plt.show()

