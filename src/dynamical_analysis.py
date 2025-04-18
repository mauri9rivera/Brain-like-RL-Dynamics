import os
import sys
import numpy as np
import matplotlib.pyplot as plt


from dsa_repo.DSA.dsa import DND, Procrustes

# 2. Load your precomputed PCA neural activity
#    Assume `neural_activity1` and `neural_activity2` are numpy arrays of shape (T, 20)
#    containing the 20 leading PCA components for each model.
#    Replace these with your actual data-loading routines.
#
# Example placeholder data (remove when using real data):
# neural_activity1 = np.load('model1_pca.npy')  # shape (T, 20)
# neural_activity2 = np.load('model2_pca.npy')  # shape (T, 20)

# 3. Compute time-delayed embeddings (DND) with 10 delays for each model
delays = 10
dnd = DND(k=delays)

# Embed each full 20-dimensional trajectory
emb1 = dnd.embed(neural_activity1)
emb2 = dnd.embed(neural_activity2)

# 4. Compute Procrustes dissimilarity
procrustes = Procrustes()

# Self-dissimilarities
diss11 = procrustes.distance(emb1, emb1)
diss22 = procrustes.distance(emb2, emb2)
# Cross-dissimilarity
diss12 = procrustes.distance(emb1, emb2)

# 5. Compute component-wise dissimilarities across the 20 PCA ranks
component_scores = np.zeros(20)
for i in range(20):
    # embed single-component time series
    comp1 = neural_activity1[:, i:i+1]
    comp2 = neural_activity2[:, i:i+1]
    e1 = dnd.embed(comp1)
    e2 = dnd.embed(comp2)
    component_scores[i] = procrustes.distance(e1, e2)

# 6. Create heatmap of full component-vs-component dissimilarities
heatmap = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        e1 = dnd.embed(neural_activity1[:, i:i+1])
        e2 = dnd.embed(neural_activity2[:, j:j+1])
        heatmap[i, j] = procrustes.distance(e1, e2)

plt.figure(figsize=(8, 6))
plt.imshow(heatmap, aspect='auto', origin='lower')
plt.colorbar(label='Procrustes dissimilarity')
plt.xlabel('Model 2 PCA component')
plt.ylabel('Model 1 PCA component')
plt.title('Heatmap of Procrustes Dissimilarity')
plt.tight_layout()
plt.show()

# 7. Barplot of self vs cross-model dissimilarities
labels = ['Model1 vs Model1', 'Model2 vs Model2', 'Model1 vs Model2']
values = [diss11, diss22, diss12]

plt.figure(figsize=(6, 4))
plt.bar(labels, values)
plt.ylabel('Procrustes dissimilarity')
plt.title('Self vs Cross-Model Dissimilarity')
plt.tight_layout()
plt.show()
