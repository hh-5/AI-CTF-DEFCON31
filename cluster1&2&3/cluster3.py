import numpy as np
import matplotlib. pyplot as plt
from sklearn.cluster import KMeans

data = np.load("data.npz")
keys = data.files
tokens = data['tokens']
points = data ['points']
# print(tokens)
# print(points)
# from sklearn.decomposition import PCA

# # Reduce the dimensionality of the embeddings to 2D using PCA
# pca = PCA(n_components=2)
# embeddings_2d_pca = pca.fit_transform(data['points'])

# # Plot the 2D embeddings using PCA
# plt.figure(figsize=(12, 8))
# plt.scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1], alpha=0.5)
# plt.title('2D Visualization of Token Embeddings (PCA)')
# plt.xlabel('PCA dimension 1')
# plt.ylabel('PCA dimension 2')
# plt.show()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Reduce the dimensionality of the embeddings to 2D using t-SNE
embeddings_2d = TSNE(n_components=2, random_state=42).fit_transform(data['points'])

# Plot the 2D embeddings
plt.figure(figsize=(12, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
plt.title('2D Visualization of Token Embeddings')
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42).fit(embeddings_2d)

# Assign each token to its respective cluster
cluster_assignments = kmeans.labels_

# Collect tokens for each cluster
clusters_tokens = {}
for i in range(4):
    clusters_tokens[i] = tokens[cluster_assignments == i]


    # plt.figure(figsize=(15, 12))

    idx = (cluster_assignments == i)
    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], alpha=0.6, color='y', label='Cluster')

    # Annotate each point with its corresponding token
    for j, coord in enumerate(embeddings_2d[idx]):
        plt.annotate(clusters_tokens[i][j], (coord[0], coord[1]), fontsize=9, alpha=0.7)

    plt.title(f'2D Visualization of Token Embeddings for Cluster {i}')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend()
    plt.grid(True)
    plt.show()
    # plt.savefig(f"cluster_{i}.jpg")
