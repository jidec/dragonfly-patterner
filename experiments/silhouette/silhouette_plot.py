from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# Generate some data for clustering
X = np.random.rand(100, 2)

# Fit a Gaussian Mixture model with 2 components
gmm = GaussianMixture(n_components=2)
labels = gmm.fit_predict(X)

# Calculate Silhouette Coefficient for each sample
silhouette_vals = silhouette_samples(X, labels)

# Calculate average Silhouette Coefficient for the entire dataset
silhouette_avg = silhouette_score(X, labels)

# Create a Silhouette Plot
y_lower, y_upper = 0, 0
yticks = []
for i, cluster in enumerate(np.unique(labels)):
    cluster_silhouette_vals = silhouette_vals[labels == cluster]
    cluster_silhouette_vals.sort()
    y_upper += len(cluster_silhouette_vals)
    color = plt.cm.tab20(i / len(np.unique(labels)))
    plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, height=1.0,
             edgecolor='none', color=color)
    yticks.append((y_lower + y_upper) / 2.)
    y_lower += len(cluster_silhouette_vals)

plt.axvline(x=silhouette_avg, color='red', linestyle='--')
plt.yticks(yticks, np.unique(labels))
plt.ylabel('Cluster')
plt.xlabel('Silhouette Coefficient')
plt.show()