# Import necessary libraries
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Step 1: Generate synthetic data for demonstration
X, y = make_blobs(n_samples=200, centers=3, random_state=42)

# Step 2: Create a KMeans clustering object and specify the number of clusters
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)  # Set n_init explicitly to avoid random initialization issues

# Step 3: Perform clustering on the data
kmeans.fit(X)

# Step 4: Get the cluster labels and cluster centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Step 5: Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)  # Scatter plot of data points, color-coded by cluster label
plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='red')  # Plot centroids as red 'x'
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
