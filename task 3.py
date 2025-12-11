from sklearn.cluster import KMeans
import numpy as np


customers = np.array([
    [5, 200], [6, 220], [4, 180], [15, 600], [16, 640],
    [14, 580], [8, 300], [9, 320], [7, 250], [20, 900],
    [18, 850], [2, 90], [3, 120], [1, 60], [19, 880]
])


kmeans = KMeans(n_clusters=3)


kmeans.fit(customers)
centroids = kmeans.cluster_centers_

labels = kmeans.labels_

print("Centroids:")
print(centroids)

print("\nLabels:")
print(labels)
