__author__ = 'hao'

from numpy import *
import kMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


import numpy as np
from scipy import sparse

# # kMeans.drawPlot(dataMat)
dataMat = mat(kMeans.loadDataSet('testSet.txt'))
dataMat = StandardScaler().fit_transform(dataMat)
# db = DBSCAN(eps=0.3, min_samples=5).fit(dataMat)
# labels = db.labels_
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# print('Estimated number of clusters: %d' % n_clusters_)


# centers = [[1, 1], [-1, -1], [1, -1]]
# dataMat, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
#                             random_state=0)

dataMat = StandardScaler().fit_transform(dataMat)
db = DBSCAN(eps=0.5, min_samples=10).fit(dataMat)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(dataMat, labels))


import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = dataMat[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], xy[:, 2], '.', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = dataMat[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], xy[:, 2], '.', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()