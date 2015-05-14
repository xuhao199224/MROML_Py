__author__ = 'hao'


from numpy import *
import kMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


def DBScan(dist=0.3, min_neighbor=10, dataSet_scaled=None):
    """Perform DBSCAN clustering from features or distance matrix.
    Parameters
    ----------
    eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    dataSet_scaled : dataSet that has been scaled by StandardScaler().fit_transform(dataSet)
    """
    if dataSet_scaled is None:
        return None
    db = DBSCAN(eps=dist, min_samples=min_neighbor).fit(dataSet_scaled)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = dataSet_scaled[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], xy[:, 2], '.', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        xy = dataSet_scaled[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], xy[:, 2], '.', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()