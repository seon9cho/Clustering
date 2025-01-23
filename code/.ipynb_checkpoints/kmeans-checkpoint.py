import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import copy

def normalize(X, min_val=None, max_val=None):
    if min_val is None: min_val = np.nanmin(X, axis=0)
    if max_val is None: max_val = np.nanmax(X, axis=0)
    return (X - min_val) / (max_val - min_val)

class KMeans(BaseEstimator, ClusterMixin):
    """Basic k-means clustering class."""
    def __init__(self, n_clusters, max_iter=100, tol=1e-5):
        """Store clustering algorithm parameters.
        
        Parameters:
            n_clusters (int): How many clusters to compute.
            max_iter (int): The maximum number of iterations to compute.
            tol (float): The convergence tolerance.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.clusters = None
        self.centers = None
        self.instances_per_cluster = []
        self.SSE = []
    
    def fit(self, X, y=None, init_centers="random"):
        """Compute the cluster centers from random initial conditions.
        
        Parameters:
            X ((n_samples, n_classes) ndarray): the data to be clustered.
        """
        if init_centers == "random":
            old_centers = X[np.random.choice(len(X), size=self.n_clusters)]
        elif init_centers == "initial":
            old_centers = X[:self.n_clusters]
        else:
            raise ValueError("init_centers must be either 'random' or 'initial'")
        self.clusters = np.argmin(np.dstack([np.linalg.norm(X - old_centers[i], axis=1) 
                                             for i in range(self.n_clusters)]), axis=2).flatten()
        for _ in range(self.max_iter):
            new_centers = np.zeros_like(old_centers)
            for i in range(self.n_clusters):
                if len(X[self.clusters==i]) == 0:
                    new_centers[i] = X[np.random.choice(len(X))]
                else:
                    new_centers[i] = X[self.clusters==i].mean(axis=0)
            self.clusters = np.argmin(np.dstack([np.linalg.norm(X - new_centers[i], axis=1) 
                                                 for i in range(self.n_clusters)]), axis=2).flatten()
            if np.linalg.norm(old_centers - new_centers) < self.tol:
                break
        self.centers = new_centers
        for i in range(self.n_clusters):
            self.instances_per_cluster.append(sum(self.clusters==i))
            sse = np.linalg.norm(X[self.clusters==i] - km.centers[i])**2
            self.SSE.append(sse)
        
        return self
    
    def save_clusters(self, filename):
        f = open(filename, "w+") 
        f.write("{:d}\n".format(self.n_clusters))
        f.write("{:.4f}\n\n".format(sum(self.SSE)))
        for i in range(self.n_clusters):
            f.write(np.array2string(self.centers[i], precision=4, separator=","))
            f.write("\n")
            f.write("{:d}\n".format(self.instances_per_cluster[i]))
            f.write("{:.4f}\n\n".format(self.SSE[i]))
        f.close()