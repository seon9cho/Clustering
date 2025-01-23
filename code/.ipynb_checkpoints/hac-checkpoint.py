import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import copy

class HAC(BaseEstimator,ClusterMixin):

    def __init__(self, n_clusters=[3], link_type='single'): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            link_type = single or complete. when combining two clusters use complete link or single link
        """
        self.link_type = link_type
        self.n_clusters = n_clusters
        self.clusters = None
        self.cluster_dict = {k: None for k in self.n_clusters}
        self.centers = {k: [] for k in self.n_clusters}
        self.instances_per_cluster = {k: [] for k in self.n_clusters}
        self.SSE = {k: [] for k in self.n_clusters}
        
    def fit(self, X, y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        n, d = X.shape
        dist_matrix = np.linalg.norm(X - np.hstack([X for _ in range(n)]).reshape(n, n, d), axis=2)
        dist_matrix += np.diag(np.ones(n)*np.nan)
        self.clusters = np.arange(n).reshape(-1, 1).tolist()
        if self.link_type == 'single':
            self._single_link(X, dist_matrix)
        if self.link_type == 'complete':
            self._complete_link(X, dist_matrix)
        
        for k in self.n_clusters:
            for cluster in self.cluster_dict[k]:
                center = np.mean(X[cluster], axis=0)
                self.centers[k].append(center)
                self.instances_per_cluster[k].append(len(cluster))
                sse = np.linalg.norm(X[cluster] - center)**2
                self.SSE[k].append(sse)
            
        return self
    
    def _single_link(self, X, dist_matrix):
        n, d = X.shape
        k_min = min(self.n_clusters)
        m = len(self.clusters)
        while m > k_min:
            ind = np.nanargmin(dist_matrix)
            i = ind // n
            j = ind % n
            found_i = False
            found_j = False
            loc = []
            for k, cluster in enumerate(self.clusters):
                if i in cluster:
                    loc.append(k)
                    found_i = True
                elif j in cluster:
                    loc.append(k)
                    found_j = True
                if found_i and found_j:
                    break
            for inst1 in self.clusters[loc[0]]:
                for inst2 in self.clusters[loc[1]]:
                    dist_matrix[inst1, inst2] = np.nan
                    dist_matrix[inst2, inst1] = np.nan
            self.clusters[loc[0]] += self.clusters[loc[1]]
            del self.clusters[loc[1]]
            m -= 1
            if m in self.n_clusters:
                self.cluster_dict[m] = copy.deepcopy(self.clusters)
    
    def _complete_link(self, X, dist_matrix):
        n, d = X.shape
        k_min = min(self.n_clusters)
        m = len(self.clusters)
        while m > k_min:
            ind = np.nanargmin(dist_matrix)
            i = ind // n
            j = ind % n
            found_i = False
            found_j = False
            loc = []
            for k, cluster in enumerate(self.clusters):
                if i in cluster:
                    loc.append(k)
                    found_i = True
                elif j in cluster:
                    loc.append(k)
                    found_j = True
                if found_i and found_j:
                    break
            if len(self.clusters[loc[0]]) > 1 or len(self.clusters[loc[1]]) > 1:
                break
            for inst1 in self.clusters[loc[0]]:
                for inst2 in self.clusters[loc[1]]:
                    dist_matrix[inst1, inst2] = np.nan
                    dist_matrix[inst2, inst1] = np.nan
            self.clusters[loc[0]] += self.clusters[loc[1]]
            del self.clusters[loc[1]]
            m -= 1
            if m in self.n_clusters:
                self.cluster_dict[m] = copy.deepcopy(self.clusters)
        while m > k_min:
            curr_dists = np.zeros((m, m))
            for i, cluster1 in enumerate(self.clusters):
                for j, cluster2 in enumerate(self.clusters):
                    curr_dists[i,j] = np.max(dist_matrix[cluster1].T[cluster2])
            ind = np.nanargmin(curr_dists)
            i = ind // m
            j = ind % m
            self.clusters[i] += self.clusters[j]
            del self.clusters[j]
            m -= 1
            if m in self.n_clusters:
                self.cluster_dict[m] = copy.deepcopy(self.clusters)
            
            
    
    def save_clusters(self, filename, k):
        f = open(filename, "w+") 
        f.write("{:d}\n".format(k))
        f.write("{:.4f}\n\n".format(sum(self.SSE[k])))
        for i in range(k):
            f.write(np.array2string(self.centers[k][i], precision=4, separator=","))
            f.write("\n")
            f.write("{:d}\n".format(self.instances_per_cluster[k][i]))
            f.write("{:.4f}\n\n".format(self.SSE[k][i]))
        f.close()
