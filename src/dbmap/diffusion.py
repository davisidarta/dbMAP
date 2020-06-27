#####################################
# Author: Davi Sidarta-Oliveira
# School of Medical Sciences,University of Campinas,Brazil
# contact: davisidarta@gmail.com
######################################
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, find, issparse
from scipy.sparse.linalg import eigs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from . import NMSlibTransformer 
from . import multiscale

print(__doc__)

class Diffusor(TransformerMixin, BaseEstimator):
    """Sklearn estimator for using fast anisotropic diffusion with an anisotropic
    adaptive algorithm as proposed by Setty et al, 2018, and optimized by Sidarta-Oliveira, 2020.
    :param n_components: Number of diffusion components to compute. Defaults to 50. We suggest larger values if
           analyzing more than 10,000 cells.
    :param n_neighbors: Number of k-nearest-neighbors to compute. The adaptive kernel will normalize distances by each cell
           distance of its median neighbor.
    :param knn_dist: Distance metric for building kNN graph. Defaults to 'euclidean'. Users are encouraged to explore
           different metrics, such as 'cosine' and 'jaccard'. The 'hamming' and 'jaccard' distances are also available for string
           vectors.
    :param ann: Boolean. Whether to use approximate nearest neighbors for graph construction. Defaults to True.
    :param alpha: Alpha in the diffusion maps literature. Controls how much the results are biased by data distribution.
           Defaults to 1, which is suitable for normalized data.
    :param n_jobs: Number of threads to use in calculations. Defaults to all but one.
    :return: Diffusion components ['EigenVectors'], associated eigenvalues ['EigenValues'] and suggested number of resulting components to use
             during Multiscaling.
    Examples
    -------------
    >>>import dbmap
    # Fazer o resto do exemplo
    """

    def __init__(self,
                 n_components=100,
                 n_neighbors=30,
                 alpha=1,
                 n_jobs=-2,
                 ann=True,
                 ann_dist='angular_sparse',
                 M=30,
                 efC=100,
                 efS=100,
                 knn_dist='euclidean',
                 sensitivity = 2
                 ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.ann = ann
        self.ann_dist = ann_dist
        self.M = M
        self.efC = efC
        self.efS = efS
        self.knn_dist = knn_dist
        self.sensitivity = sensitivity


    def fit_transform(self, data,
                      plot_knee=False):
        """Effectively computes on data.
        :param data: input data. Takes in numpy arrays and scipy csr sparse matrices.
        Please use with sparse data for top performance. You can adjust a series of
        parameters that can make the process faster and more informational depending
        on your dataset. Read more at https://github.com/davisidarta/dbmap
        """
        self.plot_knee = plot_knee

        start = time.time()
        N = data.shape[0]
        if self.ann == True:
            # Construct an approximate k-nearest-neighbors graph
            anbrs = NMSlibTransformer(n_neighbors=self.n_neighbors,
                                      metric=self.ann_dist,
                                      method='hnsw',
                                      n_jobs=self.n_jobs,
                                      M=self.M,
                                      efC=self.efC,
                                      efS=self.efS)
            anbrs = anbrs.fit(data)
            akNN = anbrs.transform(data)
            # Adaptive k
            adaptive_k = int(np.floor(self.n_neighbors / 2))
            adaptive_std = np.zeros(N)
            for i in np.arange(len(adaptive_std)):
                adaptive_std[i] = np.sort(akNN.data[akNN.indptr[i]: akNN.indptr[i + 1]])[
                    adaptive_k - 1
                    ]
            # Distance metrics
            x, y, dists = find(akNN)  # k-nearest-neighbor distances
            # X, y specific stds
            dists = dists / adaptive_std[x]  # Normalize by the distance of median nearest neighbor
            W = csr_matrix((np.exp(-dists), (x, y)), shape=[N, N])  # Normalized distances
        else:

            # Construct a k-nearest-neighbors graph
            nbrs = NearestNeighbors(n_neighbors=int(self.n_neighbors), metric=self.knn_dist, n_jobs=self.n_jobs).fit(
                data)
            knn = nbrs.kneighbors_graph(data, mode='distance')
            # Adaptive k: distance to cell median nearest neighbors, used for kernel normalization.
            adaptive_k = int(np.floor(self.n_neighbors / 2))
            nbrs = NearestNeighbors(n_neighbors=int(adaptive_k), metric='euclidean', n_jobs=self.n_jobs).fit(data)
            adaptive_std = nbrs.kneighbors_graph(data, mode='distance').max(axis=1)
            adaptive_std = np.ravel(adaptive_std.todense())
            # Distance metrics
            x, y, dists = find(knn)  # k-nearest-neighbor distances
            # X, y specific stds
            dists = dists / adaptive_std[x]  # Normalize by the distance of median nearest neighbor
            W = csr_matrix((np.exp(-dists), (x, y)), shape=[N, N])  # Normalized distances
        # Kernel construction
        kernel = W + W.T
        # Diffusion through Markov chain
        D = np.ravel(kernel.sum(axis=1))
        if self.alpha > 0:
            # L_alpha
            D[D != 0] = D[D != 0] ** (-self.alpha)
            mat = csr_matrix((D, (range(N), range(N))), shape=[N, N])
            kernel = mat.dot(kernel).dot(mat)
            D = np.ravel(kernel.sum(axis=1))
        D[D != 0] = 1 / D[D != 0]

        # Setting the diffusion operator
        T = csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(kernel)

        # Eigen value decomposition
        D, V = eigs(T, self.n_components, tol=1e-4, maxiter=1000)
        D = np.real(D)
        V = np.real(V)
        inds = np.argsort(D)[::-1]
        D = D[inds]
        V = V[:, inds]

        # Normalize by the first diffusion component
        for i in range(V.shape[1]):
            V[:, i] = V[:, i] / np.linalg.norm(V[:, i])

        # Create the results dictionary
        res = {'T': T, 'EigenVectors': V, 'EigenValues': D}
        res['EigenVectors'] = pd.DataFrame(res['EigenVectors'])
        if not issparse(data):
            res['EigenValues'] = pd.Series(res['EigenValues'])
        res["EigenValues"] = pd.Series(res["EigenValues"])
        res['kernel'] = kernel

        multi = multiscale(n_eigs=None, plot=self.plot_knee, sensitivity=self.sensitivity)
        mms = multi.fit(res)
        mms = mms.transform(res)
        res['StructureComponents'] = mms

        end = time.time()
        print('Total computation time=%f (sec), per sample=%f (sec), per sample adjusted for thread number=%f (sec)' %
              (end - start, float(end - start) / N, self.n_jobs * float(end - start) / N))

        return res
