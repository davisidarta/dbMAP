#####################################
# Author: Davi Sidarta-Oliveira
# School of Medical Sciences,University of Campinas,Brazil
# contact: davisidarta@gmail.com
# Please note that this code has several contributions from Manu Setty et al, Nature
######################################
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, find, issparse
from scipy.sparse.linalg import eigs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from . import ann
from . import multiscale
from . import utils


print(__doc__)


class Diffusor(TransformerMixin):
    """
    Sklearn estimator for using fast anisotropic diffusion with an anisotropic
    adaptive algorithm as proposed by Setty et al, 2018, and optimized by Sidarta-Oliveira, 2020.

    Parameters
    ----------
    n_components : Number of diffusion components to compute. Defaults to 100. We suggest larger values if
                   analyzing more than 10,000 cells.

    n_neighbors : Number of k-nearest-neighbors to compute. The adaptive kernel will normalize distances by each cell
                  distance of its median neighbor.

    knn_dist : Distance metric for building kNN graph. Defaults to 'euclidean'. Users are encouraged to explore
               different metrics, such as 'cosine' and 'jaccard'. The 'hamming' and 'jaccard' distances are also available
               for string vectors.

    ann : Boolean. Whether to use approximate nearest neighbors for graph construction. Defaults to True.

    alpha : Alpha in the diffusion maps literature. Controls how much the results are biased by data distribution.
            Defaults to 1, which is suitable for normalized data.

    n_jobs : Number of threads to use in calculations. Defaults to all but one.

    sensitivity : Sensitivity to select eigenvectors if diff_normalization is set to 'knee'. Useful when dealing wit

    :returns: Diffusion components ['EigenVectors'], associated eigenvalues ['EigenValues'] and suggested number of
             resulting components to use during Multiscaling.

    Example
    -------------

    import numpy as np
    from sklearn.datasets import load_digits
    from scipy.sparse import csr_matrix
    import dbmap

    # Load the MNIST digits data, convert to sparse for speed
    digits = load_digits()
    data = csr_matrix(digits)

    # Fit the anisotropic diffusion process
    diff = dbmap.diffusion.Diffusor()
    res = diff.fit_transform(data)

    """

    def __init__(self,
                 n_components=100,
                 n_neighbors=30,
                 alpha=0.5,
                 n_jobs=10,
                 ann=True,
                 ann_dist='angular_sparse',
                 M=30,
                 efC=100,
                 efS=100,
                 knn_dist='euclidean',
                 kernel_use='sidarta',
                 sensitivity=1
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
        self.kernel_use = kernel_use
        self.sensitivity = sensitivity

    def fit(self, data, plot_knee=False):
        """Effectively computes on data.
        :param plot_knee: Whether to plot the scree plot of diffusion eigenvalues.
        :param data: input data. Takes in numpy arrays and scipy csr sparse matrices.
        Please use with sparse data for top performance. You can adjust a series of
        parameters that can make the process faster and more informational depending
        on your dataset. Read more at https://github.com/davisidarta/dbmap
        """
        self.plot_knee = plot_knee
        self.start_time = time.time()



        self.N = data.shape[0]
        if self.ann:
            # Construct an approximate k-nearest-neighbors graph
            anbrs = ann.NMSlibTransformer(n_neighbors=self.n_neighbors,
                                      metric=self.ann_dist,
                                      method='hnsw',
                                      n_jobs=self.n_jobs,
                                      M=self.M,
                                      efC=self.efC,
                                      efS=self.efS)
            anbrs = anbrs.fit(data)
            knn = anbrs.transform(data)
            # X, y specific stds: Normalize by the distance of median nearest neighbor to account for neighborhood size.
            adaptive_k = int(np.floor(self.n_neighbors / 2))
            adaptive_std = np.zeros(self.N)
            for i in np.arange(len(adaptive_std)):
                adaptive_std[i] = np.sort(knn.data[knn.indptr[i]: knn.indptr[i + 1]])[
                    adaptive_k - 1
                    ]
        else:
            # Construct a k-nearest-neighbors graph
            nbrs = NearestNeighbors(n_neighbors=int(self.n_neighbors), metric=self.knn_dist, n_jobs=self.n_jobs).fit(
                data)
            knn = nbrs.kneighbors_graph(data, mode='distance')
            # Normalize distances by the distance of median nearest neighbor to account for neighborhood size.
            adaptive_k = int(np.floor(self.n_neighbors / 2))
            nbrs = NearestNeighbors(n_neighbors=int(adaptive_k), metric='euclidean', n_jobs=self.n_jobs).fit(data)
            adaptive_std = nbrs.kneighbors_graph(data, mode='distance').max(axis=1)
            adaptive_std = np.ravel(adaptive_std.todense())

        # Distance metrics
        x, y, dists = find(knn)  # k-nearest-neighbor distances

        if self.kernel_use == 'setty':
           # X, y specific stds
           dists = dists / adaptive_std[x]  # Normalize by the distance of median nearest neighbor

        if self.kernel_use == 'sidarta':
            # X, y specific stds
            dists = dists - (dists / adaptive_std[x])  # Normalize by normalized contribution to neighborhood size.

        W = csr_matrix((np.exp(-dists), (x, y)), shape=[self.N, self.N])  # Normalized distances

        # Kernel construction
        kernel = W + W.T
        self.kernel = kernel

        return self

    def transform(self, data, n_eigs=None):
        self.n_eigs = n_eigs

        # Diffusion through Markov chain
        D = np.ravel(self.kernel.sum(axis=1))
        if self.alpha > 0:
            # L_alpha
            D[D != 0] = D[D != 0] ** (-self.alpha)
            mat = csr_matrix((D, (range(self.N), range(self.N))), shape=[self.N, self.N])
            kernel = mat.dot(self.kernel).dot(mat)
            D = np.ravel(kernel.sum(axis=1))

        D[D != 0] = 1 / D[D != 0]

        # Setting the diffusion operator
        T = csr_matrix((D, (range(self.N), range(self.N))), shape=[self.N, self.N]).dot(self.kernel)

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
        self.res = {'T': T, 'EigenVectors': V, 'EigenValues': D, 'kernel': self.kernel}
        self.res['EigenVectors'] = pd.DataFrame(self.res['EigenVectors'])
        if not issparse(data):
            self.res['EigenValues'] = pd.Series(self.res['EigenValues'])
        self.res["EigenValues"] = pd.Series(self.res["EigenValues"])

        multi = multiscale.multiscale(n_eigs=self.n_eigs, plot=self.plot_knee, sensitivity=self.sensitivity)
        mms = multi.fit(self.res)
        mms = mms.transform(self.res)
        self.res['StructureComponents'] = mms

        end = time.time()
        print('Diffusion time = %f (sec), per sample=%f (sec), per sample adjusted for thread number=%f (sec)' %
              (end - self.start_time, float(end - self.start_time) / self.N, self.n_jobs * float(end - self.start_time) / self.N))

        return self.res['StructureComponents']

    def ind_dist_grad(self, data, n_eigs=None, dense=False):
        """Effectively computes on data. Also returns the normalized diffusion distances,
        indexes and gradient obtained by approximating the Laplace-Beltrami operator.
        :param plot_knee: Whether to plot the scree plot of diffusion eigenvalues.
        :param data: input data. Takes in numpy arrays and scipy csr sparse matrices.
        Please use with sparse data for top performance. You can adjust a series of
        parameters that can make the process faster and more informational depending
        on your dataset. Read more at https://github.com/davisidarta/dbmap
        """
        self.n_eigs = n_eigs
        self.start_time = time.time()
        self.N = data.shape[0]
        if self.ann:
            # Construct an approximate k-nearest-neighbors graph
            anbrs = ann.NMSlibTransformer(n_neighbors=self.n_neighbors,
                                      metric=self.ann_dist,
                                      method='hnsw',
                                      n_jobs=self.n_jobs,
                                      M=self.M,
                                      efC=self.efC,
                                      efS=self.efS)
            anbrs = anbrs.fit(data)
            self.ind, self.dists, self.grad, kneighbors_graph = anbrs.ind_dist_grad(data)
            x, y, self.dists = find(self.dists)

            # X, y specific stds: Normalize by the distance of median nearest neighbor to account for neighborhood size.
            adaptive_k = int(np.floor(self.n_neighbors / 2))
            adaptive_std = np.zeros(self.N)
            for i in np.arange(len(adaptive_std)):
                adaptive_std[i] = np.sort(kneighbors_graph.data[kneighbors_graph.indptr[i]: kneighbors_graph.indptr[i + 1]])[
                    adaptive_k - 1
                    ]
        else:
            # Construct a k-nearest-neighbors graph
            nbrs = NearestNeighbors(n_neighbors=int(self.n_neighbors), metric=self.knn_dist, n_jobs=self.n_jobs).fit(
                data)
            knn = nbrs.kneighbors_graph(data, mode='distance')
            # Normalize distances by the distance of median nearest neighbor to account for neighborhood size.
            adaptive_k = int(np.floor(self.n_neighbors / 2))
            nbrs = NearestNeighbors(n_neighbors=int(adaptive_k), metric='euclidean', n_jobs=self.n_jobs).fit(data)
            adaptive_std = nbrs.kneighbors_graph(data, mode='distance').max(axis=1)
            adaptive_std = np.ravel(adaptive_std.todense())
            # Distance metrics
            x, y, self.dists = find(knn)  # k-nearest-neighbor distances

        if self.kernel_use == 'setty':
            # X, y specific stds
            self.dists = self.dists / adaptive_std[x]  # Normalize by the distance of median nearest neighbor

        if self.kernel_use == 'sidarta':
            # X, y specific stds
            self.dists = self.dists - (self.dists / adaptive_std[x])  # Normalize by normalized contribution to neighborhood size.

        W = csr_matrix((np.exp(-self.dists), (x, y)), shape=[self.N, self.N])  # Normalized distances

        # Kernel construction
        self.kernel = W + W.T

        # Diffusion through Markov chain
        D = np.ravel(self.kernel.sum(axis=1))
        if self.alpha > 0:
            # L_alpha
            D[D != 0] = D[D != 0] ** (-self.alpha)
            mat = csr_matrix((D, (range(self.N), range(self.N))), shape=[self.N, self.N])
            kernel = mat.dot(self.kernel).dot(mat)
            D = np.ravel(kernel.sum(axis=1))

        D[D != 0] = 1 / D[D != 0]

        # Setting the diffusion operator
        T = csr_matrix((D, (range(self.N), range(self.N))), shape=[self.N, self.N]).dot(self.kernel)

        # Eigen value decomposition
        if dense:
            from scipy.linalg import eig

            D, V = eig(T.toarray())
        else:
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
        self.res = {'T': T, 'EigenVectors': V, 'EigenValues': D, 'kernel': self.kernel}
        self.res['EigenVectors'] = pd.DataFrame(self.res['EigenVectors'])
        if not issparse(data):
            self.res['EigenValues'] = pd.Series(self.res['EigenValues'])
        self.res["EigenValues"] = pd.Series(self.res["EigenValues"])

        multi = multiscale.multiscale(n_eigs=self.n_eigs, plot=self.plot_knee, sensitivity=self.sensitivity)
        mms = multi.fit(self.res)
        mms = mms.transform(self.res)

        anbrs = ann.NMSlibTransformer(n_neighbors=self.n_neighbors,
                                  metric='cosine',
                                  method='hnsw',
                                  n_jobs=self.n_jobs,
                                  M=self.M,
                                  efC=self.efC,
                                  efS=self.efS, dense=True).fit(mms)
        self.ind, self.dists, self.grad, self.graph = anbrs.ind_dist_grad(mms)

        end = time.time()
        print('Diffusion time = %f (sec), per sample=%f (sec), per sample adjusted for thread number=%f (sec)' %
              (end - self.start_time, float(end - self.start_time) / self.N, self.n_jobs * float(end - self.start_time) / self.N))

        return self.ind, self.dists, self.grad, self.graph
