import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, find, issparse
from scipy.sparse.linalg import eigs
import hnswlib

import time
import sys

try:
    import nmslib
except ImportError:
    print("The package 'nmslib' is required to run accelerated dbMAP")
    sys.exit()

import numpy as np
from scipy.sparse import csr_matrix

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsTransformer
from sklearn.utils._testing import assert_array_almost_equal

print(__doc__)


class NMSlibTransformer(TransformerMixin, BaseEstimator):
    """Wrapper for using nmslib as sklearn's KNeighborsTransformer"""

    def __init__(self, n_neighbors=5, metric='euclidean', method='sw-graph',
                 n_jobs=1):
        self.n_neighbors = n_neighbors
        self.method = method
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, data):
        self.n_samples_fit_ = data.shape[0]

        # see more metric in the manual
        # https://github.com/nmslib/nmslib/tree/master/manual
        space = {
            'sqeuclidean': 'l2',
            'euclidean': 'l2',
            'cosine': 'cosinesimil',
            'l1': 'l1',
            'l2': 'l2',
            'hamming' : 'bit_hamming',
            'jaccard' : 'bit_jaccard',
            'jaccard_sparse' : 'jaccard_sparse',
        }[self.metric]

        self.nmslib_ = nmslib.init(method=self.method, space=space)
        self.nmslib_.addDataPointBatch(data)
        self.nmslib_.createIndex()
        return self

    def transform(self, data):
        n_samples_transform = data.shape[0]

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        n_neighbors = self.n_neighbors + 1

        results = self.nmslib_.knnQueryBatch(data, k=n_neighbors,
                                             num_threads=self.n_jobs)
        indices, distances = zip(*results)
        indices, distances = np.vstack(indices), np.vstack(distances)

        if self.metric == 'sqeuclidean':
            distances **= 2

        indptr = np.arange(0, n_samples_transform * n_neighbors + 1,
                           n_neighbors)
        kneighbors_graph = csr_matrix((distances.ravel(), indices.ravel(),
                                       indptr), shape=(n_samples_transform,
                                                       self.n_samples_fit_))

        return kneighbors_graph

    def test_transformers(self, data):
        """Test that NMSlibTransformer and KNeighborsTransformer give same results
        """
        X = np.random.RandomState(42).randn(10, 2)

        knn = KNeighborsTransformer()
        Xt0 = knn.fit_transform(data)

        nms = NMSlibTransformer()
        Xt1 = nms.fit_transform(data)

        assert_array_almost_equal(Xt0.toarray(), Xt1.toarray(), decimal=5)

def diffuse(data, n_components=100, knn=30, knn_dist='euclidean', ann=True, n_jobs=-1, alpha=1, sparse=True):
    """Runs Diffusion maps using an adaptation of the adaptive anisotropic kernel proposed by Setty et al,
        Nature Biotechnology 2019.
    :param data: Data matrix to diffuse from. Either a sparse .coo or a dense pandas dataframe.
    :param n_components: Number of diffusion components to compute. Defaults to 50. We suggest larger values if
           analyzing more than 10,000 cells.
    :param knn: Number of k-nearest-neighbors to compute. The adaptive kernel will normalize distances by each cell
           distance of its median neighbor.
    :param knn_dist: Distance metric for building kNN graph. Defaults to 'euclidean'. Users are encouraged to explore
           different metrics, such as 'cosine' and 'jaccard'. The 'hamming' distance is also available for string
           vectors.
    :param ann: Boolean. Whether to use approximate nearest neighbors for graph construction. Defaults to True.
    :param alpha: Alpha in the diffusion maps literature. Controls how much the results are biased by data distribution.
           Defaults to 1, which is suitable for normalized data.
    :param n_jobs: Number of threads to use in calculations. Defaults to all but one.
    :param sparse: Whether input data is in a sparse format (i.e., csr or coo). Defaults to True.
    :return: Diffusion components, associated eigenvalues and suggested number of resulting components to use
             during Multiscaling.
    """
    print('Converting input to sparse. Determing nearest neighbor graph...')
    if sparse == True:
        data = data.todense()
        print('Sparse input - optimizing for sparse data efficiency. Determing nearest neighbor graph...')
    else:
        print('Dense input. Determing nearest neighbor graph...')
    
    N = data.shape[0]

    if ann == True:
        # Construct an approximate k-nearest-neighbors graph
        anbrs = NMSlibTransformer(n_neighbors=knn, metric=knn_dist, method='sw-graph', n_jobs=n_jobs)
        anbrs = anbrs.fit(data)
        akNN = anbrs.transform(data)
        # Adaptive k
        adaptive_k = int(np.floor(knn / 3))
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
        nbrs = NearestNeighbors(n_neighbors=int(knn), metric=knn_dist, n_jobs=n_jobs).fit(data)
        kNN = nbrs.kneighbors_graph(data, mode='distance')
        # Adaptive k: distance to cell median nearest neighbors, used for kernel normalization.
        adaptive_k = int(np.floor(knn / 2))
        nbrs = NearestNeighbors(n_neighbors=int(adaptive_k), metric='euclidean', n_jobs=n_jobs).fit(data)
        adaptive_std = nbrs.kneighbors_graph(data, mode='distance').max(axis=1)
        adaptive_std = np.ravel(adaptive_std.todense())
        # Distance metrics
        x, y, dists = find(kNN)  # k-nearest-neighbor distances
        # X, y specific stds
        dists = dists / adaptive_std[x]  # Normalize by the distance of median nearest neighbor
        W = csr_matrix((np.exp(-dists), (x, y)), shape=[N, N])  # Normalized distances
    # Kernel construction
    kernel = W + W.T
    # Diffusion through Markov chain
    D = np.ravel(kernel.sum(axis=1))
    if alpha > 0:
        # L_alpha
        D[D != 0] = D[D != 0] ** (-alpha)
        mat = csr_matrix((D, (range(N), range(N))), shape=[N, N])
        kernel = mat.dot(kernel).dot(mat)
        D = np.ravel(kernel.sum(axis=1))

    D[D != 0] = 1 / D[D != 0]

    # Setting the diffusion operator
    T = csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(kernel)

    # Eigen value decomposition
    D, V = eigs(T, n_components, tol=1e-4, maxiter=1000)
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
        res['kernel'] = kernel

    # Suggest a number of components to use
    vals = np.ravel(res['EigenValues'])
    n_eigs = np.argsort(vals[:(len(vals) - 1)] - vals[1:])[-1] + 1
    if n_eigs < 3:
        n_eigs = np.argsort(vals[:(len(vals) - 1)] - vals[1:])[-2] + 1
    res['Suggested_eigs'] = n_eigs

    return res
