import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, find, issparse
from scipy.sparse.linalg import eigs
import numexpr as ne
import time
import sys
from scipy.sparse import csr_matrix

try:
    import nmslib
except ImportError:
    print("The package 'nmslib' is required to run accelerated dbMAP")
    sys.exit()

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

print(__doc__)


class NMSlibTransformer(TransformerMixin, BaseEstimator):
    """Wrapper for using nmslib as sklearn's KNeighborsTransformer"""

    def __init__(self,
                 n_neighbors=30,
                 metric='cosinesimil_sparse_fast',
                 method='hnsw',
                 n_jobs=-1,
                 M=30,
                 efC=100,
                 efS=100):
        self.n_neighbors = n_neighbors
        self.method = method
        self.metric = metric
        self.n_jobs = n_jobs
        self.M = M
        self.efC = efC
        self.efS = efS

    def fit(self, data):
        self.n_samples_fit_ = data.shape[0]
        self.space = str = {
            'l2': 'l2',
            'cosinesimil': 'cosinesimil',
            'cosinesimil_sparse': 'cosinesimil_sparse',
            'cosinesimil_sparse_fast': 'cosinesimil_sparse_fast',
            'l1': 'l1',
        }[self.metric]
        # see more metrics in the manual
        # https://github.com/nmslib/nmslib/tree/master/manual

        if issparse(data) == True:
            print('Sparse input. Proceding without converting...')
        if issparse(data) == False:
            print('Converting input to sparse...')
            try:
                data = data.tocsr()
            except SyntaxError:
                print("Conversion to csr failed. Please provide a numpy array or a pandas dataframe."
                      "Trying internal construction...")
                sys.exit()
            try:
                data = csr_matrix(data)
            except SyntaxError:
                print("Conversion to csr failed. Please provide a numpy array or a pandas dataframe.")

        index_time_params = {'M': self.M, 'indexThreadQty': self.n_jobs, 'efConstruction': self.efC, 'post': 0}

        self.nmslib_ = nmslib.init(method='hnsw',
                                       space='cosinesimil_sparse_fast',
                                       data_type=nmslib.DataType.SPARSE_VECTOR)

        self.nmslib_.addDataPointBatch(data)
        start = time.time()
        self.nmslib_.createIndex(index_time_params)
        end = time.time()
        print('Index-time parameters', 'M:', self.M, 'n_threads:', self.n_jobs, 'efConstruction:', self.efC , 'post:0')
        print('Indexing time = %f (sec)' % (end-start))
        return self

    def transform(self, data):
        start = time.time()
        n_samples_transform = data.shape[0]
        query_time_params = {'efSearch': self.efS}
        print('Setting query-time parameters:', query_time_params)
        self.nmslib_.setQueryTimeParams(query_time_params)

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        self.n_neighbors = self.n_neighbors + 1

        results = self.nmslib_.knnQueryBatch(data, k=self.n_neighbors,
                                             num_threads=self.n_jobs)

        indices, distances = zip(*results)
        indices, distances = np.vstack(indices), np.vstack(distances)

        query_qty = data.shape[0]

        if self.metric == 'sqeuclidean':
            distances **= 2

        indptr = np.arange(0, n_samples_transform * self.n_neighbors + 1,
                           self.n_neighbors)
        kneighbors_graph = csr_matrix((distances.ravel(), indices.ravel(),
                                       indptr), shape=(n_samples_transform,
                                                       self.n_samples_fit_))
        end = time.time()
        print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
              (end - start, float(end - start) / query_qty, self.n_jobs * float(end - start) / query_qty))

        return kneighbors_graph

    def test_efficiency(self, data, data_use=0.1):
        """Test that NMSlibTransformer and KNeighborsTransformer give same results
        """
        self.data_use = data_use

        query_qty = data.shape[0]

        (dismiss, test) = train_test_split(data, test_size=self.data_use)
        query_time_params = {'efSearch': self.efS}
        print('Setting query-time parameters', query_time_params)
        self.nmslib_.setQueryTimeParams(query_time_params)

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        self.n_neighbors = self.n_neighbors + 1
        start = time.time()
        ann_results = self.nmslib_.knnQueryBatch(data, k=self.n_neighbors,
                                                 num_threads=self.n_jobs)
        end = time.time()
        print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
              (end - start, float(end - start) / query_qty, self.n_jobs * float(end - start) / query_qty))

        # Use sklearn for exact neighbor search
        start = time.time()
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors,
                                metric='cosine',
                                algorithm='brute').fit(data)
        knn = nbrs.kneighbors(data)
        end = time.time()
        print('brute-force gold-standart kNN time total=%f (sec), per query=%f (sec)' %
              (end - start, float(end - start) / query_qty))

        recall = 0.0
        for i in range(0, query_qty):
            correct_set = set(knn[1][i])
            ret_set = set(ann_results[i][0])
            recall = recall + float(len(correct_set.intersection(ret_set))) / len(correct_set)
        recall = recall / query_qty
        print('kNN recall %f' % recall)

class Diffusor(TransformerMixin, BaseEstimator):
    """Sklearn estimator for using fast anisotropic diffusion with an anisotropic
    adaptive algorithm as proposed by Setty et al, 2018, and optimized by Sidarta-Oliveira, 2020.

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
                 ann_dist='cosinesimil_sparse_fast',
                 M=30,
                 efC=100,
                 efS=100,
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
    def fit_transform(self, data):
        start = time.time()
        N = data.shape[0]
        if self.ann == True:
            # Construct an approximate k-nearest-neighbors graph
            anbrs = NMSlibTransformer(n_neighbors= self.n_neighbors,
                 metric= self.ann_dist,
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
            nbrs = NearestNeighbors(n_neighbors=int(self.n_neighbors), metric=knn_dist, n_jobs=n_jobs).fit(data)
            kNN = nbrs.kneighbors_graph(data, mode='distance')
            # Adaptive k: distance to cell median nearest neighbors, used for kernel normalization.
            adaptive_k = int(np.floor(self.n_neighbors / 2))
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
            res['kernel'] = kernel

        # Suggest a number of components to use
        vals = np.ravel(res['EigenValues'])
        n_eigs = np.argsort(vals[:(len(vals) - 1)] - vals[1:])[-1] + 1
        if n_eigs < 3:
            n_eigs = np.argsort(vals[:(len(vals) - 1)] - vals[1:])[-2] + 1
        res['Suggested_eigs'] = n_eigs
        end = time.time()
        print('Total computation time=%f (sec), per sample=%f (sec), per sample adjusted for thread number=%f (sec)' %
              (end-start, float(end-start)/N, self.n_jobs * float(end-start)/N))

        return res
