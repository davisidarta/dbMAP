#####################################
# NMSLIB approximate-nearest neighbors sklearn wrapper
# NMSLIB: https://github.com/nmslib/nmslib
# Wrapper author: Davi Sidarta-Oliveira
# School of Medical Sciences,University of Campinas,Brazil
# contact: davisidarta@gmail.com
######################################

import time
import sys
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from scipy.sparse import csr_matrix, find, issparse
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

try:
    import nmslib
except ImportError:
    print("The package 'nmslib' is required. Please install it 'with pip3 install nmslib'.")
    sys.exit()

print(__doc__)


class NMSlibTransformer(TransformerMixin, BaseEstimator):
    """
    Wrapper for using nmslib as sklearn's KNeighborsTransformer. This implements
    an escalable approximate k-nearest-neighbors graph on spaces defined by nmslib.
    Read more about nmslib and its various available metrics at
    https://github.com/nmslib/nmslib.

    Calling 'nn <- NMSlibTransformer()' initializes the class with
     neighbour search parameters.

    Parameters
    ----------
    n_neighbors: number of nearest-neighbors to look for. In practice,
                     this should be considered the average neighborhood size and thus vary depending
                     on your number of features, samples and data intrinsic dimensionality. Reasonable values
                     range from 5 to 100. Smaller values tend to lead to increased graph structure
                     resolution, but users should beware that a too low value may render granulated and vaguely
                     defined neighborhoods that arise as an artifact of downsampling. Defaults to 30. Larger
                     values can slightly increase computational time.

    metric: accepted NMSLIB metrics. Should be 'metric' or 'metric_sparse' depending on dense
                or sparse inputs. Defaults to 'cosine_sparse'. Accepted metrics include:
                -'sqeuclidean'
                -'euclidean'
                -'euclidean_sparse'
                -'l1'
                -'l1_sparse'
                -'cosine'
                -'cosine_sparse'
                -'angular'
                -'angular_sparse'
                -'negdotprod'
                -'negdotprod_sparse'
                -'levenshtein'
                -'hamming'
                -'jaccard'
                -'jaccard_sparse'
                -'jansen-shan'

    method: approximate-neighbor search method. Defaults to 'hsnw' (usually the fastest).

    n_jobs: number of threads to be used in computation. Defaults to 10 (~5 cores).

    efC: increasing this value improves the quality of a constructed graph and leads to higher
             accuracy of search. However this also leads to longer indexing times. A reasonable
             range is 100-2000. Defaults to 100.

    efS: similarly to efC, improving this value improves recall at the expense of longer
             retrieval time. A reasonable range is 100-2000.

    M: defines the maximum number of neighbors in the zero and above-zero layers during HSNW
           (Hierarchical Navigable Small World Graph). However, the actual default maximum number
           of neighbors for the zero layer is 2*M. For more information on HSNW, please check
           https://arxiv.org/abs/1603.09320. HSNW is implemented in python via NMSLIB. Please check
           more about NMSLIB at https://github.com/nmslib/nmslib .

    dense: Whether to force the algorithm to use dense data, such as np.ndarrays and pandas DataFrames.

    :returns: Class for really fast approximate-nearest-neighbors search.

    Example
    -------------

    import numpy as np
    from sklearn.datasets import load_digits
    from scipy.sparse import csr_matrix
    from dbmap.ann import NMSlibTransformer

    # Load the MNIST digits data, convert to sparse for speed
    digits = load_digits()
    data = csr_matrix(digits)

    # Start class with parameters
    nn = NMSlibTransformer()
    nn = nn.fit(data)

    # Obtain kNN graph
    knn = nn.transform(data)

    # Obtain kNN indices, distances and distance gradient
    ind, dist, grad = nn.ind_dist_grad(data)

    # Test for recall efficiency during approximate nearest neighbors search
    test = nn.test_efficiency(data)


    """

    def __init__(self,
                 n_neighbors=30,
                 metric='cosine_sparse',
                 method='hnsw',
                 n_jobs=10,
                 efC=100,
                 efS=100,
                 M=30,
                 dense=False
                 ):

        self.n_neighbors = n_neighbors
        self.method = method
        self.metric = metric
        self.n_jobs = n_jobs
        self.M = M
        self.efC = efC
        self.efS = efS
        self.space = {
            'sqeuclidean': 'l2',
            'euclidean': 'l2',
            'euclidean_sparse': 'l2_sparse',
            'cosine': 'cosinesimil',
            'cosine_sparse': 'cosinesimil_sparse_fast',
            'l1': 'l1',
            'l1_sparse': 'l1_sparse',
            'linf': 'linf',
            'linf_sparse': 'linf_sparse',
            'angular': 'angulardist',
            'angular_sparse': 'angulardist_sparse_fast',
            'negdotprod': 'negdotprod',
            'negdotprod_sparse': 'negdotprod_sparse_fast',
            'levenshtein': 'leven',
            'hamming': 'bit_hamming',
            'jaccard': 'bit_jaccard',
            'jaccard_sparse': 'jaccard_sparse',
            'jansen-shan': 'jsmetrfastapprox'
        }[self.metric]
        self.dense = dense

    def fit(self, data):
        # see more metrics in the manual
        # https://github.com/nmslib/nmslib/tree/master/manual

        self.n_samples_fit_ = data.shape[0]

        index_time_params = {'M': self.M, 'indexThreadQty': self.n_jobs, 'efConstruction': self.efC, 'post': 0}

        if self.dense:
            self.nmslib_ = nmslib.init(method=self.method,
                                       space=self.space,
                                       data_type=nmslib.DataType.DENSE_VECTOR)

        else:
            if issparse(data) == True:
                print('Sparse input. Proceding without converting...')
                if isinstance(data, np.ndarray):
                    data = csr_matrix(data)
            if issparse(data) == False:

                print('Input data is ' + str(type(data)) + ' .Converting input to sparse...')

                import pandas as pd
                if isinstance(data, pd.DataFrame):
                    data = csr_matrix(data.values.T)

        self.n_samples_fit_ = data.shape[0]

        index_time_params = {'M': self.M, 'indexThreadQty': self.n_jobs, 'efConstruction': self.efC, 'post': 0}

        if (issparse(data) == True) and (not self.dense) and (not isinstance(data,np.ndarray)):
            self.nmslib_ = nmslib.init(method=self.method,
                                       space=self.space,
                                       data_type=nmslib.DataType.SPARSE_VECTOR)

        else:
            self.nmslib_ = nmslib.init(method=self.method,
                                       space=self.space,
                                       data_type=nmslib.DataType.DENSE_VECTOR)

        self.nmslib_.addDataPointBatch(data)
        start = time.time()
        self.nmslib_.createIndex(index_time_params)
        end = time.time()
        print('Index-time parameters', 'M:', self.M, 'n_threads:', self.n_jobs, 'efConstruction:', self.efC, 'post:0')
        print('Indexing time = %f (sec)' % (end - start))

        return self

    def transform(self, data):
        start = time.time()
        n_samples_transform = data.shape[0]
        query_time_params = {'efSearch': self.efS}
        print('Query-time parameter efSearch:', self.efS)
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

    def ind_dist_grad(self, data):
        start = time.time()
        n_samples_transform = data.shape[0]
        query_time_params = {'efSearch': self.efS}
        print('Query-time parameter efSearch:', self.efS)
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
        x, y, dists = find(kneighbors_graph)

        # Define gradients
        grad = np.gradient(dists)

        if self.metric == 'cosine' or self.metric == 'cosine_sparse':
            norm_x = 0.0
            norm_y = 0.0
            for i in range(x.shape[0]):
                norm_x += x[i] ** 2
                norm_y += y[i] ** 2
            if norm_x == 0.0 and norm_y == 0.0:
                grad = np.zeros(x.shape)
            elif norm_x == 0.0 or norm_y == 0.0:
                grad = np.zeros(x.shape)
            else:
                grad = -(x * dists - y * norm_x) / np.sqrt(norm_x ** 3 * norm_y)

        if self.metric == 'euclidean' or self.metric == 'euclidean_sparse':
            grad = x - y / (1e-6 + np.sqrt(dists))

        if self.metric == 'sqeuclidean':
            grad = x - y / (1e-6 + dists)

        if self.metric == 'linf' or self.metric == 'linf_sparse':
            result = 0.0
            max_i = 0
            for i in range(x.shape[0]):
                v = np.abs(x[i] - y[i])
                if v > result:
                    result = dists
                    max_i = i
            grad = np.zeros(x.shape)
            grad[max_i] = np.sign(x[max_i] - y[max_i])

        end = time.time()

        print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
              (end - start, float(end - start) / query_qty, self.n_jobs * float(end - start) / query_qty))

        return indices, distances, grad, kneighbors_graph

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


