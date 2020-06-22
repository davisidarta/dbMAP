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
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

print(__doc__)


class NMSlibTransformer(TransformerMixin, BaseEstimator):
    """Wrapper for using nmslib as sklearn's KNeighborsTransformer"""

    def __init__(self,
                 n_neighbors=5,
                 metric='cosinesimil_sparse_fast',
                 method='hnsw',
                 n_jobs=-1,
                 data_type='sparse',
                 M=30,
                 efC=100,
                 efS=100,
                 make_sparse=True):
        self.n_neighbors = n_neighbors
        self.method = method
        self.metric = metric
        self.n_jobs = n_jobs
        self.M = M
        self.efC = efC
        self.efS = efS
        self.make_sparse = make_sparse
        self.data_type = data_type

    def fit(self, data):
        self.n_samples_fit_ = data.shape[0]
        self.space = str = {
            'sqeuclidean': 'l2',
            'euclidean': 'l2',
            'cosinesimil': 'cosinesimil',
            'cosinesimil_sparse': 'cosinesimil_sparse',
            'cosinesimil_sparse_fast': 'cosinesimil_sparse_fast',
            'l1': 'l1',
            'l2': 'l2',
            'bit_hamming': 'bit_hamming',
            'bit_jaccard': 'bit_jaccard',
            'jaccard_sparse': 'jaccard_sparse',
        }[self.metric]
        # see more metrics in the manual
        # https://github.com/nmslib/nmslib/tree/master/manual
        index_time_params = {'M': self.M, 'indexThreadQty': self.n_jobs, 'efConstruction': self.efC, 'post': 0}
        if self.make_sparse == True:
            try:
                data = data.tocsr()
            except SyntaxError:
                print("If make_sparse is set to True, please provide an object with a valid .tocsr() attribute.")
                sys.exit()
            try:
                data = csr_matrix(data)
            except SyntaxError:
                print("Conversion to csr failed. Please provide a numpy array or a pandas dataframe.")
        if ((self.data_type != 'sparse') & (self.data_type != 'dense')):
            print("dbMAP 1.1 runs on sparse matrices (csr), numpy arrays and panda dataframes."
                  "Try converting to these formats and run again.")
            sys.exit()
        if self.data_type == 'sparse':
            self.nmslib_ = nmslib.init(method=self.method,
                                       space=self.space,
                                       data_type=nmslib.DataType.SPARSE_VECTOR,
                                       dtype=nmslib.DistType.FLOAT)
        if self.data_type == 'dense':
            self.nmslib_ = nmslib.init(method=self.method,
                                       space=self.space,
                                       data_type=nmslib.DataType.DENSE_VECTOR,
                                       dtype=nmslib.DistType.FLOAT)
        self.nmslib_.addDataPointBatch(data)
        start = time.time()
        self.nmslib_.createIndex(index_time_params)
        end = time.time()
        print('Index-time parameters', index_time_params)
        print('Indexing time = %f' % (end - start))
        return self

    def transform(self, data):
        start = time.time()
        n_samples_transform = data.shape[0]
        query_time_params = {'efSearch': self.efS}
        print('Setting query-time parameters', query_time_params)
        self.nmslib_.setQueryTimeParams(query_time_params)

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        n_neighbors = self.n_neighbors + 1

        results = self.nmslib_.knnQueryBatch(data, k=self.n_neighbors,
                                             num_threads=self.n_jobs)

        indices, distances = zip(*results)
        indices, distances = np.vstack(indices), np.vstack(distances)

        query_qty = data.shape[0]

        if self.metric == 'sqeuclidean':
            distances **= 2

        indptr = np.arange(0, n_samples_transform * n_neighbors + 1,
                           n_neighbors)
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
        n_neighbors = self.n_neighbors + 1
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
