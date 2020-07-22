#####################################
# Author: Davi Sidarta-Oliveira
# School of Medical Sciences,University of Campinas,Brazil
# contact: davisidarta@gmail.com
# Please note that this code has several contributions from Manu Setty et al, Nature
######################################
import time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix, find, issparse
from scipy.sparse.linalg import eigs
from sklearn.base import TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from . import ann
from . import multiscale


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

    verbose : controls verbosity.


    Returns
    -------------
        Diffusion components ['EigenVectors'], associated eigenvalues ['EigenValues'] and suggested number of
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
                 n_components=50,
                 n_neighbors=10,
                 alpha=1,
                 n_jobs=10,
                 ann=True,
                 ann_dist='cosine',
                 M=30,
                 efC=100,
                 efS=100,
                 knn_dist='cosine',
                 verbose=True
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
        self.verbose = verbose

    def fit(self, data):
        """Fits an adaptive anisotropic kernel to the data.
        :param data: input data. Takes in numpy arrays and scipy csr sparse matrices.
        Use with sparse data for top performance. You can adjust a series of
        parameters that can make the process faster and more informational depending
        on your dataset. Read more at https://github.com/davisidarta/dbmap
        """
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
                                          efS=self.efS,
                                          verbose=self.verbose).fit(data)
            ind, dist, grad, knn = anbrs.ind_dist_grad(data)

            # X, y specific stds: Normalize by the distance of median nearest neighbor to account for neighborhood size.
            median_k = np.floor(self.n_neighbors / 2).astype(np.int)
            adap_sd = np.zeros(self.N)
            for i in np.arange(len(adap_sd)):
                adap_sd[i] = np.sort(knn.data[knn.indptr[i]: knn.indptr[i + 1]])[
                    median_k - 1
                    ]
        else:
            # Construct a k-nearest-neighbors graph
            nbrs = NearestNeighbors(n_neighbors=int(self.n_neighbors), metric=self.knn_dist, n_jobs=self.n_jobs).fit(
                data)
            knn = nbrs.kneighbors_graph(data, mode='distance')
            x, y, dist = find(knn)
            # X, y specific stds: Normalize by the distance of median nearest neighbor to account for neighborhood size.
            median_k = np.floor(self.n_neighbors / 2).astype(np.int)
            adap_sd = np.zeros(self.N)
            for i in np.arange(len(adap_sd)):
                adap_sd[i] = np.sort(knn.data[knn.indptr[i]: knn.indptr[i + 1]])[
                    median_k - 1
                    ]

        # define decay as sample's pseudomedian k-nearest-neighbor
        pm = np.interp(adap_sd, (adap_sd.min(), adap_sd.max()), (2, self.n_neighbors))

        # adaptive bandwidth
        bandwidth = dist[:, self.n_neighbors - 1]
        # check for zero bandwidth
        bandwidth = np.maximum(bandwidth, np.finfo(float).eps)

        distances = [d for d in dist]
        indices = [i for i in ind]

        # construct anisotropic adaptive (radius and decay rate) kernel
        kernel = np.concatenate([(distances[i] / bandwidth[i]) ** pm[i]
                                 for i in range(len(distances))]
                                )

        indices = np.concatenate(indices)
        indptr = np.concatenate([[0], np.cumsum([len(d) for d in distances])])
        self.K = csr_matrix((kernel, indices, indptr), shape=(self.N, self.N))

        # construct new, anisotropic kernel
        D = np.array(self.K.toarray().sum(1)).flatten()
        if self.alpha > 0:
            Kc = self.K.tocoo()
            self.K.data = Kc.data / ((D[Kc.row] * D[Kc.col]) ** self.alpha)

        # handle nan, zeros
        self.K.data = np.where(np.isnan(self.K.data), 1, self.K.data)

        return self

    def transform(self, data, n_components=None):
        if n_components is not None:
            self.n_components = n_components
        # Eigen value decomposition
        D, V = eigs(self.K, self.n_components, tol=1e-4, maxiter=self.N)
        D = np.real(D)
        V = np.real(V)
        inds = np.argsort(D)[::-1]
        D = D[inds]
        V = V[:, inds]

        # Normalize by the first diffusion component
        for i in range(V.shape[1]):
            V[:, i] = V[:, i] / np.linalg.norm(V[:, i])

        # Create the results dictionary
        self.res = {'EigenVectors': V, 'EigenValues': D, 'kernel': self.K}
        self.res['EigenVectors'] = pd.DataFrame(self.res['EigenVectors'])
        self.res["EigenValues"] = pd.Series(self.res["EigenValues"])

        mms = multiscale.multiscale(self.res)
        self.res['StructureComponents'] = mms

        end = time.time()
        if self.verbose:
            print('Diffusion time = %f (sec), per sample=%f (sec), per sample adjusted for thread number=%f (sec)' %
                  (end - self.start_time, float(end - self.start_time) / self.N, self.n_jobs * float(end - self.start_time) / self.N))

        return self.res['StructureComponents']

    def ind_dist_grad(self, data, n_components=None, dense=False):
        """Effectively computes on data. Also returns the normalized diffusion distances,
        indexes and gradient obtained by approximating the Laplace-Beltrami operator.
        :param plot_knee: Whether to plot the scree plot of diffusion eigenvalues.
        :param data: input data. Takes in numpy arrays and scipy csr sparse matrices.
        Please use with sparse data for top performance. You can adjust a series of
        parameters that can make the process faster and more informational depending
        on your dataset. Read more at https://github.com/davisidarta/dbmap
        """
        if n_components is not None:
            self.n_components = n_components
        # Eigen value decomposition
        D, V = eigs(self.K, self.n_components, tol=1e-4, maxiter=self.N)
        D = np.real(D)
        V = np.real(V)
        inds = np.argsort(D)[::-1]
        D = D[inds]
        V = V[:, inds]

        # Normalize by the first diffusion component
        for i in range(V.shape[1]):
            V[:, i] = V[:, i] / np.linalg.norm(V[:, i])

        # Create the results dictionary
        self.res = {'EigenVectors': V, 'EigenValues': D, 'kernel': self.K}
        self.res['EigenVectors'] = pd.DataFrame(self.res['EigenVectors'])
        if not issparse(data):
            self.res['EigenValues'] = pd.Series(self.res['EigenValues'])
        self.res["EigenValues"] = pd.Series(self.res["EigenValues"])

        mms = multiscale.multiscale(self.res)
        self.res['StructureComponents'] = mms

        anbrs = ann.NMSlibTransformer(n_neighbors=self.n_neighbors,
                                  metric='cosine',
                                  method='hnsw',
                                  n_jobs=self.n_jobs,
                                  M=self.M,
                                  efC=self.efC,
                                  efS=self.efS,
                                  dense=True,
                                  verbose=self.verbose
                                      ).fit(mms)

        ind, dists, grad, graph = anbrs.ind_dist_grad(mms)

        end = time.time()
        print('Diffusion time = %f (sec), per sample=%f (sec), per sample adjusted for thread number=%f (sec)' %
              (end - self.start_time, float(end - self.start_time) / self.N, self.n_jobs * float(end - self.start_time) / self.N))

        return ind, dists, grad, graph

    def transform_dict(self, data, n_components=None):
        """
        :return: Dictionary containing normalized and multiscaled Diffusion Components
        (['StructureComponents']), their eigenvalues ['EigenValues'], non-normalized
        components (['EigenVectors']) and the kernel used for transformation of distances
        into affinities (['kernel']).
        """

        if n_components is None:
            n_components = self.n_components
        else:
            self.n_components = n_components
        # Eigen value decomposition
        D, V = eigs(self.K, self.n_components, tol=1e-4, maxiter=self.N)
        D = np.real(D)
        V = np.real(V)
        inds = np.argsort(D)[::-1]
        D = D[inds]
        V = V[:, inds]

        # Normalize by the first diffusion component
        for i in range(V.shape[1]):
            V[:, i] = V[:, i] / np.linalg.norm(V[:, i])

        # Create the results dictionary
        self.res = {'EigenVectors': V, 'EigenValues': D, 'kernel': self.K}
        self.res['EigenVectors'] = pd.DataFrame(self.res['EigenVectors'])
        if not issparse(data):
            self.res['EigenValues'] = pd.Series(self.res['EigenValues'])
        self.res["EigenValues"] = pd.Series(self.res["EigenValues"])

        mms = multiscale.multiscale(self.res)
        self.res['StructureComponents'] = mms

        end = time.time()
        print('Diffusion time = %f (sec), per sample=%f (sec), per sample adjusted for thread number=%f (sec)' %
              (end - self.start_time, float(end - self.start_time) / self.N, self.n_jobs * float(end - self.start_time) / self.N))

        return self.res
