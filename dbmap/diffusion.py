#####################################
# Author: Davi Sidarta-Oliveira
# School of Medical Sciences,University of Campinas,Brazil
# contact: davisidarta@fcm.unicamp.com
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
                 p=None,
                 M=30,
                 efC=100,
                 efS=100,
                 knn_dist='cosine',
                 kernel_use='decay',
                 transitions=True,
                 eigengap=True,
                 norm=False,
                 verbose=True
                 ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.ann = ann
        self.ann_dist = ann_dist
        self.p = p
        self.M = M
        self.efC = efC
        self.efS = efS
        self.knn_dist = knn_dist
        self.kernel_use = kernel_use
        self.transitions = transitions
        self.eigengap = eigengap
        self.norm = norm
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
        if self.kernel_use == 'orig' and self.transitions == 'False':
            print('The original kernel implementation used transitions computation. Set `transitions` to `True`'
                  'for similar results.')
        if self.kernel_use not in ['simple', 'simple_adaptive', 'decay', 'decay_adaptive']:
            raise Exception('Kernel must be either \'simple\', \'simple_adaptive\', \'decay\' or \'decay_adaptive\'.') 
        if self.ann:
            if self.ann_dist == 'lp' and self.p < 1:
                print('Fractional L norms are slower to compute. Computations are faster for fractions'
                      ' of the form \'1/2ek\', where k is a small integer (i.g. 0.5, 0.25) ')
            # Construct an approximate k-nearest-neighbors graph
            anbrs = ann.NMSlibTransformer(n_neighbors=self.n_neighbors,
                                          metric=self.ann_dist,
                                          p=self.p,
                                          method='hnsw',
                                          n_jobs=self.n_jobs,
                                          M=self.M,
                                          efC=self.efC,
                                          efS=self.efS,
                                          verbose=self.verbose).fit(data)
            knn = anbrs.transform(data)
            # X, y specific stds: Normalize by the distance of median nearest neighbor to account for neighborhood size.
            median_k = np.floor(self.n_neighbors / 2).astype(np.int)
            adap_sd = np.zeros(self.N)
            for i in np.arange(len(adap_sd)):
                adap_sd[i] = np.sort(knn.data[knn.indptr[i]: knn.indptr[i + 1]])[
                    median_k - 1
                    ]
        else:
            if self.ann_dist == 'lp':
                raise Exception('Generalized Lp distances are available only with `ann` set to True.')

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

        # Distance metrics
        x, y, dists = find(knn)  # k-nearest-neighbor distances

        # define decay as sample's pseudomedian k-nearest-neighbor
        pm = np.interp(adap_sd, (adap_sd.min(), adap_sd.max()), (2, self.n_neighbors))

        # adaptive neighborhood size
        if self.kernel_use == 'simple_adaptive' or self.kernel_use == 'decay_adaptive':
            # increase neighbor search:
            anbrs_new = ann.NMSlibTransformer(n_neighbors=int(self.n_neighbors + (self.n_neighbors - pm.max())),
                                              metric=self.ann_dist,
                                              method='hnsw',
                                              n_jobs=self.n_jobs,
                                              p=self.p,
                                              M=self.M,
                                              efC=self.efC,
                                              efS=self.efS).fit(data)
            knn_new = anbrs_new.transform(data)

            x_new, y_new, dists_new = find(knn_new)

            # adaptive neighborhood size
            adap_nbr = np.zeros(self.N)
            for i in np.arange(len(adap_nbr)):
                adap_k = int(np.floor(pm[i]))
                adap_nbr[i] = np.sort(knn_new.data[knn_new.indptr[i]: knn_new.indptr[i + 1]])[
                    int(pm[i])
                ]

        if self.kernel_use == 'simple':
            # X, y specific stds
            dists = dists / (adap_sd[x] + 1e-10)  # Normalize by the distance of median nearest neighbor
            W = csr_matrix((np.exp(-dists), (x, y)), shape=[self.N, self.N])  # Normalized distances

        if self.kernel_use == 'simple_adaptive':
            # X, y specific stds
            dists = dists_new / (adap_nbr[x_new] + 1e-10)  # Normalize by normalized contribution to neighborhood size.
            W = csr_matrix((np.exp(-dists), (x_new, y_new)), shape=[self.N, self.N])  # Normalized distances

        if self.kernel_use == 'decay':
            # X, y specific stds
            dists = (dists / (adap_sd[x] + 1e-10)) ** np.power(2, ((self.n_neighbors - pm[x]) / pm[x]))
            W = csr_matrix((np.exp(-dists), (x, y)), shape=[self.N, self.N])  # Normalized distances

        if self.kernel_use == 'decay_adaptive':
            # X, y specific stds
            dists = (dists_new / (adap_nbr[x_new]+ 1e-10)) ** np.power(2, (((int(self.n_neighbors + (self.n_neighbors - pm.max()))) - pm[x_new]) / pm[x_new]))  # Normalize by normalized contribution to neighborhood size.
            W = csr_matrix((np.exp(-dists), (x_new, y_new)), shape=[self.N, self.N])  # Normalized distances

        # Kernel construction
        kernel = W + W.T
        self.K = kernel

        # handle nan, zeros
        self.K.data = np.where(np.isnan(self.K.data), 1, self.K.data)
        # Diffusion through Markov chain

        D = np.ravel(self.K.sum(axis=1))
        if self.alpha > 0:
            # L_alpha
            D[D != 0] = D[D != 0] ** (-self.alpha)
            mat = csr_matrix((D, (range(self.N), range(self.N))), shape=[self.N, self.N])
            kernel = mat.dot(self.K).dot(mat)
            D = np.ravel(kernel.sum(axis=1))

        D[D != 0] = 1 / D[D != 0]

        # Setting the diffusion operator
        if not self.norm:
            self.T = csr_matrix((D, (range(self.N), range(self.N))), shape=[self.N, self.N]).dot(self.K)
        else:
            self.K = kernel
            self.T = csr_matrix((D, (range(self.N), range(self.N))), shape=[self.N, self.N]).dot(self.K)

        return self


    def transform(self, data):

        # Fit an optimal number of components based on the eigengap
        # Use user's  or default initial guess
        multiplier = self.N // 10e4
        # initial eigen value decomposition
        if self.transitions:
            D, V = eigs(self.T, self.n_components, tol=1e-4, maxiter=self.N)
        else:
            D, V = eigs(self.K, self.n_components, tol=1e-4, maxiter=self.N)
        D = np.real(D)
        V = np.real(V)
        inds = np.argsort(D)[::-1]
        D = D[inds]
        V = V[:, inds]
        # Normalize by the first diffusion component
        for i in range(V.shape[1]):
            V[:, i] = V[:, i] / np.linalg.norm(V[:, i])
        vals = np.array(V)
        pos = np.sum(vals > 0, axis=0)
        residual = np.sum(vals < 0, axis=0)

        if self.eigengap and len(residual) < 1:
            #expand eigendecomposition
            target = self.n_components * multiplier
            while residual < 3:
                print('Eigengap not found for determined number of components. Expanding eigendecomposition to '
                      + str(target) + 'components.')
                if self.transitions:
                    D, V = eigs(self.T, target, tol=1e-4, maxiter=self.N)
                else:
                    D, V = eigs(self.K, target, tol=1e-4, maxiter=self.N)
                D = np.real(D)
                V = np.real(V)
                inds = np.argsort(D)[::-1]
                D = D[inds]
                V = V[:, inds]
                # Normalize by the first diffusion component
                for i in range(V.shape[1]):
                    V[:, i] = V[:, i] / np.linalg.norm(V[:, i])
                vals = np.array(V)
                residual = np.sum(vals < 0, axis=0)
                target = target * 2

        if len(residual) > 30:
            self.n_components = len(pos) + 15
            # adapted eigen value decomposition
            if self.transitions:
               D, V = eigs(self.T, self.n_components, tol=1e-4, maxiter=self.N)
            else:
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
        else:
            # Fit an optimal number of components based on the eigengap
            # Use user's  or default initial guess
            multiplier = self.N // 10e4
            # initial eigen value decomposition
            if self.transitions:
                D, V = eigs(self.T, self.n_components, tol=1e-4, maxiter=self.N)
            else:
                D, V = eigs(self.K, self.n_components, tol=1e-4, maxiter=self.N)
            D = np.real(D)
            V = np.real(V)
            inds = np.argsort(D)[::-1]
            D = D[inds]
            V = V[:, inds]
            # Normalize by the first diffusion component
            for i in range(V.shape[1]):
                V[:, i] = V[:, i] / np.linalg.norm(V[:, i])
            vals = np.array(V)
            pos = np.sum(vals > 0, axis=0)
            residual = np.sum(vals < 0, axis=0)

            if self.eigengap and len(residual) < 1:
                #expand eigendecomposition
                target = self.n_components * multiplier
                while residual < 3:
                    print('Eigengap not found for determined number of components. Expanding eigendecomposition to '
                          + str(target) + 'components.')
                    if self.transitions:
                        D, V = eigs(self.T, target, tol=1e-4, maxiter=self.N)
                    else:
                        D, V = eigs(self.K, target, tol=1e-4, maxiter=self.N)
                    D = np.real(D)
                    V = np.real(V)
                    inds = np.argsort(D)[::-1]
                    D = D[inds]
                    V = V[:, inds]
                    # Normalize by the first diffusion component
                    for i in range(V.shape[1]):
                        V[:, i] = V[:, i] / np.linalg.norm(V[:, i])
                    vals = np.array(V)
                    residual = np.sum(vals < 0, axis=0)
                    target = target * 2

            if len(residual) > 30:
                self.n_components = len(pos) + 15
                # adapted eigen value decomposition
                if self.transitions:
                    D, V = eigs(self.T, self.n_components, tol=1e-4, maxiter=self.N)
                else:
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

    def return_dict(self):
        """
        :return: Dictionary containing normalized and multiscaled Diffusion Components
        (['StructureComponents']), their eigenvalues ['EigenValues'], non-normalized
        components (['EigenVectors']) and the kernel used for transformation of distances
        into affinities (['kernel']).
        """

        return self.res
