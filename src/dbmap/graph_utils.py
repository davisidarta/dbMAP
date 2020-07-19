import numpy as np
from scipy.sparse import coo_matrix, issparse

from . import diffusion
from . import ann
from . import multiscale

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf
INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


def fuzzy_simplicial_set_nmslib(
        X,
        n_neighbors,
        knn_indices=None,
        knn_dists=None,
        nmslib_metric='cosine',
        nmslib_n_jobs=None,
        nmslib_efC=100,
        nmslib_efS=100,
        nmslib_M=30,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        apply_set_operations=True,
        verbose=False):
    """Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The data to be modelled as a fuzzy simplicial set.
    n_neighbors: int
        The number of neighbors to use to approximate geodesic distance.
        Larger numbers induce more global estimates of the manifold that can
        miss finer detail, while smaller values will focus on fine manifold
        structure to the detriment of the larger picture.
    nmslib_metric: str (optional, default 'cosine')
        accepted NMSLIB metrics. Accepted metrics include:
                -'sqeuclidean'
                -'euclidean'
                -'l1'
                -'l1_sparse'
                -'cosine'
                -'angular'
                -'negdotprod'
                -'levenshtein'
                -'hamming'
                -'jaccard'
                -'jansen-shan'
    nmslib_n_jobs: int (optional, default None)
        Number of threads to use for approximate-nearest neighbor search.
    nmslib_efC: int (optional, default 100)
        increasing this value improves the quality of a constructed graph and leads to higher
        accuracy of search. However this also leads to longer indexing times. A reasonable
        range is 100-2000.
    nmslib_efS: int (optional, default 100)
        similarly to efC, improving this value improves recall at the expense of longer
        retrieval time. A reasonable range is 100-2000.
    nmslib_M: int (optional, default 30).
        defines the maximum number of neighbors in the zero and above-zero layers during HSNW
        (Hierarchical Navigable Small World Graph). However, the actual default maximum number
        of neighbors for the zero layer is 2*M. For more information on HSNW, please check
        https://arxiv.org/abs/1603.09320. HSNW is implemented in python via NMSLIB. Please check
        more about NMSLIB at https://github.com/nmslib/nmslib .    n_epochs: int (optional, default None)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If None is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).
    knn_indices: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the indices of the k-nearest neighbors as a row for
        each data point.
    knn_dists: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the distances of the k-nearest neighbors as a row for
        each data point.
    set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.
    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    Returns
    -------
    fuzzy_simplicial_set: coo_matrix
        A fuzzy simplicial set represented as a sparse matrix. The (i,
        j) entry of the matrix represents the membership strength of the
        1-simplex between the ith and jth sample points.
    """
    if knn_indices is None or knn_dists is None:
        print('Running fast approximate nearest neighbors with NMSLIB using HNSW...')
        if nmslib_metric is not {'sqeuclidean',
                                 'euclidean',
                                 'l1',
                                 'cosine',
                                 'angular',
                                 'negdotprod',
                                 'levenshtein',
                                 'hamming',
                                 'jaccard',
                                 'jansen-shan'}:
            print('Please input a metric compatible with NMSLIB when use_nmslib is set to True')
        knn_indices, knn_dists = approximate_n_neighbors(X,
                                                         n_neighbors=n_neighbors,
                                                         metric=nmslib_metric,
                                                         method='hnsw',
                                                         n_jobs=nmslib_n_jobs,
                                                         efC=nmslib_efC,
                                                         efS=nmslib_efS,
                                                         M=nmslib_M,
                                                         verbose=verbose)

    knn_dists = knn_dists.astype(np.float32)

    sigmas, rhos = smooth_knn_dist(
        knn_dists, float(n_neighbors), local_connectivity=float(local_connectivity),
    )

    rows, cols, vals = compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos
    )

    result = coo_matrix(
        (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
    )
    result.eliminate_zeros()

    if apply_set_operations:
        transpose = result.transpose()

        prod_matrix = result.multiply(transpose)

        result = (
                set_op_mix_ratio * (result + transpose - prod_matrix)
                + (1.0 - set_op_mix_ratio) * prod_matrix
        )

    result.eliminate_zeros()

    return result, sigmas, rhos


def compute_connectivities_adapmap(
        data,
        n_components=100,
        n_neighbors=30,
        alpha=0.0,
        n_jobs=10,
        ann=True,
        ann_dist='cosine',
        M=30,
        efC=100,
        efS=100,
        knn_dist='euclidean',
        kernel_use='sidarta',
        sensitivity=1,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,

):
    """
        Sklearn estimator for using fast anisotropic diffusion with an anisotropic
        adaptive algorithm as proposed by Setty et al, 2018, and optimized by Sidarta-Oliveira, 2020.
        This procedure generates diffusion components that effectivelly carry the maximum amount of
        information regarding the data geometric structure (structure components).
        These structure components then undergo a fuzzy-union of simplicial sets. This step is
        from umap.fuzzy_simplicial_set [McInnes18]_. Given a set of data X, a neighborhood size, and a measure of distance
        compute the fuzzy simplicial set (here represented as a fuzzy graph in the form of a sparse matrix) associated to the data. This is done by
        locally approximating geodesic distance at each point, creating a fuzzy simplicial set for each such point, and then combining all the local
        fuzzy simplicial sets into a global one via a fuzzy union.

    Parameters
    ----------
    n_components : Number of diffusion components to compute. Defaults to 100. We suggest larger values if
                   analyzing more than 10,000 cells.

    n_neighbors : Number of k-nearest-neighbors to compute. The adaptive kernel will normalize distances by each cell
                  distance of its median neighbor.

    knn_dist : Distance metric for building kNN graph. Defaults to 'euclidean'. 

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
    from scipy.sparse import csr_matrix, issparse
    if issparse(data) == True:
        print('Sparse input. Proceding without converting...')
        if isinstance(data, np.ndarray):
            data = csr_matrix(data)
    if issparse(data) == False:
        print('Converting input to sparse...')
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            data = csr_matrix(data.values.T)
    elif isinstance(data, np.ndarray):
        data = data

    diff = diffusion.Diffusor(
        n_components=n_components,
        n_neighbors=n_neighbors,
        alpha=alpha,
        n_jobs=n_jobs,
        ann=ann,
        ann_dist=ann_dist,
        M=M,
        efC=efC,
        efS=efS,
        knn_dist=knn_dist,
        kernel_use=kernel_use,
        sensitivity=sensitivity).fit(data)
    knn_indices, knn_distances, knn_grad, knn_graph = diff.ind_dist_grad(data, dense=False)

    if not issparse(knn_graph):
        knn_graph = csr_matrix(knn_graph)

    distances, connectivities, rhos = fuzzy_simplicial_set_nmslib(
        knn_graph,
        n_neighbors,
        knn_indices=knn_indices,
        knn_dists=knn_distances,
        nmslib_metric=ann_dist,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
    )
    if isinstance(connectivities, tuple):
        # In umap-learn 0.4, this returns (result, sigmas, rhos)
        connectivities = connectivities[0]
    if issparse(connectivities):
        if isinstance(connectivities, np.ndarray):
            connectivities = csr_matrix(connectivities)
        else:
            connectivities = connectivities.tocsr()

    return knn_graph, connectivities


def get_sparse_matrix_from_indices_distances_dbmap(knn_indices, knn_dists, n_obs, n_neighbors):
    rows = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_obs * n_neighbors), dtype=np.float64)

    for i in range(knn_indices.shape[0]):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            else:
                val = knn_dists[i, j]

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    result = coo_matrix((vals, (rows, cols)),
                        shape=(n_obs, n_obs))
    result.eliminate_zeros()
    return result.tocsr()


def approximate_n_neighbors(data,
                            n_neighbors=30,
                            metric='cosine',
                            method='hnsw',
                            n_jobs=10,
                            efC=100,
                            efS=100,
                            M=30,
                            dense=False,
                            verbose=False
                            ):
    """
    Simple function using NMSlibTransformer from dbmap.ann. This implements a very fast
    and scalable approximate k-nearest-neighbors graph on spaces defined by nmslib.
    Read more about nmslib and its various available metrics at
    https://github.com/nmslib/nmslib. Read more about dbMAP at
    https://github.com/davisidarta/dbMAP.


    Parameters
    ----------
    n_neighbors: number of nearest-neighbors to look for. In practice,
                     this should be considered the average neighborhood size and thus vary depending
                     on your number of features, samples and data intrinsic dimensionality. Reasonable values
                     range from 5 to 100. Smaller values tend to lead to increased graph structure
                     resolution, but users should beware that a too low value may render granulated and vaguely
                     defined neighborhoods that arise as an artifact of downsampling. Defaults to 30. Larger
                     values can slightly increase computational time.

    metric: accepted NMSLIB metrics. Defaults to 'cosine'. Accepted metrics include:
                -'sqeuclidean'
                -'euclidean'
                -'l1'
                -'cosine'
                -'angular'
                -'negdotprod'
                -'levenshtein'
                -'hamming'
                -'jaccard'
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

    :returns: k-nearest-neighbors indices and distances. Can be customized to also return
        return the k-nearest-neighbors graph and its gradient.

    Example
    -------------

    knn_indices, knn_dists = approximate_n_neighbors(data)


    """
    print("Finding Approximate Nearest Neighbors...")
    import dbmap as dm
    from scipy.sparse import csr_matrix, issparse
    if issparse(data):
        data = data.tocsr()
    else:
        data = csr_matrix(data)

    anbrs = dm.ann.NMSlibTransformer(n_neighbors=n_neighbors,
                                     metric=metric,
                                     method=method,
                                     n_jobs=n_jobs,
                                     efC=efC,
                                     efS=efS,
                                     M=M, 
                                     dense=dense,
                                     verbose=verbose).fit(data)

    knn_inds, knn_distances, grad, knn_graph = anbrs.ind_dist_grad(data)

    return knn_inds, knn_distances


def compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos):
    """Construct the membership strength data for the 1-skeleton of each local
    fuzzy simplicial set -- this is formed as a sparse matrix where each row is
    a local fuzzy simplicial set, with a membership strength for the
    1-simplex to each other data point.
    Parameters
    ----------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.
    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.
    sigmas: array of shape(n_samples)
        The normalization factor derived from the metric tensor approximation.
    rhos: array of shape(n_samples)
        The local connectivity adjustment.
    Returns
    -------
    rows: array of shape (n_samples * n_neighbors)
        Row data for the resulting sparse matrix (coo format)
    cols: array of shape (n_samples * n_neighbors)
        Column data for the resulting sparse matrix (coo format)
    vals: array of shape (n_samples * n_neighbors)
        Entries for the resulting sparse matrix (coo format)
    """
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    return rows, cols, vals


def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    """Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In essence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.
    Parameters
    ----------
    distances: array of shape (n_samples, n_neighbors)
        Distances to nearest neighbors for each samples. Each row should be a
        sorted list of distances to a given samples nearest neighbors.
    k: float
        The number of nearest neighbors to approximate for.
    n_iter: int (optional, default 64)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.
    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.
    bandwidth: float (optional, default 1)
        The target bandwidth of the kernel, larger values will produce
        larger return values.
    Returns
    -------
    knn_dist: array of shape (n_samples,)
        The distance to kth nearest neighbor, as suitably approximated.
    nn_dist: array of shape (n_samples,)
        The distance to the 1st nearest neighbor for each point.
    """
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    result = np.zeros(distances.shape[0], dtype=np.float32)

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (
                            non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0

            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        result[i] = mid

        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                result[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if result[i] < MIN_K_DIST_SCALE * mean_distances:
                result[i] = MIN_K_DIST_SCALE * mean_distances

    return result, rho


def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""
    import igraph as ig

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except:
        pass
    if g.vcount() != adjacency.shape[0]:
        print(
            f'The constructed graph has only {g.vcount()} nodes. '
            'Your adjacency matrix contained redundant nodes.'
        )
    return g
