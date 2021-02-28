from warnings import warn
import numpy as np
from . import ann
from . import diffusion
import scipy.sparse
import scipy.sparse.csgraph
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import pairwise_distances

def component_layout(
    data,
    n_components,
    flavor,
    dim,
    random_state,
    metric="cosine",
    p=None,
    precomputed=False,
):
    """Provide a layout relating the separate connected components. This is done
    by taking the centroid of each component and then performing a spectral embedding
    of the centroids. Derived from UMAP initialization.
    Parameters
    ----------
    data: array of shape (n_samples, n_samples)
        Distance matrix from data.
    n_components: int
        The number of distinct components to be layed out.
    flavor: str (optional, default 'adaptive')
        Which method to use to build the similarity matrix. If 'kernel', builds
        an adaptive diffusion kernel connectivity matrix. If 'transitions', uses the
        adaptive transition probabilities as an affinity metric. Both methods take the settings
        `kernel_use` and 'norm' to customize the kernel adaptability level.
        If 'uniform', simply takes exponential pairwise euclidean distances of the data matrix,
        as in the original UMAP implementation, and the `kernel_use` and `norm` parameters are
        set to `None`.
    kernel_use:

    norm:

    component_labels: array of shape (n_samples)
        For each vertex in the graph the label of the component to
        which the vertex belongs.
    dim: int
        The chosen embedding dimension.
    metric: string or callable (optional, default 'cosine')
        The metric used to measure distances among the source data points.
    p: float (optional, default None)
        The p norm to be used when using '
    metric_kwds: dict (optional, default {})
        Keyword arguments to be passed to the metric function.
        If metric is 'precomputed', 'linkage' keyword can be used to specify
        'average', 'complete', or 'single' linkage. Default is 'average'
    n_jobs: int
        Number of threads for nearest-neighbor search.
    Returns
    -------
    component_embedding: array of shape (n_components, dim)
        The ``dim``-dimensional embedding of the ``n_components``-many
        connected components.
    """

    component_centroids = np.empty((n_components, data.shape[1]), dtype=np.float64)
    if flavor == 'uniform':
        if precomputed:
            distance_matrix = data

        else:
            anbrs = ann.NMSlibTransformer(n_neighbors=n_nbrs,
                                          metric=metric,
                                          p=p,
                                          method='hnsw',
                                          n_jobs=n_jobs,
                                          verbose=False).fit(data)

            knn = anbrs.transform(data)
            x, y, dists = find(knn)
            distance_matrix = dists

        affinity_matrix = np.exp(-(distance_matrix ** 2))

    if flavor == 'adaptive':
        if precomputed:
            affinity_matrix = data
        else:
            diff = diffusion.Diffusor(n_neighbors=n_nbrs,
                                          ann_dist=metric,
                                          n_jobs=n_jobs,
                                          verbose=False).fit(data)
            affinity_matrix = diff.K

    if flavor == 'transitions':
        if precomputed:
            affinity_matrix = data

        else:
            diff = diffusion.Diffusor(n_neighbors=50,
                                          ann_dist=metric,
                                          n_jobs=n_jobs,
                                          kernel_use=kernel_use,
                                          norm=norm,
                                          verbose=False).fit(data)
            affinity_matrix = diff.T

    component_embedding = SpectralEmbedding(n_components=dim,
                                            affinity="precomputed",
                                            random_state=random_state,
                                            n_neighbors=n_nbrs).fit_transform(affinity_matrix)
    component_embedding /= component_embedding.max()

    return component_embedding


def multi_component_layout(data,
    n_components,
    flavor,
    component_labels,
    dim,
    random_state,
    nn_method='nmslib',
    metric="cosine",
    metric_kwds={}
):
    """Specialised layout algorithm for dealing with graphs with many connected components.
    This will first fid relative positions for the components by spectrally embedding
    their centroids, then spectrally embed each individual connected component positioning
    them according to the centroid embeddings. This provides a decent embedding of each
    component while placing the components in good relative positions to one another.
    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data -- required so we can generate centroids for each
        connected component of the graph.
    graph: sparse matrix
        The adjacency matrix of the graph to be emebdded.
    n_components: int
        The number of distinct components to be layed out.
    component_labels: array of shape (n_samples)
        For each vertex in the graph the label of the component to
        which the vertex belongs.
    dim: int
        The chosen embedding dimension.
    metric: string or callable (optional, default 'euclidean')
        The metric used to measure distances among the source data points.
    metric_kwds: dict (optional, default {})
        Keyword arguments to be passed to the metric function.
    Returns
    -------
    embedding: array of shape (n_samples, dim)
        The initial embedding of ``graph``.
    """

    result = np.empty((graph.shape[0], dim), dtype=np.float32)

    if n_components > 2 * dim:
        meta_embedding = component_layout(
            data,
            n_components,
            component_labels,
            dim,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )
    else:
        k = int(np.ceil(n_components / 2.0))
        base = np.hstack([np.eye(k), np.zeros((k, dim - k))])
        meta_embedding = np.vstack([base, -base])[:n_components]

    for label in range(n_components):
        component_graph = graph.tocsr()[component_labels == label, :].tocsc()
        component_graph = component_graph[:, component_labels == label].tocoo()

        distances = pairwise_distances([meta_embedding[label]], meta_embedding)
        data_range = distances[distances > 0.0].min() / 2.0

        if component_graph.shape[0] < 2 * dim:
            result[component_labels == label] = (
                random_state.uniform(
                    low=-data_range,
                    high=data_range,
                    size=(component_graph.shape[0], dim),
                )
                + meta_embedding[label]
            )
            continue

        diag_data = np.asarray(component_graph.sum(axis=0))
        # standard Laplacian
        # D = scipy.sparse.spdiags(diag_data, 0, graph.shape[0], graph.shape[0])
        # L = D - graph
        # Normalized Laplacian
        I = scipy.sparse.identity(component_graph.shape[0], dtype=np.float64)
        D = scipy.sparse.spdiags(
            1.0 / (np.sqrt(diag_data)+10e-6),
            0,
            component_graph.shape[0],
            component_graph.shape[0],
        )
        L = I - D * component_graph * D

        k = dim + 1
        num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(component_graph.shape[0])))
        try:
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                L,
                k,
                which="SM",
                ncv=num_lanczos_vectors,
                tol=1e-4,
                v0=np.ones(L.shape[0]),
                maxiter=graph.shape[0] * 5,
            )
            order = np.argsort(eigenvalues)[1:k]
            component_embedding = eigenvectors[:, order]
            expansion = data_range / np.max(np.abs(component_embedding))
            component_embedding *= expansion
            result[component_labels == label] = (
                component_embedding + meta_embedding[label]
            )
        except scipy.sparse.linalg.ArpackError:
            warn(
                "WARNING: spectral initialisation failed! The eigenvector solver\n"
                "failed. This is likely due to too small an eigengap. Consider\n"
                "adding some noise or jitter to your data.\n\n"
                "Falling back to random initialisation!"
            )
            result[component_labels == label] = (
                random_state.uniform(
                    low=-data_range,
                    high=data_range,
                    size=(component_graph.shape[0], dim),
                )
                + meta_embedding[label]
            )

    return result


def spectral_layout(data, graph, dim, random_state, metric="euclidean", metric_kwds={}):
    """Given a graph compute the spectral embedding of the graph. This is
    simply the eigenvectors of the laplacian of the graph. Here we use the
    normalized laplacian.
    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data
    graph: sparse matrix
        The (weighted) adjacency matrix of the graph as a sparse matrix.
    dim: int
        The dimension of the space into which to embed.
    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.
    Returns
    -------
    embedding: array of shape (n_vertices, dim)
        The spectral embedding of the graph.
    """
    n_samples = graph.shape[0]
    n_components, labels = scipy.sparse.csgraph.connected_components(graph)

    if n_components > 1:
        return multi_component_layout(
            data,
            graph,
            n_components,
            labels,
            dim,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )

    diag_data = np.asarray(graph.sum(axis=0))
    # standard Laplacian
    # D = scipy.sparse.spdiags(diag_data, 0, graph.shape[0], graph.shape[0])
    # L = D - graph
    # Normalized Laplacian
    I = scipy.sparse.identity(graph.shape[0], dtype=np.float64)
    D = scipy.sparse.spdiags(
        1.0 / (np.sqrt(diag_data)+10e-8), 0, graph.shape[0], graph.shape[0]
    )
    L = I - D * graph * D

    k = dim + 1
    num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))
    try:
        if L.shape[0] < 2000000:
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                L,
                k,
                which="SM",
                ncv=num_lanczos_vectors,
                tol=1e-4,
                v0=np.ones(L.shape[0]),
                maxiter=graph.shape[0] * 5,
            )
        else:
            eigenvalues, eigenvectors = scipy.sparse.linalg.lobpcg(
                L, random_state.normal(size=(L.shape[0], k)), largest=False, tol=1e-8
            )
        order = np.argsort(eigenvalues)[1:k]
        return eigenvectors[:, order]
    except scipy.sparse.linalg.ArpackError:
        warn(
            "WARNING: spectral initialisation failed! The eigenvector solver\n"
            "failed. This is likely due to too small an eigengap. Consider\n"
            "adding some noise or jitter to your data.\n\n"
            "Falling back to random initialisation!"
        )
        return random_state.uniform(low=-10.0, high=10.0, size=(graph.shape[0], dim))
