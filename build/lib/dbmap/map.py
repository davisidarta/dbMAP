from typing import Optional, Union
import warnings
import time
import numpy as np
from packaging import version
from sklearn.utils import check_random_state, check_array

AnyRandom = Union[None, int, np.random.RandomState]

try:
    from typing import Literal
except ImportError:
    try:
        from typing_extensions import Literal
    except ImportError:

        class LiteralMeta(type):
            def __getitem__(cls, values):
                if not isinstance(values, tuple):
                    values = (values,)
                return type('Literal_', (Literal,), dict(__args__=values))


        class Literal(metaclass=LiteralMeta):
            pass

from .graph_utils import simplicial_set_embedding, find_ab_params


def fuzzy_embedding(graph,
                    n_components=2,
                    initial_alpha=1,
                    min_dist=0.6,
                    spread=1.2,
                    n_epochs=500,
                    metric='cosine',
                    metric_kwds=None,
                    output_metric='euclidean',
                    output_metric_kwds=None,
                    gamma=1,
                    negative_sample_rate=5,
                    init='diffuse_spectral',
                    random_state=None,
                    euclidean_output=True,
                    parallel=True,
                    njobs=-1,
                    verbose=False,
                    a=None,
                    b=None,
                    densmap=False,
                    densmap_kwds=None,
                    output_dens=False
                    ):
    """\
    Perform a fuzzy simplicial set embedding, using a specified
    initialisation method and then minimizing the fuzzy set cross entropy
    between the 1-skeletons of the high and low dimensional fuzzy simplicial
    sets. The fuzzy simplicial set embedding was proposed and implemented by
    Leland McInnes on UMAP (see `umap-learn <https://github.com/lmcinnes/umap>`).


    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data to be embedded by UMAP.
    graph: sparse matrix
        The 1-skeleton of the high dimensional fuzzy simplicial set as
        represented by a graph for which we require a sparse matrix for the
        (weighted) adjacency matrix.
    n_components: int
        The dimensionality of the euclidean space into which to embed the data.
    initial_alpha: float
        Initial learning rate for the SGD.
    a: float
        Parameter of differentiable approximation of right adjoint functor
    b: float
        Parameter of differentiable approximation of right adjoint functor
    gamma: float
        Weight to apply to negative samples.
    negative_sample_rate: int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.
    n_epochs: int (optional, default 0)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If 0 is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).
    init: string
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.
    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.
    metric: string or callable
        The metric used to measure distance in high dimensional space; used if
        multiple connected components need to be layed out.
    metric_kwds: dict
        Key word arguments to be passed to the metric function; used if
        multiple connected components need to be layed out.
    densmap: bool
        Whether to use the density-augmented objective function to optimize
        the embedding according to the densMAP algorithm.
    densmap_kwds: dict
        Key word arguments to be used by the densMAP optimization.
    output_dens: bool
        Whether to output local radii in the original data and the embedding.
    output_metric: function
        Function returning the distance between two points in embedding space and
        the gradient of the distance wrt the first argument.
    output_metric_kwds: dict
        Key word arguments to be passed to the output_metric function.
    euclidean_output: bool
        Whether to use the faster code specialised for euclidean output metrics
    parallel: bool (optional, default False)
        Whether to run the computation using numba parallel.
        Running in parallel is non-deterministic, and is not used
        if a random seed has been set, to ensure reproducibility.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized of ``graph`` into an ``n_components`` dimensional
        euclidean space.
    aux_data: dict
        Auxiliary output returned with the embedding. When densMAP extension
        is turned on, this dictionary includes local radii in the original
        data (``rad_orig``) and in the embedding (``rad_emb``).
    """
    if metric_kwds is None:
        _metric_kwds = {}
    else:
        _metric_kwds = metric_kwds
    if output_metric_kwds is None:
        _output_metric_kwds = {}

    # Compat for umap 0.4 -> 0.5
    if a is None or b is None:
        a, b = find_ab_params(spread, min_dist)
    else:
        a = a
        b = b
        # the data matrix X is really only used for determining the number of connected components
        # for the init condition in the UMAP embedding (high-resolution spectral layout)
    start_time = time.time()
    X_map = simplicial_set_embedding(data,
                                      graph,
                                      n_components,
                                      initial_alpha,
                                      a,
                                      b,
                                      gamma,
                                      negative_sample_rate,
                                      n_epochs,
                                      init,
                                      random_state,
                                      metric,
                                      metric_kwds,
                                      densmap,
                                      densmap_kwds,
                                      output_dens,
                                      output_metric,
                                      output_metric_kwds,
                                      euclidean_output,
                                      parallel,
                                      verbose)
    end_time = time.time()
    if verbose:
        print('Layout optimization time = %f (sec), '
              'adjusted for thread number=%f (sec)' %
              (end_time - start_time,
               njobs * float(end_time - start_time)))
    return X_map
