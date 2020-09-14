from typing import Union, Optional

import numpy as np
from numpy import random
from sklearn.base import TransformerMixin
import networkx as nx
AnyRandom = Union[None, int, random.RandomState]  # maybe in the future random.Generator


#from .. import _utils
#from .. import logging as logg
#from .utils import get_init_pos_from_paga
from .graph_utils import compute_connectivities_adapmap
from .spectral import spectral_layout

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

_LAYOUTS = ('fr', 'drl', 'kk', 'grid_fr', 'lgl', 'rt', 'rt_circular', 'fa')
_Layout = Literal[_LAYOUTS]


class force_directed_layout(TransformerMixin):
    """\
        Force-directed graph drawing [Islam11]_ [Jacomy14]_ [Chippada18]_.
        An alternative to tSNE that often preserves the topology of the data
        better. This requires to run :func:`~scanpy.pp.neighbors`, first.
        The default layout ('fa', `ForceAtlas2`) [Jacomy14]_ uses the package |fa2|_
        [Chippada18]_, which can be installed via `pip install fa2`.
        `Force-directed graph drawing`_ describes a class of long-established
        algorithms for visualizing graphs.
        It has been suggested for visualizing single-cell data by [Islam11]_.
        Many other layouts as implemented in igraph [Csardi06]_ are available.
        Similar approaches have been used by [Zunder15]_ or [Weinreb17]_.
        .. |fa2| replace:: `fa2`
        .. _fa2: https://github.com/bhargavchippada/forceatlas2
        .. _Force-directed graph drawing: https://en.wikipedia.org/wiki/Force-directed_graph_drawing
        Parameters
        ----------
        data
            Data matrix. Accepts numpy arrays and csr matrices.
        layout
            'fa' (`ForceAtlas2`) or any valid `igraph layout
            <http://igraph.org/c/doc/igraph-Layout.html>`__. Of particular interest
            are 'fr' (Fruchterman Reingold), 'grid_fr' (Grid Fruchterman Reingold,
            faster than 'fr'), 'kk' (Kamadi Kawai', slower than 'fr'), 'lgl' (Large
            Graph, very fast), 'drl' (Distributed Recursive Layout, pretty fast) and
            'rt' (Reingold Tilford tree layout).
        root
            Root for tree layouts.
        random_state
            For layouts with random initialization like 'fr', change this to use
            different intial states for the optimization. If `None`, no seed is set.
        proceed
            Continue computation, starting off with 'X_draw_graph_`layout`'.
        init_pos
            `'paga'`/`True`, `None`/`False`, or any valid 2d-`.obsm` key.
            Use precomputed coordinates for initialization.
            If `False`/`None` (the default), initialize randomly.

        **kwds
            Parameters of chosen igraph layout. See e.g. `fruchterman-reingold`_
            [Fruchterman91]_. One of the most important ones is `maxiter`.
            .. _fruchterman-reingold: http://igraph.org/python/doc/igraph.Graph-class.html#layout_fruchterman_reingold
        Returns
        -------
        Depending on `copy`, returns or updates `adata` with the following field.
        **X_draw_graph_layout** : `adata.obsm`
            Coordinates of graph layout. E.g. for layout='fa' (the default),
            the field is called 'X_draw_graph_fa'
        """
    def __init__(self,
                 layout='fa',
                 init_pos=None,
                 use_paga=False,
                 root=None,
                 random_state = 0,
                 n_jobs = 10,
                 **kwds
                 ):
        self.layout = layout
        self.init_pos = init_pos
        self.use_paga = use_paga
        self.root = root
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kwds = kwds

    def fit(self, data):

        start = print(f'drawing graph using layout {self.layout!r}')
        if self.layout not in _LAYOUTS:
            raise ValueError(f'Provide a valid layout, one of {_LAYOUTS}.')


        # init coordinates
        if self.init_pos is not None:
            self.init_coords = self.init_pos

        #elif (self.use_paga == True):
            # TODO: add util function to get initial coordinates from a PAGA coarsed graph
            # init_coords = get_init_pos_from_paga()

        if (self.init_pos is None) :
            self.distances, self.connectivities = compute_connectivities_adapmap(
                    data,
                    n_components=100,
                    n_neighbors=30,
                    alpha=0.5,
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

                )
            np.random.seed(self.random_state)

        initialisation = spectral_layout(
            self.connectivities,
            self.distances,
            dim=2,
            random_state=self.random_state)

        self.init_coords = np.random.random((self.connectivities.shape[0], 2))

        return self

    def transform(self, X, y=None, **fit_params):
        # see whether fa2 is installed
        self.G = nx.random_geometric_graph(400, 0.2)
        if self.layout == 'fa':
            try:
                from fa2 import ForceAtlas2
            except ImportError:
                logg.warning(
                    "Package 'fa2' is not installed, falling back to layout 'lgl'."
                    "To use the faster and better ForceAtlas2 layout, "
                    "install package 'fa2' (`pip install fa2`)."
                )
                self.layout = 'fr'
        # actual drawing
        if self.layout == 'fa':
            forceatlas2 = ForceAtlas2(
                # Behavior alternatives
                outboundAttractionDistribution=False,  # Dissuade hubs
                linLogMode=False,  # NOT IMPLEMENTED
                adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                edgeWeightInfluence=1.0,
                # Performance
                jitterTolerance=1.0,  # Tolerance
                barnesHutOptimize=True,
                barnesHutTheta=1.2,
                multiThreaded=False,  # NOT IMPLEMENTED
                # Tuning
                scalingRatio=2.0,
                strongGravityMode=False,
                gravity=1.0,
                # Log
                verbose=False,
            )
            if 'maxiter' in self.kwds:
                iterations = self.kwds['maxiter']
            elif 'iterations' in self.kwds:
                iterations = self.kwds['iterations']
            else:
                iterations = 500
            positions = forceatlas2.forceatlas2(
                self.connectivities, pos=self.init_coords, iterations=iterations
            )
            positions = np.array(positions)
        else:
            # igraph doesn't use numpy seed
            random.seed(self.random_state)

            self.G = graph_utils.get_igraph_from_adjacency(self.connectivities)
            if self.layout in {'fr', 'drl', 'kk', 'grid_fr'}:
                ig_layout =  self.G.layout(self.layout, seed=self.init_coords.tolist(), **self.kwds)
            elif 'rt' in self.layout:
                if self.root is not None:
                    self.root = [ self.root]
                ig_layout =  self.G.layout(self.layout, root=self.root, **self.kwds)
            else:
                ig_layout =  self.G.layout(self.layout, **self.kwds)

        self.positions = np.array(ig_layout.coords)


    def plot_graph(self, node_size=20, with_labels=False, node_color="blue", node_alpha=0.4, plot_edges=True,
                   edge_color="green", edge_alpha=0.05):
        import matplotlib.pyplot as plt
        nx.draw_networkx_nodes(self.G, self.positions, node_size=20, with_labels=False, node_color="blue",
                               alpha=node_alpha)
        if plot_edges:
            nx.draw_networkx_edges(self.G, self.positions, edge_color="green", alpha=edge_alpha)
        plt.axis('off')
        plt.show()
