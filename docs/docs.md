# Module dbMAP
    
## Sub-modules

* [dbmap.ann](#dbmap.ann)
* [dbmap.diffusion](#dbmap.diffusion)
* [dbmap.graph_utils](#dbmap.graph_utils)
* [dbmap.layout](#dbmap.layout)
* [dbmap.map](#dbmap.map)
* [dbmap.multiscale](#dbmap.multiscale)
* [dbmap.plot](#dbmap.plot)
* [dbmap.spectral](#dbmap.spectral)
* [dbmap.umapper](#dbmap.umapper)
* [dbmap.utils](#dbmap.utils)



   
# Approximate Nearest Neighbors {#dbmap.ann}

    
### Class `NMSlibTransformer` {#dbmap.ann.NMSlibTransformer}




>     class NMSlibTransformer(
>         n_neighbors=30,
>         metric='cosine',
>         method='hnsw',
>         n_jobs=10,
>         p=None,
>         M=30,
>         efC=100,
>         efS=100,
>         dense=False,
>         verbose=False
>     )


Wrapper for using nmslib as sklearn's KNeighborsTransformer. This implements
an escalable approximate k-nearest-neighbors graph on spaces defined by nmslib.
Read more about nmslib and its various available metrics at
<https://github.com/nmslib/nmslib.>
Calling 'nn <- NMSlibTransformer()' initializes the class with
 neighbour search parameters.
#### Parameters

**```n_neighbors```** :&ensp;<code>int (optional</code>, default <code>30)</code>
:   number of nearest-neighbors to look for. In practice,
    this should be considered the average neighborhood size and thus vary depending
    on your number of features, samples and data intrinsic dimensionality. Reasonable values
    range from 5 to 100. Smaller values tend to lead to increased graph structure
    resolution, but users should beware that a too low value may render granulated and vaguely
    defined neighborhoods that arise as an artifact of downsampling. Defaults to 30. Larger
    values can slightly increase computational time.


**```metric```** :&ensp;<code>str (optional</code>, default `'cosine')`
:   accepted NMSLIB metrics. Defaults to 'cosine'. Accepted metrics include:
    -'sqeuclidean'
    -'euclidean'
    -'l1'
    -'lp' - requires setting the parameter <code>p</code>
    -'cosine'
    -'angular'
    -'negdotprod'
    -'levenshtein'
    -'hamming'
    -'jaccard'
    -'jansen-shan'


**```method```** :&ensp;<code>str (optional</code>, default `'hsnw')`
:   approximate-neighbor search method. Available methods include:
            -'hnsw' : a Hierarchical Navigable Small World Graph.
            -'sw-graph' : a Small World Graph.
            -'vp-tree' : a Vantage-Point tree with a pruning rule adaptable to non-metric distances.
            -'napp' : a Neighborhood APProximation index.
            -'simple_invindx' : a vanilla, uncompressed, inverted index, which has no parameters.
            -'brute_force' : a brute-force search, which has no parameters.
    'hnsw' is usually the fastest method, followed by 'sw-graph' and 'vp-tree'.


**```n_jobs```** :&ensp;<code>int (optional</code>, default <code>1)</code>
:   number of threads to be used in computation. Defaults to 1. The algorithm is highly
    scalable to multi-threading.


**```M```** :&ensp;<code>int (optional</code>, default <code>30)</code>
:   defines the maximum number of neighbors in the zero and above-zero layers during HSNW
    (Hierarchical Navigable Small World Graph). However, the actual default maximum number
    of neighbors for the zero layer is 2*M.  A reasonable range for this parameter
    is 5-100. For more information on HSNW, please check <https://arxiv.org/abs/1603.09320.>
    HSNW is implemented in python via NMSlib. Please check more about NMSlib at <https://github.com/nmslib/nmslib.>


**```efC```** :&ensp;<code>int (optional</code>, default <code>100)</code>
:   A 'hnsw' parameter. Increasing this value improves the quality of a constructed graph
    and leads to higher accuracy of search. However this also leads to longer indexing times.
    A reasonable range for this parameter is 50-2000.


**```efS```** :&ensp;<code>int (optional</code>, default <code>100)</code>
:   A 'hnsw' parameter. Similarly to efC, increasing this value improves recall at the
    expense of longer retrieval time. A reasonable range for this parameter is 100-2000.


**```dense```** :&ensp;<code>bool (optional</code>, default <code>False)</code>
:   Whether to force the algorithm to use dense data, such as np.ndarrays and pandas DataFrames.

#### Returns

Class for really fast approximate-nearest-neighbors search.
#### Example
```
import numpy as np
from sklearn.datasets import load_digits
from scipy.sparse import csr_matrix
from dbmap.ann import NMSlibTransformer
#
### Load the MNIST digits data, convert to sparse for speed
digits = load_digits()
data = csr_matrix(digits)
#
### Start class with parameters
nn = NMSlibTransformer()
nn = nn.fit(data)
#
### Obtain kNN graph
knn = nn.transform(data)
#
### Obtain kNN indices, distances and distance gradient
ind, dist, grad = nn.ind_dist_grad(data)
#
### Test for recall efficiency during approximate nearest neighbors search
test = nn.test_efficiency(data)
```

    
#### Ancestors (in MRO)

* [sklearn.base.TransformerMixin](#sklearn.base.TransformerMixin)
* [sklearn.base.BaseEstimator](#sklearn.base.BaseEstimator)






    
#### Methods


    
##### Method `fit` {#dbmap.ann.NMSlibTransformer.fit}




>     def fit(
>         self,
>         data
>     )




    
##### Method `ind_dist_grad` {#dbmap.ann.NMSlibTransformer.ind_dist_grad}




>     def ind_dist_grad(
>         self,
>         data,
>         return_grad=True,
>         return_graph=True
>     )




    
##### Method `test_efficiency` {#dbmap.ann.NMSlibTransformer.test_efficiency}




>     def test_efficiency(
>         self,
>         data,
>         data_use=0.1
>     )


Test if NMSlibTransformer and KNeighborsTransformer give same results

    
##### Method `transform` {#dbmap.ann.NMSlibTransformer.transform}




>     def transform(
>         self,
>         data
>     )




    
##### Method `update_search` {#dbmap.ann.NMSlibTransformer.update_search}




>     def update_search(
>         self,
>         n_neighbors
>     )


Updates number of neighbors for kNN distance computation.
###### Parameters

n_neighbors: New number of neighbors to look for.

    
# Diffusion harmonics {#dbmap.diffusion}




    
### Class `Diffusor` {#dbmap.diffusion.Diffusor}




>     class Diffusor(
>         n_components=50,
>         n_neighbors=10,
>         alpha=1,
>         n_jobs=10,
>         ann=True,
>         ann_dist='cosine',
>         p=None,
>         M=30,
>         efC=100,
>         efS=100,
>         knn_dist='cosine',
>         kernel_use='decay',
>         transitions=True,
>         eigengap=True,
>         norm=False,
>         verbose=True
>     )


Sklearn estimator for using fast anisotropic diffusion with a multiscaling
adaptive algorithm as proposed by Setty et al, 2018, and optimized by Sidarta-Oliveira, 2020.

#### Parameters

**```n_components```** :&ensp;<code>Number</code> of <code>diffusion components to compute. Defaults to 100. We suggest larger values if</code>
:   analyzing more than 10,000 cells.


**```n_neighbors```** :&ensp;<code>Number</code> of `k-nearest-neighbors to compute. The adaptive kernel will normalize distances by each cell`
:   distance of its median neighbor.


**```knn_dist```** :&ensp;`Distance metric for building kNN graph. Defaults to 'euclidean'. Users are encouraged to explore`
:   different metrics, such as 'cosine' and 'jaccard'. The 'hamming' and 'jaccard' distances are also available
           for string vectors.


ann : Boolean. Whether to use approximate nearest neighbors for graph construction. Defaults to True.

alpha : Alpha in the diffusion maps literature. Controls how much the results are biased by data distribution.
        Defaults to 1, which is suitable for normalized data.

n_jobs : Number of threads to use in calculations. Defaults to all but one.

verbose : controls verbosity.


#### Returns

    Diffusion components ['EigenVectors'], associated eigenvalues ['EigenValues'] and suggested number of
    resulting components to use during Multiscaling.

#### Example
```

import numpy as np
from sklearn.datasets import load_digits
from scipy.sparse import csr_matrix
import dbmap

### Load the MNIST digits data, convert to sparse for speed
digits = load_digits()
data = csr_matrix(digits)

### Fit the anisotropic diffusion process
diff = dbmap.diffusion.Diffusor()
res = diff.fit_transform(data)

```
    
#### Ancestors (in MRO)

* [sklearn.base.TransformerMixin](#sklearn.base.TransformerMixin)






    
#### Methods


    
##### Method `fit` {#dbmap.diffusion.Diffusor.fit}




>     def fit(
>         self,
>         data
>     )


Fits an adaptive anisotropic kernel to the data.
:param data: input data. Takes in numpy arrays and scipy csr sparse matrices.
Use with sparse data for top performance. You can adjust a series of
parameters that can make the process faster and more informational depending
on your dataset. Read more at <https://github.com/davisidarta/dbmap>

    
##### Method `ind_dist_grad` {#dbmap.diffusion.Diffusor.ind_dist_grad}




>     def ind_dist_grad(
>         self,
>         data,
>         n_components=None,
>         dense=False
>     )


Effectively computes on data. Also returns the normalized diffusion distances,
indexes and gradient obtained by approximating the Laplace-Beltrami operator.
:param plot_knee: Whether to plot the scree plot of diffusion eigenvalues.
:param data: input data. Takes in numpy arrays and scipy csr sparse matrices.
Please use with sparse data for top performance. You can adjust a series of
parameters that can make the process faster and more informational depending
on your dataset. Read more at <https://github.com/davisidarta/dbmap>

    
##### Method `return_dict` {#dbmap.diffusion.Diffusor.return_dict}




>     def return_dict(
>         self
>     )


:return: Dictionary containing normalized and multiscaled Diffusion Components
(['StructureComponents']), their eigenvalues ['EigenValues'], non-normalized
components (['EigenVectors']) and the kernel used for transformation of distances
into affinities (['kernel']).

    
##### Method `transform` {#dbmap.diffusion.Diffusor.transform}




>     def transform(
>         self,
>         data
>     )






    
# Graph utilities {#dbmap.graph_utils}



    
### Function `approximate_n_neighbors` {#dbmap.graph_utils.approximate_n_neighbors}




>     def approximate_n_neighbors(
>         data,
>         n_neighbors=30,
>         metric='cosine',
>         method='hnsw',
>         n_jobs=10,
>         efC=100,
>         efS=100,
>         M=30,
>         dense=False,
>         verbose=False
>     )


Simple function using NMSlibTransformer from dbmap.ann. This implements a very fast
and scalable approximate k-nearest-neighbors graph on spaces defined by nmslib.
Read more about nmslib and its various available metrics at
<https://github.com/nmslib/nmslib.> Read more about dbMAP at
<https://github.com/davisidarta/dbMAP.>


###### Parameters

**```n_neighbors```** :&ensp;<code>number</code> of `nearest-neighbors to look for. In practice,`
:   this should be considered the average neighborhood size and thus vary depending
                 on your number of features, samples and data intrinsic dimensionality. Reasonable values
                 range from 5 to 100. Smaller values tend to lead to increased graph structure
                 resolution, but users should beware that a too low value may render granulated and vaguely
                 defined neighborhoods that arise as an artifact of downsampling. Defaults to 30. Larger
                 values can slightly increase computational time.


**```metric```** :&ensp;`accepted NMSLIB metrics. Defaults to 'cosine'. Accepted metrics include:`
:   -'sqeuclidean'
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

**```efC```** :&ensp;<code>increasing this value improves the quality</code> of <code>a constructed graph and leads to higher</code>
:   accuracy of search. However this also leads to longer indexing times. A reasonable
         range is 100-2000. Defaults to 100.


**```efS```** :&ensp;<code>similarly to efC, improving this value improves recall at the expense</code> of <code>longer</code>
:   retrieval time. A reasonable range is 100-2000.


**```M```** :&ensp;<code>defines the maximum number</code> of `neighbors in the zero and above-zero layers during HSNW`
:   (Hierarchical Navigable Small World Graph). However, the actual default maximum number
       of neighbors for the zero layer is 2*M. For more information on HSNW, please check
       <https://arxiv.org/abs/1603.09320.> HSNW is implemented in python via NMSLIB. Please check
       more about NMSLIB at <https://github.com/nmslib/nmslib> .


:returns: k-nearest-neighbors indices and distances. Can be customized to also return
    return the k-nearest-neighbors graph and its gradient.

###### Example

```
knn_indices, knn_dists = approximate_n_neighbors(data)
```
    
### Function `compute_connectivities_adapmap` {#dbmap.graph_utils.compute_connectivities_adapmap}




>     def compute_connectivities_adapmap(
>         data,
>         n_components=100,
>         n_neighbors=30,
>         alpha=0.0,
>         n_jobs=10,
>         ann=True,
>         ann_dist='cosine',
>         M=30,
>         efC=100,
>         efS=100,
>         knn_dist='euclidean',
>         kernel_use='sidarta',
>         sensitivity=1,
>         set_op_mix_ratio=1.0,
>         local_connectivity=1.0
>     )


Sklearn estimator for using fast anisotropic diffusion with an anisotropic
    adaptive algorithm as proposed by Setty et al, 2018, and optimized by Sidarta-Oliveira, 2020.
    This procedure generates diffusion components that effectivelly carry the maximum amount of
    information regarding the data geometric structure (structure components).
    These structure components then undergo a fuzzy-union of simplicial sets. This step is
    from umap.fuzzy_simplicial_set [McInnes18]_. Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.

###### Parameters

**```n_components```** :&ensp;<code>Number</code> of <code>diffusion components to compute. Defaults to 100. We suggest larger values if</code>
:   analyzing more than 10,000 cells.


**```n_neighbors```** :&ensp;<code>Number</code> of `k-nearest-neighbors to compute. The adaptive kernel will normalize distances by each cell`
:   distance of its median neighbor.


**```knn_dist```** :&ensp;`Distance metric for building kNN graph. Defaults to 'euclidean'. `
:   &nbsp;


ann : Boolean. Whether to use approximate nearest neighbors for graph construction. Defaults to True.

alpha : Alpha in the diffusion maps literature. Controls how much the results are biased by data distribution.
        Defaults to 1, which is suitable for normalized data.

n_jobs : Number of threads to use in calculations. Defaults to all but one.

**```sensitivity```** :&ensp;`Sensitivity to select eigenvectors if diff_normalization is set to 'knee'. Useful when dealing wit`
:   &nbsp;


:returns: Diffusion components ['EigenVectors'], associated eigenvalues ['EigenValues'] and suggested number of
         resulting components to use during Multiscaling.

###### Example

```
import numpy as np
from sklearn.datasets import load_digits
from scipy.sparse import csr_matrix
import dbmap

##### Load the MNIST digits data, convert to sparse for speed
digits = load_digits()
data = csr_matrix(digits)

##### Fit the anisotropic diffusion process
diff = dbmap.diffusion.Diffusor()
res = diff.fit_transform(data)
```
    
### Function `compute_membership_strengths` {#dbmap.graph_utils.compute_membership_strengths}




>     def compute_membership_strengths(
>         knn_indices,
>         knn_dists,
>         sigmas,
>         rhos
>     )


Construct the membership strength data for the 1-skeleton of each local
fuzzy simplicial set -- this is formed as a sparse matrix where each row is
a local fuzzy simplicial set, with a membership strength for the
1-simplex to each other data point.
###### Parameters

**```knn_indices```** :&ensp;<code>array</code> of <code>shape (n\_samples, n\_neighbors)</code>
:   The indices on the <code>n\_neighbors</code> closest points in the dataset.


**```knn_dists```** :&ensp;<code>array</code> of <code>shape (n\_samples, n\_neighbors)</code>
:   The distances to the <code>n\_neighbors</code> closest points in the dataset.


**```sigmas```** :&ensp;<code>array</code> of <code>shape(n\_samples)</code>
:   The normalization factor derived from the metric tensor approximation.


**```rhos```** :&ensp;<code>array</code> of <code>shape(n\_samples)</code>
:   The local connectivity adjustment.

###### Returns

**```rows```** :&ensp;<code>array</code> of `shape (n_samples * n_neighbors)`
:   Row data for the resulting sparse matrix (coo format)


**```cols```** :&ensp;<code>array</code> of `shape (n_samples * n_neighbors)`
:   Column data for the resulting sparse matrix (coo format)


**```vals```** :&ensp;<code>array</code> of `shape (n_samples * n_neighbors)`
:   Entries for the resulting sparse matrix (coo format)



    
### Function `fuzzy_simplicial_set_nmslib` {#dbmap.graph_utils.fuzzy_simplicial_set_nmslib}




>     def fuzzy_simplicial_set_nmslib(
>         X,
>         n_neighbors,
>         knn_indices=None,
>         knn_dists=None,
>         nmslib_metric='cosine',
>         nmslib_n_jobs=None,
>         nmslib_efC=100,
>         nmslib_efS=100,
>         nmslib_M=30,
>         set_op_mix_ratio=1.0,
>         local_connectivity=1.0,
>         apply_set_operations=True,
>         verbose=False
>     )


Given a set of data X, a neighborhood size, and a measure of distance
compute the fuzzy simplicial set (here represented as a fuzzy graph in
the form of a sparse matrix) associated to the data. This is done by
locally approximating geodesic distance at each point, creating a fuzzy
simplicial set for each such point, and then combining all the local
fuzzy simplicial sets into a global one via a fuzzy union.
###### Parameters

**```X```** :&ensp;<code>array</code> of <code>shape (n\_samples, n\_features)</code>
:   The data to be modelled as a fuzzy simplicial set.


**```n_neighbors```** :&ensp;<code>int</code>
:   The number of neighbors to use to approximate geodesic distance.
    Larger numbers induce more global estimates of the manifold that can
    miss finer detail, while smaller values will focus on fine manifold
    structure to the detriment of the larger picture.


**```nmslib_metric```** :&ensp;<code>str (optional</code>, default `'cosine')`
:   accepted NMSLIB metrics. Accepted metrics include:
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


**```nmslib_n_jobs```** :&ensp;<code>int (optional</code>, default <code>None)</code>
:   Number of threads to use for approximate-nearest neighbor search.


**```nmslib_efC```** :&ensp;<code>int (optional</code>, default <code>100)</code>
:   increasing this value improves the quality of a constructed graph and leads to higher
    accuracy of search. However this also leads to longer indexing times. A reasonable
    range is 100-2000.


**```nmslib_efS```** :&ensp;<code>int (optional</code>, default <code>100)</code>
:   similarly to efC, improving this value improves recall at the expense of longer
    retrieval time. A reasonable range is 100-2000.


nmslib_M: int (optional, default 30).
    defines the maximum number of neighbors in the zero and above-zero layers during HSNW
    (Hierarchical Navigable Small World Graph). However, the actual default maximum number
    of neighbors for the zero layer is 2*M. For more information on HSNW, please check
    <https://arxiv.org/abs/1603.09320.> HSNW is implemented in python via NMSLIB. Please check
    more about NMSLIB at <https://github.com/nmslib/nmslib> .    n_epochs: int (optional, default None)
    The number of training epochs to be used in optimizing the
    low dimensional embedding. Larger values result in more accurate
    embeddings. If None is specified a value will be selected based on
    the size of the input dataset (200 for large datasets, 500 for small).
**```knn_indices```** :&ensp;<code>array</code> of <code>shape (n\_samples, n\_neighbors) (optional)</code>
:   If the k-nearest neighbors of each point has already been calculated
    you can pass them in here to save computation time. This should be
    an array with the indices of the k-nearest neighbors as a row for
    each data point.


**```knn_dists```** :&ensp;<code>array</code> of <code>shape (n\_samples, n\_neighbors) (optional)</code>
:   If the k-nearest neighbors of each point has already been calculated
    you can pass them in here to save computation time. This should be
    an array with the distances of the k-nearest neighbors as a row for
    each data point.


**```set_op_mix_ratio```** :&ensp;<code>float (optional</code>, default <code>1.0)</code>
:   Interpolate between (fuzzy) union and intersection as the set operation
    used to combine local fuzzy simplicial sets to obtain a global fuzzy
    simplicial sets. Both fuzzy set operations use the product t-norm.
    The value of this parameter should be between 0.0 and 1.0; a value of
    1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
    intersection.


**```local_connectivity```** :&ensp;<code>int (optional</code>, default <code>1)</code>
:   The local connectivity required -- i.e. the number of nearest
    neighbors that should be assumed to be connected at a local level.
    The higher this value the more connected the manifold becomes
    locally. In practice this should be not more than the local intrinsic
    dimension of the manifold.


**```verbose```** :&ensp;<code>bool (optional</code>, default <code>False)</code>
:   Whether to report information on the current progress of the algorithm.

###### Returns

**```fuzzy_simplicial_set```** :&ensp;<code>coo\_matrix</code>
:   A fuzzy simplicial set represented as a sparse matrix. The (i,
    j) entry of the matrix represents the membership strength of the
    1-simplex between the ith and jth sample points.



    
### Function `get_igraph_from_adjacency` {#dbmap.graph_utils.get_igraph_from_adjacency}




>     def get_igraph_from_adjacency(
>         adjacency,
>         directed=None
>     )


Get igraph graph from adjacency matrix.

    
### Function `get_sparse_matrix_from_indices_distances_dbmap` {#dbmap.graph_utils.get_sparse_matrix_from_indices_distances_dbmap}




>     def get_sparse_matrix_from_indices_distances_dbmap(
>         knn_indices,
>         knn_dists,
>         n_obs,
>         n_neighbors
>     )




    
### Function `smooth_knn_dist` {#dbmap.graph_utils.smooth_knn_dist}




>     def smooth_knn_dist(
>         distances,
>         k,
>         n_iter=64,
>         local_connectivity=1.0,
>         bandwidth=1.0
>     )


Compute a continuous version of the distance to the kth nearest
neighbor. That is, this is similar to knn-distance but allows continuous
k values rather than requiring an integral k. In essence we are simply
computing the distance such that the cardinality of fuzzy set we generate
is k.
###### Parameters

**```distances```** :&ensp;<code>array</code> of <code>shape (n\_samples, n\_neighbors)</code>
:   Distances to nearest neighbors for each samples. Each row should be a
    sorted list of distances to a given samples nearest neighbors.


**```k```** :&ensp;<code>float</code>
:   The number of nearest neighbors to approximate for.


**```n_iter```** :&ensp;<code>int (optional</code>, default <code>64)</code>
:   We need to binary search for the correct distance value. This is the
    max number of iterations to use in such a search.


**```local_connectivity```** :&ensp;<code>int (optional</code>, default <code>1)</code>
:   The local connectivity required -- i.e. the number of nearest
    neighbors that should be assumed to be connected at a local level.
    The higher this value the more connected the manifold becomes
    locally. In practice this should be not more than the local intrinsic
    dimension of the manifold.


**```bandwidth```** :&ensp;<code>float (optional</code>, default <code>1)</code>
:   The target bandwidth of the kernel, larger values will produce
    larger return values.

###### Returns

**```knn_dist```** :&ensp;<code>array</code> of <code>shape (n\_samples,)</code>
:   The distance to kth nearest neighbor, as suitably approximated.


**```nn_dist```** :&ensp;<code>array</code> of <code>shape (n\_samples,)</code>
:   The distance to the 1st nearest neighbor for each point.






    
# Graph layout {#dbmap.layout} 


  
### Class `force_directed_layout` {#dbmap.layout.force_directed_layout}




>     class force_directed_layout(
>         layout='fa',
>         init_pos=None,
>         use_paga=False,
>         root=None,
>         random_state=0,
>         n_jobs=10,
>         **kwds
>     )


Force-directed graph drawing [Islam11]_ [Jacomy14]_ [Chippada18]_.
An alternative to tSNE that often preserves the topology of the data
better. This requires to run :func:`~scanpy.pp.neighbors`, first.
The default layout ('fa', <code>ForceAtlas2</code>) [Jacomy14]_ uses the package |fa2|_
[Chippada18]_, which can be installed via <code>pip install fa2</code>.
`Force-directed graph drawing`_ describes a class of long-established
algorithms for visualizing graphs.
It has been suggested for visualizing single-cell data by [Islam11]_.
Many other layouts as implemented in igraph [Csardi06]_ are available.
Similar approaches have been used by [Zunder15]_ or [Weinreb17]_.
.. |fa2| replace:: <code>fa2</code>
.. _fa2: <https://github.com/bhargavchippada/forceatlas2>
.. _Force-directed graph drawing: <https://en.wikipedia.org/wiki/Force-directed_graph_drawing>
#### Parameters

**```data```**
:   Data matrix. Accepts numpy arrays and csr matrices.


**```layout```**
:   'fa' (<code>ForceAtlas2</code>) or any valid `igraph layout
    <http://igraph.org/c/doc/igraph-Layout.html>`__. Of particular interest
    are 'fr' (Fruchterman Reingold), 'grid_fr' (Grid Fruchterman Reingold,
    faster than 'fr'), 'kk' (Kamadi Kawai', slower than 'fr'), 'lgl' (Large
    Graph, very fast), 'drl' (Distributed Recursive Layout, pretty fast) and
    'rt' (Reingold Tilford tree layout).


**```root```**
:   Root for tree layouts.


**```random_state```**
:   For layouts with random initialization like 'fr', change this to use
    different intial states for the optimization. If <code>None</code>, no seed is set.


**```proceed```**
:   Continue computation, starting off with 'X_draw_graph_<code>layout</code>'.


**```init_pos```**
:   `'paga'`/<code>True</code>, <code>None</code>/<code>False</code>, or any valid 2d-<code>.obsm</code> key.
    Use precomputed coordinates for initialization.
    If <code>False</code>/<code>None</code> (the default), initialize randomly.


**```**kwds```**
:   Parameters of chosen igraph layout. See e.g. `fruchterman-reingold`_
    [Fruchterman91]_. One of the most important ones is <code>maxiter</code>.
    .. _fruchterman-reingold: <http://igraph.org/python/doc/igraph.Graph-class.html#layout_fruchterman_reingold>

#### Returns

Depending on <code>copy</code>, returns or updates <code>adata</code> with the following field.
**X_draw_graph_layout** : <code>adata.obsm</code>
    Coordinates of graph layout. E.g. for layout='fa' (the default),
    the field is called 'X_draw_graph_fa'


    
#### Ancestors (in MRO)

* [sklearn.base.TransformerMixin](#sklearn.base.TransformerMixin)






    
#### Methods


    
##### Method `fit` {#dbmap.layout.force_directed_layout.fit}




>     def fit(
>         self,
>         data
>     )




    
##### Method `plot_graph` {#dbmap.layout.force_directed_layout.plot_graph}




>     def plot_graph(
>         self,
>         node_size=20,
>         with_labels=False,
>         node_color='blue',
>         node_alpha=0.4,
>         plot_edges=True,
>         edge_color='green',
>         edge_alpha=0.05
>     )




    
##### Method `transform` {#dbmap.layout.force_directed_layout.transform}




>     def transform(
>         self,
>         X,
>         y=None,
>         **fit_params
>     )






    
# Mapping - UMAP/TriMaps {#dbmap.map}





    
### Class `Mapper` {#dbmap.map.Mapper}




>     class Mapper(
>         n_components=2,
>         n_neighbors=15,
>         metric='euclidean',
>         output_metric='euclidean',
>         n_epochs=None,
>         learning_rate=1.5,
>         init='spectral',
>         min_dist=0.6,
>         spread=1.5,
>         low_memory=False,
>         set_op_mix_ratio=1.0,
>         local_connectivity=1.0,
>         repulsion_strength=1.0,
>         negative_sample_rate=5,
>         transform_queue_size=4.0,
>         a=None,
>         b=None,
>         random_state=None,
>         angular_rp_forest=False,
>         target_n_neighbors=-1,
>         target_metric='categorical',
>         target_weight=0.5,
>         transform_seed=42,
>         force_approximation_algorithm=False,
>         verbose=False,
>         unique=False
>     )


Layouts diffusion structure with UMAP to achieve dbMAP dimensional reduction. This class refers to the lower
dimensional representation of diffusion components obtained through an adaptive diffusion maps algorithm initially
proposed by [Setty18]. Alternatively, other diffusion approaches can be used, such as
### To do: Fazer a adaptacao p outros algoritmos de diff maps
:param n_components: int (optional, default 2). The dimension of the space to embed into. This defaults to 2 to
provide easy visualization, but can reasonably be set to any integer value in the range 2 to K, K being the number
of samples or diffusion components to embedd.
:param n_neighbors: The size of local neighborhood (in terms of number of neighboring sample points) used for
manifold approximation. Larger values result in more global views of the manifold, while smaller values result in
more local data being preserved. In general values should be in the range 2 to 100.
:param n_jobs: Number of threads to use in calculations. Defaults to all but one.
:param min_dist: The effective minimum distance between embedded points. Smaller values will result in a more
clustered/clumped embedding where nearby points on the manifold are drawn closer together, while larger values will
result on a more even dispersal of points. The value should be set relative to the spread value, which determines
the scale at which embedded points will be spread out.
:param spread: The effective scale of embedded points. In combination with min_dist this determines how
clustered/clumped the embedded points are.
:param learning_rate: The initial learning rate for the embedding optimization.
:return: dbMAP embeddings.


    
#### Ancestors (in MRO)

* [sklearn.base.TransformerMixin](#sklearn.base.TransformerMixin)






    
#### Methods


    
##### Method `fit` {#dbmap.map.Mapper.fit}




>     def fit(
>         self,
>         data,
>         y=0
>     )




    
##### Method `fit_transform` {#dbmap.map.Mapper.fit_transform}




>     def fit_transform(
>         self,
>         data,
>         y=0
>     )


Fit to data, then transform it.

Fits transformer to X and y with optional parameters fit_params
and returns a transformed version of X.

###### Parameters

**```X```** :&ensp;`{array-like, sparse matrix, dataframe}` of <code>shape                 (n\_samples, n\_features)</code>
:   &nbsp;


**```y```** :&ensp;<code>ndarray</code> of <code>shape (n\_samples,)</code>, default=<code>None</code>
:   Target values.


**```**fit_params```** :&ensp;<code>dict</code>
:   Additional fit parameters.

###### Returns

**```X_new```** :&ensp;<code>ndarray array</code> of <code>shape (n\_samples, n\_features\_new)</code>
:   Transformed array.





    
# Multiscale diffusion {#dbmap.multiscale}



    
### Function `multiscale` {#dbmap.multiscale.multiscale}




>     def multiscale(
>         res,
>         n_eigs=None
>     )


Determine multi scale space of the data
:param n_eigs: Number of eigen vectors to use. If None specified, the number
       of eigen vectors will be determined using eigen gap identification.
:return: Multi scaled data matrix




    
# Plotting utilities {#dbmap.plot}



    
### Function `scatter_plot` {#dbmap.plot.scatter_plot}




>     def scatter_plot(
>         res,
>         title=None,
>         fontsize=18,
>         labels=None,
>         pt_size=None,
>         marker='o',
>         opacity=1
>     )

   
# Optimized UMAP (AMAP) {#dbmap.umapper}




    
### Class `AMAP` {#dbmap.umapper.AMAP}




>     class AMAP(
>         n_neighbors=15,
>         n_components=2,
>         metric='euclidean',
>         metric_kwds=None,
>         output_metric='euclidean',
>         output_metric_kwds=None,
>         use_nmslib=True,
>         nmslib_metric='cosine',
>         nmslib_n_jobs=10,
>         nmslib_efC=100,
>         nmslib_efS=100,
>         nmslib_M=30,
>         n_epochs=None,
>         learning_rate=1.5,
>         init='spectral',
>         min_dist=0.6,
>         spread=1.5,
>         low_memory=False,
>         set_op_mix_ratio=1.0,
>         local_connectivity=1.0,
>         repulsion_strength=1.0,
>         negative_sample_rate=5,
>         transform_queue_size=4.0,
>         a=None,
>         b=None,
>         random_state=None,
>         angular_rp_forest=False,
>         target_n_neighbors=-1,
>         target_metric='categorical',
>         target_metric_kwds=None,
>         target_weight=0.5,
>         transform_seed=42,
>         force_approximation_algorithm=False,
>         verbose=False,
>         unique=False
>     )


Adaptive Manifold Approximation and Projection
Finds a low dimensional embedding of the data that approximates
the underlying manifold through fuzzy-union layout. Accelerated
when use_nmslib = <code>True</code>.
#### Parameters

**```n_neighbors```** :&ensp;<code>float (optional</code>, default <code>15)</code>
:   The size of local neighborhood (in terms of number of neighboring
    sample points) used for manifold approximation. Larger values
    result in more global views of the manifold, while smaller
    values result in more local data being preserved. In general
    values should be in the range 2 to 100.


**```n_components```** :&ensp;<code>int (optional</code>, default <code>2)</code>
:   The dimension of the space to embed into. This defaults to 2 to
    provide easy visualization, but can reasonably be set to any
    integer value in the range 2 to 100.


**```use_nmslib```** :&ensp;<code>bool (optional</code>, default <code>True)</code>
:   Whether to use NMSLibTransformer to compute fast approximate nearest
    neighbors. This is a wrapper aroud NMSLIB that supports fast and parallelized
    computation with an array of handy features. If set to True, distances
    are measured in the space defined on the ann_metric parameter.


**```nmslib_metric```** :&ensp;<code>str (optional</code>, default `'cosine')`
:   accepted NMSLIB metrics. Defaults to 'cosine'. Accepted metrics include:
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


**```nmslib_n_jobs```** :&ensp;<code>int (optional</code>, default <code>None)</code>
:   Number of threads to use for approximate-nearest neighbor search.


**```nmslib_efC```** :&ensp;<code>int (optional</code>, default <code>100)</code>
:   increasing this value improves the quality of a constructed graph and leads to higher
    accuracy of search. However this also leads to longer indexing times. A reasonable
    range is 100-2000.


**```nmslib_efS```** :&ensp;<code>int (optional</code>, default <code>100)</code>
:   similarly to efC, improving this value improves recall at the expense of longer
    retrieval time. A reasonable range is 100-2000.


nmslib_M: int (optional, default 30).
    defines the maximum number of neighbors in the zero and above-zero layers during HSNW
    (Hierarchical Navigable Small World Graph). However, the actual default maximum number
    of neighbors for the zero layer is 2*M. For more information on HSNW, please check
    <https://arxiv.org/abs/1603.09320.> HSNW is implemented in python via NMSLIB. Please check
    more about NMSLIB at <https://github.com/nmslib/nmslib> .    n_epochs: int (optional, default None)
    The number of training epochs to be used in optimizing the
    low dimensional embedding. Larger values result in more accurate
    embeddings. If None is specified a value will be selected based on
    the size of the input dataset (200 for large datasets, 500 for small).
**```learning_rate```** :&ensp;<code>float (optional</code>, default <code>1.0)</code>
:   The initial learning rate for the embedding optimization.


**```init```** :&ensp;<code>string (optional</code>, default `'spectral')`
:   How to initialize the low dimensional embedding. Options are:
        * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
        * 'random': assign initial embedding positions at random.
        * A numpy array of initial embedding positions.


**```min_dist```** :&ensp;<code>float (optional</code>, default <code>0.1)</code>
:   The effective minimum distance between embedded points. Smaller values
    will result in a more clustered/clumped embedding where nearby points
    on the manifold are drawn closer together, while larger values will
    result on a more even dispersal of points. The value should be set
    relative to the <code>spread</code> value, which determines the scale at which
    embedded points will be spread out.


**```spread```** :&ensp;<code>float (optional</code>, default <code>1.0)</code>
:   The effective scale of embedded points. In combination with <code>min\_dist</code>
    this determines how clustered/clumped the embedded points are.


**```metric```** :&ensp;<code>string</code> or <code>function (optional</code>, default `'euclidean')`
:   Used if use_nmslib = <code>False</code>. The metric to use to compute distances
    in high dimensional space. If a string is passed it must match a valid
    predefined metric. If a general metric is required a function that takes
    two 1d arrays and returns a float can be provided. For performance purposes
    it is required that this be a numba jit'd function. Valid string metrics
    that should be used within AMAP include:
        * euclidean
        * manhattan
        * seuclidean
        * cosine
        * correlation
        * haversine
        * hamming
        * jaccard


**```low_memory```** :&ensp;<code>bool (optional</code>, default <code>False)</code>
:   If you find that AMAP is failing due to memory constraints
    consider setting use_nmslib to <code>False</code> and this option to <code>True</code>. This approach
    is more computationally expensive, but avoids excessive memory use.


**```set_op_mix_ratio```** :&ensp;<code>float (optional</code>, default <code>1.0)</code>
:   Interpolate between (fuzzy) union and intersection as the set operation
    used to combine local fuzzy simplicial sets to obtain a global fuzzy
    simplicial sets. Both fuzzy set operations use the product t-norm.
    The value of this parameter should be between 0.0 and 1.0; a value of
    1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
    intersection.


**```local_connectivity```** :&ensp;<code>int (optional</code>, default <code>1)</code>
:   The local connectivity required -- i.e. the number of nearest
    neighbors that should be assumed to be connected at a local level.
    The higher this value the more connected the manifold becomes
    locally. In practice this should be not more than the local intrinsic
    dimension of the manifold.


**```repulsion_strength```** :&ensp;<code>float (optional</code>, default <code>1.0)</code>
:   Weighting applied to negative samples in low dimensional embedding
    optimization. Values higher than one will result in greater weight
    being given to negative samples.


**```negative_sample_rate```** :&ensp;<code>int (optional</code>, default <code>5)</code>
:   The number of negative samples to select per positive sample
    in the optimization process. Increasing this value will result
    in greater repulsive force being applied, greater optimization
    cost, but slightly more accuracy.


**```transform_queue_size```** :&ensp;<code>float (optional</code>, default <code>4.0)</code>
:   For transform operations (embedding new points using a trained model_
    this will control how aggressively to search for nearest neighbors.
    Larger values will result in slower performance but more accurate
    nearest neighbor evaluation.


**```a```** :&ensp;<code>float (optional</code>, default <code>None)</code>
:   More specific parameters controlling the embedding. If None these
    values are set automatically as determined by <code>min\_dist</code> and
    <code>spread</code>.


**```b```** :&ensp;<code>float (optional</code>, default <code>None)</code>
:   More specific parameters controlling the embedding. If None these
    values are set automatically as determined by <code>min\_dist</code> and
    <code>spread</code>.


**```random_state```** :&ensp;<code>int, RandomState instance</code> or <code>None</code>, optional `(default: None)`
:   If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by <code>np.random</code>.


**```metric_kwds```** :&ensp;<code>dict (optional</code>, default <code>None)</code>
:   Arguments to pass on to the metric, such as the <code>p</code> value for
    Minkowski distance. If None then no arguments are passed on.


**```angular_rp_forest```** :&ensp;<code>bool (optional</code>, default <code>False)</code>
:   Whether to use an angular random projection forest to initialise
    the approximate nearest neighbor search. This can be faster, but is
    mostly on useful for metric that use an angular style distance such
    as cosine, correlation etc. In the case of those metrics angular forests
    will be chosen automatically.


**```target_n_neighbors```** :&ensp;<code>int (optional</code>, default `-1)`
:   The number of nearest neighbors to use to construct the target simplcial
    set. If set to -1 use the <code>n\_neighbors</code> value.


**```target_metric```** :&ensp;<code>string</code> or <code>callable (optional</code>, default `'categorical')`
:   The metric used to measure distance for a target array is using supervised
    dimension reduction. By default this is 'categorical' which will measure
    distance in terms of whether categories match or are different. Furthermore,
    if semi-supervised is required target values of -1 will be trated as
    unlabelled under the 'categorical' metric. If the target array takes
    continuous values (e.g. for a regression problem) then metric of 'l1'
    or 'l2' is probably more appropriate.


**```target_metric_kwds```** :&ensp;<code>dict (optional</code>, default <code>None)</code>
:   Keyword argument to pass to the target metric when performing
    supervised dimension reduction. If None then no arguments are passed on.


**```target_weight```** :&ensp;<code>float (optional</code>, default <code>0.5)</code>
:   weighting factor between data topology and target topology. A value of
    0.0 weights entirely on data, a value of 1.0 weights entirely on target.
    The default of 0.5 balances the weighting equally between data and target.


**```transform_seed```** :&ensp;<code>int (optional</code>, default <code>42)</code>
:   Random seed used for the stochastic aspects of the transform operation.
    This ensures consistency in transform operations.


**```verbose```** :&ensp;<code>bool (optional</code>, default <code>False)</code>
:   Controls verbosity of logging.


**```unique```** :&ensp;<code>bool (optional</code>, default <code>False)</code>
:   Controls if the rows of your data should be uniqued before being
    embedded.  If you have more duplicates than you have n_neighbour
    you can have the identical data points lying in different regions of
    your space.  It also violates the definition of a metric.




    
#### Ancestors (in MRO)

* [sklearn.base.BaseEstimator](#sklearn.base.BaseEstimator)






    
#### Methods


    
##### Method `fit` {#dbmap.umapper.AMAP.fit}




>     def fit(
>         self,
>         X,
>         y=None
>     )


Fit X into an embedded space.
Optionally use y for supervised dimension reduction.
###### Parameters

**```X```** :&ensp;<code>array, shape (n\_samples, n\_features)</code> or <code>(n\_samples, n\_samples)</code>
:   If the metric is 'precomputed' X must be a square distance
    matrix. Otherwise it contains a sample per row. If the method
    is 'exact', X may be a sparse matrix of type 'csr', 'csc'
    or 'coo'.


**```y```** :&ensp;<code>array, shape (n\_samples)</code>
:   A target array for supervised dimension reduction. How this is
    handled is determined by parameters UMAP was instantiated with.
    The relevant attributes are <code>target\_metric</code> and
    <code>target\_metric\_kwds</code>.



    
##### Method `fit_transform` {#dbmap.umapper.AMAP.fit_transform}




>     def fit_transform(
>         self,
>         X,
>         y=None
>     )


Fit X into an embedded space and return that transformed
output.
###### Parameters

**```X```** :&ensp;<code>array, shape (n\_samples, n\_features)</code> or <code>(n\_samples, n\_samples)</code>
:   If the metric is 'precomputed' X must be a square distance
    matrix. Otherwise it contains a sample per row.


**```y```** :&ensp;<code>array, shape (n\_samples)</code>
:   A target array for supervised dimension reduction. How this is
    handled is determined by parameters UMAP was instantiated with.
    The relevant attributes are <code>target\_metric</code> and
    <code>target\_metric\_kwds</code>.

###### Returns

**```X_new```** :&ensp;<code>array, shape (n\_samples, n\_components)</code>
:   Embedding of the training data in low-dimensional space.



    
##### Method `inverse_transform` {#dbmap.umapper.AMAP.inverse_transform}




>     def inverse_transform(
>         self,
>         X
>     )


Transform X in the existing embedded space back into the input
data space and return that transformed output.
###### Parameters

**```X```** :&ensp;<code>array, shape (n\_samples, n\_components)</code>
:   New points to be inverse transformed.

###### Returns

**```X_new```** :&ensp;<code>array, shape (n\_samples, n\_features)</code>
:   Generated data points new data in data space.



    
##### Method `transform` {#dbmap.umapper.AMAP.transform}




>     def transform(
>         self,
>         X
>     )


Transform X into the existing embedded space and return that
transformed output.
###### Parameters

**```X```** :&ensp;<code>array, shape (n\_samples, n\_features)</code>
:   New data to be transformed.

###### Returns

**```X_new```** :&ensp;<code>array, shape (n\_samples, n\_components)</code>
:   Embedding of the new data in low-dimensional space.



    
### Class `DataFrameUMAP` {#dbmap.umapper.DataFrameUMAP}




>     class DataFrameUMAP(
>         metrics,
>         n_neighbors=15,
>         n_components=2,
>         output_metric='euclidean',
>         output_metric_kwds=None,
>         n_epochs=None,
>         learning_rate=1.0,
>         init='spectral',
>         min_dist=0.1,
>         spread=1.0,
>         set_op_mix_ratio=1.0,
>         local_connectivity=1.0,
>         repulsion_strength=1.0,
>         negative_sample_rate=5,
>         transform_queue_size=4.0,
>         a=None,
>         b=None,
>         random_state=None,
>         angular_rp_forest=False,
>         target_n_neighbors=-1,
>         target_metric='categorical',
>         target_metric_kwds=None,
>         target_weight=0.5,
>         transform_seed=42,
>         verbose=False
>     )


Base class for all estimators in scikit-learn

#### Notes

All estimators should specify all the parameters that can be set
at the class level in their <code>\_\_init\_\_</code> as explicit keyword
arguments (no ``*args`` or ``**kwargs``).


    
#### Ancestors (in MRO)

* [sklearn.base.BaseEstimator](#sklearn.base.BaseEstimator)






    
#### Methods


    
##### Method `fit` {#dbmap.umapper.DataFrameUMAP.fit}




>     def fit(
>         self,
>         X,
>         y=None
>     )










-----
Generated by *pdoc* 0.9.1 (<https://pdoc3.github.io>).
