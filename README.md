[![Latest PyPI version](https://img.shields.io/pypi/v/dbmap.svg)](https://pypi.org/project/dbmap/)
[![License: GPL-2.0](https://img.shields.io/badge/License-GNU--GLP%20v2.0-green.svg)](https://opensource.org/licenses/GPL-2.0)
[![Documentation Status](https://readthedocs.org/projects/dbmap/badge/?version=latest)](https://dbmap.readthedocs.io/en/latest/?badge=latest)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/DaviSidarta.svg?label=Follow%20%40davisidarta&style=social)](https://twitter.com/davisidarta)



# dbMAP (diffusion-based Manifold Approximation and Projection)
Diffusion-based Manifold Approximaiton and Projection (dbMAP) is a fast, accurate and modularized machine-learning framework that includes metric-learning,
diffusion harmonics and dimensional reduction.  dbMAP is particularly useful for analyzing highly-structured
data, such as from single-cell RNA sequencing assays. dbMAP was originally designed for the analysis
and visualization of single-cell omics data - yet, as a general dimensional reduction approach based on solid discrete differential geometry, it can be useful in virtually any field in which analysis 
of high-dimensional data is challenging. 

dbMAP explores the use of the [Laplace-Beltrami Operator](https://en.wikipedia.org/wiki/Laplace%E2%80%93Beltrami_operator) (LBO) to geometrically describe heterogeneity within a given high-dimensional dataset.
By extending [Diffusion Maps](https://en.wikipedia.org/wiki/Diffusion_map#:~:text=Diffusion%20maps%20is%20a%20dimensionality,diffusion%20operator%20on%20the%20data.) and providing a scalable and computationally efficient implementation of the algorithm,
it is possible to learn a diffusion basis that approximate the LBO. In dbMAP, this adaptive diffusion basis is used to estimate data's intrinsic dimensionality and then multiscaled to account for all possible diffusion timescales.
From this basis, it is possible to build a _diffusion graph_, which can be visualized with different layout optimizations. Originally, we devised the layout optimization step to be performed with UMAP with a multi-component Laplacian Eigenmaps initialization.
Since the LBO is approximated by the diffusion basis, the LE initialization and the seminal [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html) optimization, this leads to a geometrical consensus on the structure of the underlying data, providing fine visualizations.
However, recently developed optimization methods, such as [PaCMAP](https://github.com/YingfanWang/PaCMAP) and [pyMDE](https://pymde.org/index.html) can also be employed for laying out the diffusion graph.
For more information on dbMAP, check our [preprint](https://www.researchgate.net/publication/341267564_Comprehensive_Visualization_of_High-Dimensional_Single-Cell_Data_With_Diffusion-Based_Manifold_Approximation_and_Projection_dbMAP).

This implementation includes a flexible and extendable wrapper for nmslib, for state-of-the-art approximate nearest-neighbors search, functions for fast computation of 
diffusion dynamics and multiscale diffusion maps into a diffusion basis, and faster implementations of adapted UMAP and PaCUMAP layouts.
Further documentation is available at [Read the Docs](https://dbmap.readthedocs.io/en/latest/).

## Installation and dependencies

   Prior to installing dbMAP, make sure you have scikit-build and cmake available in your system. These are required for installation.
   ```
     $> sudo apt-get install cmake
     $> pip3 install scikit-build
   ```
   We're also going to need NMSlib for really fast approximate nearest-neighborhood search, and Annoy for fast indexing:
   ```
    $> pip3 install nmslib annoy
   ```
   You can read more about NMSlib  [here](https://github.com/nmslib/nmslib), and check more on the available distances and spaces documentation [here](https://github.com/nmslib/nmslib/blob/master/manual/spaces.md). dbMAP implements functions derived from scikit-learn base transformers tat make NMSlib more generally extendable to machine-leraning workflows, and we are grateful to the nmslib community for their insights during this process.
   For now the dependency on annoy is intended to give support to the PaCMAP optimization, but I'm working on keeping it to all to NMSlib.

## Using dbMAP
  dbMAP consists of two main steps: an adaptive anisotropic reproduction of the initial input diffusion structure, followed by an accelerated UMAP or graph layout. dbMAP runs on numpy arrays, pandas dataframes and csr or coo sparse matrices. The adaptive diffusion reduction is recommended over PCA if data is significantly non-linear, and is useful for clustering and downstream analysis. The UMAP and graph layouts are also useful for big data visualization. 
  Here follows some examples on using dbMAP implemented algorithms, including fast neighborhood search, adaptive multiscaled diffusion maps and accelerated UMAP and graph layouts:
  
  ### 1 - Fast approximate k-nearest-neighbors
  dbMAP implements the NMSlibTransformer() class, which calls nmslib to perform a fast and accurate approximate nearest neighbor search. The NMSlibTransformer() class has several methods to compute and retrieve this information, and an additional function to measure it's accuracy.

   ```
  # Load some libraries:
  from sklearn.datasets import load_digits
  from scipy.sparse import csr_matrix
  import dbmap as dm

  # Load some data and convert to CSR for speed:
  digits = load_digits()
  data = csr_matrix(digits.data)

  # Initialize the NMSlibTransformer() object and index the data:
  anbrs = dm.ann.NMSlibTransformer() # Feel free to play with parameters
  anbrs = anbrs.fit(data)

  # Compute the knn_neighbors graph:
  knn_graph = anbrs.transform(data)

  # Compute indices, distances, gradient and knn_neighbors graph:
  inds, dists, grad, knn = anbrs.ind_dist_grad(data)

  # Test approximate-neighbors accuracy:
  anbrs.test_efficiency(data)
   ```


  ### 2 - Fast adaptive multiscaled diffusion maps
  dbMAP implements the Diffusor() class, which allows state-of-the-art dimensional reduction by the fast approximation of the Laplace Beltrami operator and automatic detection of intrinsic dimensionality. This algorithm learns a local metric which is normalized and embedded as a diffusion distance on the series of orthogonal components that define structure variability within the initial informational space.
  Default machine-learning analysis sometimes employs PCA on highly non-linear data despite its caveat of being unsuitable for datasets which cannot be represented as a series of linear correlations. The main reason for this is the low computational cost of PCA compared to non-linear dimensional reduction methods. Our implementation is scalable to extremely high-dimensional datasets (10e9 samples) and oughts to provide more reliable information than PCA on real-world, non-linear data. Similarly to our fast nearest-neighbor implementation, we provide utility functions to obtain results in different formats.
  
  ```
  # Load some libraries:
  from sklearn.datasets import load_digits
  from scipy.sparse import csr_matrix
  import dbmap as dm

  # Load some data and convert to CSR for speed:
  digits = load_digits()
  data = csr_matrix(digits.data)
   
  # Initialize the diffusor object and fit data:
  diff = dm.diffusion.Diffusor().fit(data)
   
  # Return low dimensional representation of data:
  res = diff.transform(data)
   
  # Return the diffusion indices, distances, diffusion gradient and diffusion graph:
  ind, dist, grad, graph = diff.ind_dist_grad(data)
   
 ```
  A key feature of dbMAP diffusion approach is its ability to indirectly estimate data intrinsic dimensionality by looking for all positive-eigenvalued components. The algorithm tries to find an optimal number of final components for eigendecomposition such as to find an eigengap that maximizes the information each component carries. In other words, we want to compute the minimal number of components needed to find negative-valued components. These can then be visualized as follows:
 
 ```
 import matplotlib.pyplot as plt
 
 res = diff.return_dict()

 plt.plot(range(0, len(res['EigenValues'])), res['EigenValues'], marker='o')
```
  
  ### 3 - Fast mapping layout:
   
   For scalable big data visualization, we provide a fast mapping layout of the adaptive multiscale diffusion components space. We adapted UMAP to construct fast approximate simplicial complexes wich normalizes the data structure, rendering a comprehensive layout. We also provide fast graph layout of the resulting components with fa2, which implements scalable and interative layouts within networkx. A vanilla UMAP implementation is also provided.
      
  ```
  # Load some libraries:
  from sklearn.datasets import load_digits
  from scipy.sparse import csr_matrix
  import dbmap as dm
  import umap

  # Load some data and convert to CSR for speed:
  digits = load_digits()
  data = csr_matrix(digits.data)
   
  # Fit the diffusion model
  digits_diff = dm.diffusion.Diffusor(n_neighbors=30, n_components=120,
                                        kernel_use='simple_adaptive', # The diffusion tool is highly
                                        norm=True,          # customizable!
                                        transitions=False).fit(digits.data)
  # Decompose the diffusion basis
  db = digits_diff.transform(digits.data)
    
  # Visualize data intrinsic dimensionality
  res = digits_diff.return_dict()
  plt.plot(range(0, len(res['EigenValues'])), res['EigenValues'])
  plt.show()  

  # We're currently improving the non-uniform simplicial embedding and introducing an adaptive optimization layout procedure (AdapMAP). 
  # Please use vanilla UMAP or our implementation of PaCMAP in the meanwhile:
  
  # Embed diffusion graph with vanilla UMAP:
  db_umap_emb = dm.map.UMAP(n_epochs=100).fit(transform(db.to_numpy(dtype='float32'))
  
  plt.scatter(db_umap_emb[:, 0], db_umap_emb[:, 1], c=digits.target, cmap='Spectral', s=5)
  plt.gca().set_aspect('equal', 'datalim')
  plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
  plt.title('UMAP projection of Digits diffusion graph', fontsize=24)
  plt.show()

  # Embed diffusion graph with PaCMAP using a sparsely-optimized UMAP initialization:
  init = dm.map.UMAP(n_epochs=100).fit(transform(db.to_numpy(dtype='float32'))
  dbpac = dm.pacmapper.PaCMAP(n_dims=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0) 
  dbpac_emb = embedding.fit_transform(db, init=db_umap_emb)  

  plt.scatter(dacmap_emb[:, 0], dacmap_emb[:, 1], c=digits.target, cmap='Spectral', s=5)
  plt.gca().set_aspect('equal', 'datalim')
  plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
  plt.title('PaCMAP projection of the Digits diffusion graph', fontsize=24)
  ```
 ![dbMAP handwritten digits visualization](https://github.com/davisidarta/dbMAP/blob/master/Digits.png)

# Examples of using dbMAP on single-cell RNA sequencing data

dbMAP excels at the analysis of single-cell RNA sequencing data. We provide straighforward scripts to use dbMAP with
scanpy in python and with Seurat in R.

## Python - Scanpy

```
# Import some packages
import numpy as np
from scipy.sparse import csr_matrix
import dbmap as dm
import scanpy as sc

# Load your data (raw or normalized cells x genes 
# or cells x variable_genes, but NOT SCALED!)
adata = sc.read_h5ad(YOUR ANNDATA)

# It's computationally very cheap to diffuse on sparse matrices
data = csr_matrix(adata.X) 

# Fit the diffusion process
diff = dm.diffusion.Diffusor(ann_dist='cosine', 
                             n_jobs=8,
                             n_neighbors=15, n_components=120,
                             kernel_use='decay_adaptive',
                             transitions=False, norm=False).fit(data)
# Decompose graph
db = diff.transform(data)

db = np.array(db)
res = diff.return_dict()
adata.obsm['X_db'] = db

# Visualize meaningful components and the eigengap
import matplotlib.pyplot as plt
plt.plot(range(0, len(res['EigenValues'])), res['EigenValues'], marker='o')

# Diffusion graph clustering
sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_db', metric='euclidean')
sc.tl.leiden(adata, resolution=0.4)

# Diffusion graph layout with UMAP
import umap
db_umap_emb = umap.UMAP().fit_transform(db)
adata.obsm['X_dbmap'] = db_umap_emb
sc.pl.embedding(adata, basis ='X_dbmap', color='clusters') 

# Diffusion graph layout with PaCMAP on the diffusion basis
db_pac_emb = dm.pacmapper.PaCMAP(n_dims=2, n_neighbors=15, MN_ratio=3, FP_ratio=.5) 
db_pac_emb_fit = db_pac_emb.fit_transform(db, init='random')
adata.obsm['X_db_pacmap'] = db_pac_emb_fit
sc.pl.embedding(adata, basis ='X_db_pacmap', color='clusters')

# Diffusion graph layout with PaCMAPUMAP on the diffusion basis
db_pac_emb = dm.pacmapper.PaCMAP(n_dims=2, n_neighbors=15, MN_ratio=3, FP_ratio=.5) 
db_pac_emb_fit = db_pac_emb.fit_transform(db, init='random')
adata.obsm['X_db_pacmap'] = db_pac_emb_fit
```


## R with Reticulate - Seurat

```
# Call dbMAP python diffusion functions with reticulate and use vanilla UMAP layout of the diffusion graph on Seurat

library(reticulate)
np <- reticulate::import("numpy")
pd <- reticulate::import("pandas")
sp <- reticulate::import("scipy")
dbmap <- reticulate::import('dbmap')

dat <- YOUR_SEURAT_OBJECT

data <- t(dat@assays$YOUR_ASSAY@data[VariableFeatures(dat),])
a <- r_to_py(data)
b <- a$tocsr()
diff <- dbmap$diffusion$Diffusor(n_components = as.integer(80), n_neighbors = as.integer(15),
                                 transitions = as.logical(F),
                                 norm = as.logical(T), ann_dist = as.character('cosine'),
                                 n_jobs = as.integer(10), kernel_use = as.character('decay_adaptive')
)
diff = diff$fit(b)
db = as.matrix(diff$transform(b))
res = diff$return_dict()

#Visualize meaningful diffusion components.
evals <- py_to_r(res$EigenValues)
plot(evals) 

# Deal with names
rownames(db) <- colnames(dat)
new_names <- list()
for(i in 1:length(colnames(db))){new_names[i] <- paste('DB_' , as.integer(colnames(sc[i])) + 1, sep = '')}
colnames(db) <- as.vector(new_names)
names(evals) <- as.vector(new_names)

# Return to Seurat
dat[["db"]] <- CreateDimReducObject(embeddings = db, key = "db_", assay = DefaultAssay(dat))

# Cluster on the diffusion graph
dat <- FindNeighbors(dat, reduction = 'db', dims = 1:ncol(dat@reductions$db@cell.embeddings), annoy.metric = 'cosine', graph.name = 'dbgraph')
dat <- FindClusters(dat, resolution = 1, graph.name = 'dbgraph', algorithm = 2)

# UMAP layout of the diffusion graph
dat <- RunUMAP(dat, reduction = 'db', dims = 1:ncol(dat@reductions$db@cell.embeddings), n.neighbors = 10, init = 'spectral',
               min.dist = 0.6, spread = 1.5, learning.rate = 1.5, n.epochs = 200, reduction.key = 'dbMAP_', reduction.name = 'dbmap')

# Plot
DimPlot(dat, reduction = 'dbmap', group.by = 'seurat_clusters', pt.size = 1)

```


# Benchmarking

As we prepare for a second version of the manuscript, extensive benchmarking of dbMAP and other dimensionality reduction methods is underway. For the time being, consider the runtime comparison between PCA, dbMAP (and its diffusion process alone) and the fastest non-linear algorithms to date: PHATE and UMAP.

![dbMAP_runtime_benchmark](https://github.com/davisidarta/dbMAP/blob/master/benchmark.png)

# Citation

We kindly ask that you cite dbMAP preprint if you find it useful for your work:

Sidarta-Oliveira, D., & Velloso, L. (2020). Comprehensive Visualization of High-Dimensional Single-Cell Data With&nbsp;Diffusion-Based Manifold Approximation and Projection(dbMAP). SSRN Electronic Journal. https://doi.org/10.2139/ssrn.3582067




# License and disclaimer

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.
