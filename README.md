[![License: GPL-3.0](https://img.shields.io/badge/License-GNU--GLP%20v3.0-green.svg)](https://opensource.org/licenses/GPL-3.0)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/DaviSidarta.svg?label=Follow%20%40DaviSidarta&style=social)](https://twitter.com/DaviSidarta)



# dbMAP (diffusion-based Manifold Approximation and Projection)
A python module for running diffusion-based Manifold Approximaiton and Projection (dbMAP), a dimensionality reduction method based on diffusion maps and UMAP that is both generalized and computationally efficient. Please check our preprint at SneakPeak: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3582067


# Installation and dependencies

   Prior to installing dbMAP, make sure you have scikit-build and cmake available in your system. These are required for installation.
   ```
     $> sudo apt-get install cmake
     $> pip3 install scikit-build
```
   
   dbMAP has been implemented in Python3, and can be installed using:
```
     $> git clone git://github.com/davisidarta/dbMAP.git
     $> cd dbMAP
     $> sudo -H pip3 install .
```
   dbMAP depends on a handful of Python3 packages available in PyPi, which are listed in setup.py and automatically installed using the above commands. dbMAP was developed and tested in Unix environments and can also be used in Windows machines.

# Usage - Python

  dbMAP runs on numpy arrays, pandas dataframes and csr or coo sparse matrices. It takes three steps to run dbMAP on a high-dimensional matrix (such as a gene expression matrix):
        
  ```
  from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
import pandas as pd
import dbmap
from dbmap.common import embedding_plot

#Load some data
digits = load_digits()
data = digits.data
df = pd.DataFrame(digits.data)

X = digits.data
y = digits.target
n_samples, n_features = X.shape


#Runs Diffusion Maps to approximate the Laplace-Beltrami operator
diff = dbmap.diffusion.diffuse(df, force_sparse=False, n_components= 100, knn= 30)
evals = diff['EigenValues'].ravel()
plt.plot(evals) 
plt.show() #Select the most significant eigenvectors to use in dbMAP by evaluating an elbow plot. 
plt.clf()

#If a number of eigenvector is not chosen, Multiscale() estimates an adequate number using the eigen gap. 
res = dbmap.multiscale.Multiscale(diff = diff)  

#Visualize the high-dimensional, multiscaled diffusion map results with UMAP as the final step of dbMAP.
embedding = dbmap.map.map(res = res, min_dist = 0.05, n_neighbors = 30)
embedding_plot(embedding, y, 'dbMAP visualization of the Digits dataset')
plt.show()
plt.savefig('dbMAP_digits_numbers.png', dpi = 600)
   
  ```
  
  ![dbMAP visualization of the Digits dataset](https://github.com/davisidarta/py_dbMAP/blob/master/dbMAP_digits_numbers.png)


# Usage - R

dbMAP can be easily used in R with the R library reticulate (https://rstudio.github.io/reticulate/). Here we provide an easy example on how to use it to analyze single-cell data with the Seurat single-cell genomics toolkit.

  ```
  #Load required libraries and data -> SeuratData distributes single-cell data
  library(reticulate)
  library(Seurat)
  library(SeuratData)
  
  InstallData("pbmc3k")
  data('pbmc3k')
  
  #Normalize, find variable genes and scale
  pbmc3k <- NormalizeData(pbmc3k)
  pbmc3k <- FindVariableFeatures(pbmc3k)
  pbmc3k <- ScaleData(pbmc3k)
  pbmc3k <- RunPCA(pbmc3k)
  
  #Load dbmap with reticulate
  dbmap <- reticulate::import('dbmap')
  
  #Extract scaled data with the expression of high-variable genes
  data <- t(dat@assays$RNA@scale.data) 
  data <- as.sparse(data)
  data <- r_to_py(data)
  data <- data$tocoo()
   
  #Run dbMAP
  diff <- dbmap$diffusion$diffuse(data)
  res <- dbmap$multiscale$Multiscale(diff)
  emb <- dbmap$map$map(res)
  
  db <- as.matrix(res)
  embedding <- as.matrix(emb)
  
  #Add to Seurat
  pbmc3k@reductions$db <- pbmc3k@reductions$pca   #Create a new reduction slot from PCA
  rownames(db) <- colnames(pbmc3k)
  pbmc3k@reductions$db@cell.embeddings <- db
  
  pbmc3k@reductions$dbmap <- pbmc3k@reductions$pca   #Create a new reduction slot from PCA
  rownames(embedding) <- colnames(dat)
  pbmc3k@reductions$dbmap@cell.embeddings <- embedding
  
  #Plot
  DimPlot(pbmc3k, reduction = 'dbmap', group.by = 'seurat_annotations')
 
  ```
  
  ![dbMAP visualization PBMC single-cell data](https://github.com/davisidarta/py_dbMAP/blob/master/dbMAP_idents-1.png)


# Citations

dbMAP is powered by algorithms initially proposed by Manu Setty et al and Leland McInnes. Standing on the shoulder of giants, we kindly ask that you cite the following if you use dbMAP for your work:

```
Sidarta-Oliveira, Davi and Velloso, Licio, Comprehensive Visualization of High-Dimensional Single-Cell Data With Diffusion-Based Manifold Approximation and Projection (dbMAP). CELL-REPORTS-D-20-01731. Available at SSRN: https://ssrn.com/abstract=3582067 or http://dx.doi.org/10.2139/ssrn.3582067

Characterization of cell fate probabilities in single-cell data with Palantir. Setty et al., Nature Biotechnology 2019.

McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, ArXiv e-prints 1802.03426, 2018
```

# License and disclaimer

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.
