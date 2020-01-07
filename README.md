# dbMAP
A python module for running diffusion-based Manifold Approximaiton and Projection (dbMAP), a dimensionality reduction method based on diffusion maps and UMAP that is both generalized and computationally efficient.


# Installation and dependencies

   dbMAP has been implemented in Python3, and can be installed using:
```
     $> git clone git://github.com/labcellsign/py_dbMAP.git
     $> cd py_dbMAP
     $> sudo -H pip3 install .
```
   dbMAP depends on a handful of Python3 packages available in PyPi, which are listed in setup.py and automatically installed using the above commands.

# Usage

  dbMAP runs on numpy arrays, pandas dataframes and csr or coo sparse matrices. It takes three steps to run dbMAP on a high-dimensional matrix (such as a gene expression matrix):
        
  ```
  import dbmap
  from sklearn.datasets import load_digits
  import matplotlib.pyplot as plt
  
  #Load some data
  digits = load_digits()
  
  #Runs Diffusion Maps to approximate the Laplace-Beltrami operator
  diff = Run_Diffusion(digits)
  
  #Select the most significant eigenvectors to use in dbMAP. This choice should be made by evaluating an elbow plot. 
 
  plt.plt(diff['EigenValues'])
  plt.show()

  #If none is provided, estimates an adequate number using eigen gap. The selected eigenvectors (diffusion components) 
  #should be multiscaled as described by Setty et al.
  
  res = Multiscale(diff = diff, eigs = None)  
  
  #Visualize the high-dimensional, multiscaled diffusion map results with UMAP by running dbMAP.
  
  embedding = Run_dbMAP(res = res)
   
  ```
  
  

# Citations

dbMAP is powered by algorithms initially proposed in Palantir and by UMAP. If you use dbMAP for your work, please cite the following:

```
Diffusion-based Manifold Approximation and Projection (dbMAP): a comprehensive, generalized and computationally efficient approach for single-cell data visualization. In revision.

Characterization of cell fate probabilities in single-cell data with Palantir. Setty et al., Nature Biotechnology 2019.

McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, ArXiv e-prints 1802.03426, 2018
```

# License and disclaimer

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.
