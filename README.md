# dbMAP
A python repo for running diffusion-based Manifold Approximaiton and Projection, a dimensionality reduction method based on diffusion maps and UMAP that is both generalized and computationally efficient.


# Installation and dependencies

   dbMAP has been implemented in Python3, and can be installed using:
```
     $> git clone git://github.com/labcellsign/py_dbMAP.git
     $> cd py_dbMAP
     $> sudo -H pip3 install .
```
   dbMAP depends on a handful of Python3 packages available in PyPi, which are listed in setup.py and automatically installed using the above commands.

# Usage

  dbMAP runs on numpy arrays, pandas dataframes and csr or coo sparse matrices. To produce a dbMAP embedding from
  a high dimensional data matrix, Run_Diffusion must be run on the total matrix, and Run_dbMAP must be run on Run_Diffusion       results:
  
  ```
  import dbmap
  from sklearn.datasets import load_digits
  
  digits = load_digits()
  
  diff = Run_Diffusion(digits)
  embedding = Run_dbMAP(diff = diff)
   
  ```
  
  

# Citations

dbMAP is powered by algorithms initially proposed by Palantir and UMAP. If you use dbMAP for your work, please cite the following:

```
Diffusion-based Manifold Approximation and Projection (dbMAP): a comprehensive, generalized and computationally efficient approach for single-cell data visualization. In revision.

Characterization of cell fate probabilities in single-cell data with Palantir. Setty et al., Nature Biotechnology 2019.

McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, ArXiv e-prints 1802.03426, 2018
```

# License and disclaimer

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.
