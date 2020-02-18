# dbMAP
A python module for running diffusion-based Manifold Approximaiton and Projection (dbMAP), a dimensionality reduction method based on diffusion maps and UMAP that is both generalized and computationally efficient.


# Installation and dependencies

   dbMAP has been implemented in Python3, and can be installed using:
```
     $> git clone git://github.com/labcellsign/py_dbMAP.git
     $> cd py_dbMAP
     $> sudo -H pip3 install .
```
   dbMAP depends on a handful of Python3 packages available in PyPi, which are listed in setup.py and automatically installed using the above commands. dbMAP was developed and tested in Unix environments and can also be used in Windows machines.

# Usage

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
diff = dbmap.diffusion.Run_Diffusion(df, force_sparse=False, n_components= 100, knn= 30)
evals = diff['EigenValues'].ravel()
plt.plot(evals) 
plt.show() #Select the most significant eigenvectors to use in dbMAP by evaluating an elbow plot. 
plt.clf()

#If a number of eigenvector is not chosen, Multiscale() estimates an adequate number using the eigen gap. 
res = dbmap.diffusion.Multiscale(diff = diff)  

#Visualize the high-dimensional, multiscaled diffusion map results with UMAP as the final step of dbMAP.
embedding = dbmap.dbmap.Run_dbMAP(res = res, min_dist = 0.05, n_neighbors = 30)
embedding_plot(embedding, 'dbMAP visualization of the Digits dataset')
plt.show()
plt.savefig('dbMAP_digits_numbers.png', dpi = 600)
   
  ```
  
  ![dbMAP visualization of the Digits dataset](https://github.com/davisidarta/py_dbMAP/blob/master/dbMAP_digits_numbers.png)

# Citations

dbMAP is powered by algorithms initially proposed by Manu Setty et al and Leland McInnes. Standing on the shoulder of giants, we kindly ask that you cite the following if you use dbMAP for your work:

```
Diffusion-based Manifold Approximation and Projection (dbMAP): a comprehensive, generalized and computationally efficient approach for single-cell data visualization. Submitted.

Characterization of cell fate probabilities in single-cell data with Palantir. Setty et al., Nature Biotechnology 2019.

McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, ArXiv e-prints 1802.03426, 2018
```

# License and disclaimer

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.
