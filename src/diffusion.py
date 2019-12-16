import numpy as np
import pandas as pd
import os.path
import fcsparser
from scipy.sparse import csc_matrix
from scipy.io import mmread
import tables
import pydiffmap as pdm




def Run_Diffusion(data, n_components=50, knn=30, n_jobs=-1, alpha=1):
  """Run Diffusion maps using the adaptive anisotropic kernel proposed by Setty et al, Nature Biotechnology 2019 - Characterization of cell fate probabilities in single-cell data with Palantir
  
  :param data: Data matrix to diffuse from
  :param n_components: Number of diffusion components to diffuse with
  :param n_jobs: Number of threads to use in calculations
  :param knn: Number of k-nearest-neighbors to use. The adaptive kernel will normalize distances by the k/2 neighbor distances, which is the median neighbor.
  :param 
  
  # Determine the adaptive kernel
  N = data_df.shape[0]
  if not issparse(data):
    
  print('Determing nearest neighbor graph...')
  # nbrs = NearestNeighbors(n_neighbors=int(knn), metric='euclidean',
  #                         n_jobs=n_jobs).fit(data_df.values)
  # kNN = nbrs.kneighbors_graph(data_df.values, mode='distance')
  temp = sc.AnnData(data_df.values)
  sc.pp.neighbors(temp, n_pcs=0, n_neighbors=knn)
  kNN = temp.uns['neighbors']['distances']

  # Adaptive k
  adaptive_k = int(np.floor(knn / 2))
  nbrs = NearestNeighbors(n_neighbors=int(adaptive_k),
                                metric='euclidean', n_jobs=n_jobs).fit(data_df.values)
  adaptive_std = nbrs.kneighbors_graph(
       data_df.values, mode='distance').max(axis=1)
  adaptive_std = np.ravel(adaptive_std.todense())

  # Kernel
  x, y, dists = find(kNN)

  # X, y specific stds
  dists = dists / adaptive_std[x]
  W = csr_matrix((np.exp(-dists), (x, y)), shape=[N, N])

  # Diffusion components
  kernel = W + W.T


  # Markov
  D = np.ravel(kernel.sum(axis=1))

  if alpha > 0:
        # L_alpha
        D[D != 0] = D[D != 0] ** (-alpha)
        mat = csr_matrix((D, (range(N), range(N))), shape=[N, N])
        kernel = mat.dot(kernel).dot(mat)
        D = np.ravel(kernel.sum(axis=1))

  D[D != 0] = 1 / D[D != 0]
  T = csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(kernel)
  # Eigen value dcomposition
  D, V = eigs(T, n_components, tol=1e-4, maxiter=1000)
  D = np.real(D)
  V = np.real(V)
  inds = np.argsort(D)[::-1]
  D = D[inds]
  V = V[:, inds]

  # Normalize
  for i in range(V.shape[1]):
      V[:, i] = V[:, i] / np.linalg.norm(V[:, i])

  # Create the results dictionary
  res = {'T': T, 'EigenVectors': V, 'EigenValues': D}
  res['EigenVectors'] = pd.DataFrame(res['EigenVectors'])
  if not issparse(data_df):
      res['EigenVectors'].index = data_df.index
  res['EigenValues'] = pd.Series(res['EigenValues'])
  res['kernel'] = kernel
            
  #Suggest a number of components to use        
    
  vals = np.ravel(res['EigenValues'])
  n_eigs = np.argsort(vals[:(len(vals) - 1)] - vals[1:])[-1] + 1
  if n_eigs < 3:
        n_eigs = np.argsort(vals[:(len(vals) - 1)] - vals[1:])[-2] + 1
    
  print("Suggestion of components to use, accordingly to Setty et al:" + n_eigs)
    
  # Scale the data
  use_eigs = list(range(1, n_eigs))
  eig_vals = np.ravel(res['EigenValues'])
  result = res['EigenVectors'].values[:,] * (eig_vals / (1 - eig_vals))
  result = pd.DataFrame(result, index=res['EigenVectors'].index)

  return result













