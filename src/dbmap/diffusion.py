import numpy as np
import pandas as pd
import os.path
import fcsparser
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, find, issparse
from scipy.sparse.linalg import eigs
from scipy.io import mmread
import tables
import pydiffmap as pdm
import scanpy as sc

def Run_Diffusion(data, n_components=50, knn=30, n_jobs=-1, alpha=1, force_sparse = True):
	"""Run Diffusion maps using the adaptive anisotropic kernel proposed by Setty et al, Nature Biotechnology 2019 - Characterization of cell fate probabilities in single-cell data with Palantir
	:param data: Data matrix to diffuse from
	:param n_components: Number of diffusion components to diffuse with
	:param n_jobs: Number of threads to use in calculations
	:param knn: Number of k-nearest-neighbors to use. The adaptive kernel will normalize distances by the k/2 neighbor distances, which is the median neighbor.
	:param force_sparse: Whether to convert input data to the sparse format for speeding calculations.
	:return: Multiscaled results, diffusion components, associated eigenvalues and suggested number of resulting components to use.
	"""

	# Determine the kernel
	N = data.shape[0]
	if (((issparse(data) == False) & (force_sparse == False))):
		print('Using dense input. Converting to sparse with force_sparse = True is recommended for scalability. Determing nearest neighbor graph...')
		nbrs = NearestNeighbors(n_neighbors=int(knn), metric = 'euclidean', n_jobs=n_jobs).fit(data.values)
		kNN = nbrs.kneighbors_graph(data.values, mode='distance')
		# Adaptive k
		adaptive_k = int(np.floor(knn / 2))
		nbrs = NearestNeighbors(n_neighbors=int(adaptive_k), metric='euclidean', n_jobs=n_jobs).fit(data.values)
		adaptive_std = nbrs.kneighbors_graph(data.values, mode='distance').max(axis=1)
		adaptive_std = np.ravel(adaptive_std.todense())
		# Kernel
		x, y, dists = find(kNN)
		# X, y specific stds
		dists = dists / adaptive_std[x]
		W = csr_matrix((np.exp(-dists), (x, y)), shape=[N, N])
		# Diffusion components
		kernel = W + W.T
					
	if (((issparse(data) == False) & (force_sparse == True))):
		print('Converting from dense to sparse matrix. Determing nearest neighbor graph...')
		data=data.tocsr()
		nbrs = NearestNeighbors(n_neighbors=int(knn), metric = 'euclidean', n_jobs=n_jobs).fit(data)
		kNN = nbrs.kneighbors_graph(data.values, mode='distance')

		# Adaptive k
		adaptive_k = int(np.floor(knn / 2))
		nbrs = NearestNeighbors(n_neighbors=int(adaptive_k), metric='euclidean', n_jobs=n_jobs).fit(data)
		adaptive_std = nbrs.kneighbors_graph(data.values, mode='distance').max(axis=1)
		adaptive_std = np.ravel(adaptive_std.todense())
		# Kernel
		x, y, dists = find(kNN)
		# X, y specific stds
		dists = dists / adaptive_std[x]
		W = csr_matrix((np.exp(-dists), (x, y)), shape=[N, N])
		# Diffusion components
		kernel = W + W.T
	
	if issparse(data):
		print('Sparse matrix input. Determing nearest neighbor graph...')
		nbrs = NearestNeighbors(n_neighbors=int(knn), metric='euclidean', n_jobs=n_jobs).fit(data)
		kNN = nbrs.kneighbors_graph(data, mode='distance')

		# Adaptive k
		adaptive_k = int(np.floor(knn / 2))
		nbrs = NearestNeighbors(n_neighbors=int(adaptive_k), metric='euclidean', n_jobs=n_jobs).fit(data)
		adaptive_std = nbrs.kneighbors_graph(data, mode='distance').max(axis=1)
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
	if not issparse(data):
		res['EigenVectors'].index = data.index
		res['EigenValues'] = pd.Series(res['EigenValues'])
		res['kernel'] = kernel

	#Suggest a number of components to use        
	vals = np.ravel(res['EigenValues'])
	n_eigs = np.argsort(vals[:(len(vals) - 1)] - vals[1:])[-1] + 1
	if n_eigs < 3:
		n_eigs = np.argsort(vals[:(len(vals) - 1)] - vals[1:])[-2] + 1
	res['Suggested_eigs'] = n_eigs 
	print("Suggestion of components to use, accordingly to Setty et al:" + n_eigs)

	# Scale the data
	use_eigs = list(range(1, n_eigs))
	eig_vals = np.ravel(res['EigenValues'])
	res['DiffusionComponents'] = res['EigenVectors'].values[:,] * (eig_vals / (1 - eig_vals))
	res['DiffusionComponents'] = pd.DataFrame(result, index=res['EigenVectors'].index)

	return res
