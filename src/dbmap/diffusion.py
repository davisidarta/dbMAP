import numpy as np
import pandas as pd
import os.path
import fcsparser
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, find, issparse
from scipy.sparse.linalg import eigs
import tables

def diffuse(data, n_components=50, knn=30, n_jobs=-1, alpha=1, force_sparse = True):
	"""Runs Diffusion maps using an adaptation of the adaptive anisotropic kernel proposed by Setty et al, Nature Biotechnology 2019.
	:param data: Data matrix to diffuse from. Either a sparse .coo or a dense pandas dataframe.
	:param n_components: Number of diffusion components to compute. Defaults to 50. We suggest larger values if analyzing more than 10,000 cells.
	:param n_jobs: Number of threads to use in calculations. Defaults to all but one.
	:param knn: Number of k-nearest-neighbors to compute. The adaptive kernel will normalize distances by each cell distance of its median neighbor.
	:param force_sparse: Whether to convert input data (coo or dataframe) to a csr sparse format for speeding calculations. Defaults to True. 
	:return: Diffusion components, associated eigenvalues and suggested number of resulting components to use during Multiscaling.
	"""
	print('Converting input to sparse. Determing nearest neighbor graph...')
	if force_sparse == True:
		data = data.tocsr()
		print('Converting input to sparse for efficiency. Determing nearest neighbor graph...')
	else:
		print('Dense input. Determing nearest neighbor graph...')

	N = data.shape[0]
	#Construct a k-nearest-neighbors graph
	nbrs = NearestNeighbors(n_neighbors=int(knn), metric='euclidean', n_jobs=n_jobs).fit(data)
	kNN = nbrs.kneighbors_graph(data, mode='distance')
	# Adaptive k: distance to cell median nearest neighbors, used for kernel normalizaiton
	adaptive_k = int(np.floor(knn / 2))
	nbrs = NearestNeighbors(n_neighbors=int(adaptive_k), metric='euclidean', n_jobs=n_jobs).fit(data)
	adaptive_std = nbrs.kneighbors_graph(data, mode='distance').max(axis=1)
	adaptive_std = np.ravel(adaptive_std.todense())

	# Distance metrics
	x, y, dists = find(kNN)  #k-nearest-neighbor distances
	
	# X, y specific stds
	dists = dists / adaptive_std[x] #Normalize by the distance of median nearest neighbor
	W = csr_matrix((np.exp(-dists), (x, y)), shape=[N, N]) #Normalized distances

	#Kernel construction
	kernel = W + W.T 
	#Diffusion through Markov chain
	D = np.ravel(kernel.sum(axis=1))
	if alpha > 0:
		# L_alpha
		D[D != 0] = D[D != 0] ** (-alpha)
		mat = csr_matrix((D, (range(N), range(N))), shape=[N, N])
		kernel = mat.dot(kernel).dot(mat)
		D = np.ravel(kernel.sum(axis=1))

	D[D != 0] = 1 / D[D != 0]
	
	#Setting the diffusion operator
	T = csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(kernel)
	
	# Eigen value decomposition
	D, V = eigs(T, n_components, tol=1e-4, maxiter=1000)
	D = np.real(D)
	V = np.real(V)
	inds = np.argsort(D)[::-1]
	D = D[inds]
	V = V[:, inds]

	# Normalize by the first diffusion component
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
	
	return res
	
