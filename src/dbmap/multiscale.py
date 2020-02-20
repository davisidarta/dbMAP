import numpy as np
import pandas as pd
import os.path
import fcsparser
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, find, issparse
from scipy.sparse.linalg import eigs
import tables


def Multiscale(diff, n_eigs=None):
	"""Determine multi scale space of the data
	:param diff: Diffusion map results from 'diffusion()'. 
	:param n_eigs: Number of eigen vectors to use. If None specified, the number of eigen vectors will be determined using eigen gap
	:return: Multiscaled diffusion components.
	"""
	if n_eigs is None:
		vals = np.ravel(diff['EigenValues'])
		n_eigs = np.argsort(vals[:(len(vals) - 1)] - vals[1:])[-1] + 1
		if n_eigs < 3:
			n_eigs = np.argsort(vals[:(len(vals) - 1)] - vals[1:])[-2] + 1
			
	# Scale the components
	use_eigs = list(range(1, n_eigs))
	eig_vals = np.ravel(diff['EigenValues'][use_eigs])
	data = diff['EigenVectors'].values[:,use_eigs] * (eig_vals / (1 - eig_vals))
	data = pd.DataFrame(data, index=diff['EigenVectors'].index)
	return data
