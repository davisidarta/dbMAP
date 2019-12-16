import numpy as np
import pandas as pd
import os.path
import fcsparser
from scipy.sparse import csc_matrix
from scipy.io import mmread
import tables
import pydiffmap as pdm
import umap


def Run_dbMAP(diff = res['DiffusionComponents'], n_neighbors = 30, n_components = 2, learning_rate = 1.0, min_dist = 0.5, spread = 1.0):
	"""Runs UMAP appropriately on the learned diffusion components
	:param diff: Results from Run_Diffusion (Acessible via res['DiffusionComponents']).
	:param n_components: Results dimensions. Use 2 for 2D plots and 3 for 3D plots.
	:param n_jobs: Number of threads to use in calculations
	:param n_neighbors: The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation. Larger values result in more global views of the manifold, while smaller values result in more local data being preserved. In general values should be in the range 2 to 100.
	:param min_dist: The effective minimum distance between embedded points. Smaller values will result in a more clustered/clumped embedding where nearby points on the manifold are drawn closer together, while larger values will result on a more even dispersal of points. The value should be set relative to the spread value, which determines the scale at which embedded points will be spread out.
	:param spread: The effective scale of embedded points. In combination with min_dist this determines how clustered/clumped the embedded points are.
	:param learning_rate: The initial learning rate for the embedding optimization.
	:return: dbMAP embeddings.
	"""
		embedd = umap.UMAP(n_neighbors = n_neighbors, n_components = n_components, learning_rate = learning_rate, min_dist = min_dist, spread = spread).fit_transform(diff)
  
		return embedd




