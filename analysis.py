import numpy as np
from sklearn.decomposition import PCA

def pca(data,n):

	pca = PCA(n_components=n)
	pca.fit(data)
	return pca

