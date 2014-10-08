from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import numpy as np
from scipy.sparse import issparse

def cast_to_float32(X):
  if issparse(X):
    X.data = np.float32(X.data)
  else:
    X = np.float32(X)
  return X

def get_kmeans(cluster_parm):
  if cluster_parm["method"] == "mini-batch":
    kmeans = MiniBatchKMeans(n_clusters=cluster_parm["n_clusters"],
                             max_iter=cluster_parm["max_iter"],
                             batch_size=cluster_parm["batch_size"])
  else:
    kmeans = KMeans(n_clusters=cluster_parm["n_clusters"],
                    max_iter=cluster_parm["max_iter"])
  return kmeans

