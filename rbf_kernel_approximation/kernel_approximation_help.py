from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import numpy as np
import util

def preprocessing(data, preprocessing_config):
  if preprocessing_config["contrast_normalization"]:
    data = util.contrast_normalization(data, bias=3, copy=False)
  
  if preprocessing_config["whiten"]:
    M = np.mean(data, axis=0)
    U,s,V = np.linalg.svd(data-M, full_matrices=False)
    var = (s ** 2) / data.shape[0]
    data = np.dot(data-M, np.dot(V.T, np.dot(np.diag(1/(var+0.1)), V)))
  return data

def get_kmeans(cluster_parm):
  if cluster_parm["method"] == "mini-batch":
    kmeans = MiniBatchKMeans(n_clusters=cluster_parm["n_clusters"],
                             max_iter=cluster_parm["max_iter"],
                             batch_size=cluster_parm["batch_size"])
  else:
    kmeans = KMeans(n_clusters=cluster_parm["n_clusters"],
                    max_iter=cluster_parm["max_iter"])
  return kmeans


