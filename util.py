import numpy as np
from scipy.sparse import issparse

def cast_to_float32(X):
  if issparse(X):
    X.data = np.float32(X.data)
  else:
    X = np.float32(X)
  return X
