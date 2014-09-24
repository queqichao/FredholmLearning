import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.graph import graph_laplacian
from sklearn.preprocessing import LabelBinarizer
import util
from fredholm_kernel_learning import BaseL2KernelClassifier
from scipy.sparse import vstack


class LapRLSC(BaseL2KernelClassifier):

  def __init__(self, rbf_gamma=None, normalize_laplacian=False, nu=1.0,
               nu2=None):
    self.rbf_gamma = rbf_gamma
    self.normalize_laplacian = normalize_laplacian
    self.nu = nu
    if nu2 is None:
      self.nu2 = nu
    else:
      self.nu2 = nu2
    self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)

  def fit(self, X, y, unlabeled_data=None):
    num_data = X.shape[0] + unlabeled_data.shape[0]
    num_labeled = X.shape[0]
    num_unlabeled = unlabeled_data.shape[0]
    labeled = range(num_labeled)
    if issparse(X):
      self.X_ = vstack((util.cast_to_float32(X),
                        util.cast_to_float32(unlabeled_data)), format='csr')
    else:
      self.X_ = np.concatenate((util.cast_to_float32(X),
                                util.cast_to_float32(unlabeled_data)))
    self.rbf_gamma_ = (
        self.rbf_gamma if self.rbf_gamma is not None else 1.0 / X.shape[1])
    kernel_matrix_ = rbf_kernel(self.X_, gamma=self.rbf_gamma_)
    I = np.identity(num_data, dtype=np.float32)
    A = np.dot(np.diag(labeled.astype(np.float32)), kernel_matrix_)
    if self.nu2 != 0:
      laplacian_x_kernel = np.dot(graph_laplacian(
          kernel_matrix_, normed=self.normalize_laplacian), kernel_matrix_)
      A += self.nu2 * laplacian_x_kernel
    y = np.concatenate((y, -np.ones((num_unlabeled,), dtype=np.float32)),
                       axis=0)
    super(LapRLSC, self).fit(A, y)

  def predict(self, X):
    kernel_matrix_ = rbf_kernel(
        util.cast_to_float32(X), self.X_, gamma=self.rbf_gamma_)
    return super(LapRLSC, self).predict(kernel_matrix_)
