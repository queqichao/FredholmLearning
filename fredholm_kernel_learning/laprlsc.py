import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.graph import graph_laplacian
from sklearn.preprocessing import LabelBinarizer
import util
from fredholm_kernel_learning import BaseL2KernelClassifier


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

  def get_label_matrix(self, y):
    num_data = y.shape[0]
    labeled = y != -1
    unlabeled = y == -1
    num_labeled = y[labeled].shape[0]
    num_unlabeled = num_data - num_labeled
    Y_labeled = self._label_binarizer.fit_transform(y[labeled])
    Y_unlabeled = np.zeros(
        (num_unlabeled, Y_labeled.shape[1],), dtype=np.float32)
    Y = np.zeros((num_data, Y_labeled.shape[1]), dtype=np.float32)
    Y[labeled] = Y_labeled
    Y[unlabeled] = Y_unlabeled
    return Y

  def fit(self, X, y):
    num_data = X.shape[0]
    labeled = y != -1
    self.X_ = util.cast_to_float32(X)
    self.rbf_gamma_ = (
        self.rbf_gamma if self.rbf_gamma is not None else 1.0 / X.shape[1])
    kernel_matrix_ = rbf_kernel(self.X_, gamma=self.rbf_gamma_)
    I = np.identity(num_data, dtype=np.float32)
    A = np.dot(np.diag(labeled.astype(np.float32)), kernel_matrix_)
    if self.nu2 != 0:
      laplacian_x_kernel = np.dot(graph_laplacian(
          kernel_matrix_, normed=self.normalize_laplacian), kernel_matrix_)
      A += self.nu2 * laplacian_x_kernel
    super(LapRLSC, self).fit(A, y)

  def predict(self, X):
    kernel_matrix_ = rbf_kernel(
        util.cast_to_float32(X), self.X_, gamma=self.rbf_gamma_)
    return super(LapRLSC, self).predict(kernel_matrix_)
