import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel, pairwise_kernels
from sklearn.utils.graph import graph_laplacian
from sklearn.preprocessing import LabelBinarizer
import util
from fredholm_kernel_learning import BaseL2KernelClassifier
from scipy.sparse import vstack
from scipy.sparse import issparse

class LapRLSC(BaseL2KernelClassifier):

  def __init__(self, kernel='rbf', gamma=None, degree=None, coef0=None,
               normalize_laplacian=False, nu=1.0, nu2=None):
    self.normalize_laplacian = normalize_laplacian
    self.nu = nu
    self.kernel = kernel
    self.gamma = gamma
    self.degree = degree
    self.coef0 = coef0
    if nu2 is None:
      self.nu2 = nu
    else:
      self.nu2 = nu2
    super(LapRLSC, self).__init__(nu=self.nu)

  def fit(self, X, y, unlabeled_data=None):
    num_data = X.shape[0] + unlabeled_data.shape[0]
    num_labeled = X.shape[0]
    num_unlabeled = unlabeled_data.shape[0]
    labeled = np.zeros((num_data,), dtype=np.float32)
    labeled[0:num_labeled] = 1.0
    if issparse(X):
      self.X_ = vstack((util.cast_to_float32(X),
                        util.cast_to_float32(unlabeled_data)), format='csr')
    else:
      self.X_ = np.concatenate((util.cast_to_float32(X),
                                util.cast_to_float32(unlabeled_data)))
    self.gamma = (
        self.gamma if self.gamma is not None else 1.0 / X.shape[1])
    self.kernel_params = {'gamma':self.gamma, 'degree':self.degree, 'coef0':self.coef0}
    kernel_matrix = pairwise_kernels(self.X_, metric=self.kernel,
                                     filter_params=True, **self.kernel_params)
    A = np.dot(np.diag(labeled), kernel_matrix)
    if self.nu2 != 0:
      if self.kernel == 'rbf':
        laplacian_kernel_matrix = kernel_matrix
      else:
        laplacian_kernel_matrix = rbf_kernel(self.X_, gamma=self.gamma)
      laplacian_x_kernel = np.dot(graph_laplacian(
          laplacian_kernel_matrix, normed=self.normalize_laplacian), kernel_matrix)
      A += self.nu2 * laplacian_x_kernel
    y = np.concatenate((y, -np.ones((num_unlabeled,), dtype=np.float32)),
                       axis=0)
    super(LapRLSC, self).fit(A, y, class_for_unlabeled=-1)

  def predict(self, X):
    kernel_matrix = pairwise_kernels(util.cast_to_float32(X), self.X_,
                                     metric=self.kernel, filter_params=True,
                                     **self.kernel_params)
    return super(LapRLSC, self).predict(kernel_matrix)
