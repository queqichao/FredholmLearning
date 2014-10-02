import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import column_or_1d
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.externals import six
from sklearn.base import BaseEstimator, ClassifierMixin
from abc import ABCMeta
import util
from scipy.sparse import issparse, vstack


class BaseL2KernelClassifier(six.with_metaclass(ABCMeta, BaseEstimator),
                             ClassifierMixin):

  def get_label_matrix(self, y):
    return util.cast_to_float32(self._label_binarizer.fit_transform(y))

  def fit(self, kernel_matrix, y):
    self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
    Y = self.get_label_matrix(y)
    self.coef_ = np.empty(
        (Y.shape[1], kernel_matrix.shape[1]), dtype=np.float32)
    num_data = kernel_matrix.shape[0]
    kernel_matrix[np.diag_indices(num_data)] += self.nu
    cond_num = np.linalg.cond(kernel_matrix)
    for i in range(Y.shape[1]):
      y_column = Y[:, i]
      if np.isinf(cond_num):
        self.coef_[i] = np.linalg.lstsq(kernel_matrix, y_column)[0]
      else:
        self.coef_[i] = np.linalg.solve(kernel_matrix, y_column)

  def predict(self, kernel_matrix):
    scores = self.decision_function(kernel_matrix)
    if len(scores.shape) == 1:
      indices = (scores > 0).astype(np.int)
    else:
      indices = scores.argmax(axis=1)
    return self.classes_()[indices]

  def decision_function(self, kernel_matrix):
    num_centers = kernel_matrix.shape[1]
    if self.coef_.shape[1] != num_centers:
      raise ValueError("Kernel matrix has %d centers per sample; expecting %d"
                       % (num_centers, self.coef_.shape[1]))
    scores = np.dot(kernel_matrix, self.coef_.T)
    return scores.ravel() if scores.shape[1] == 1 else scores

  def classes_(self):
    return self._label_binarizer.classes_


class L2KernelClassifier(BaseL2KernelClassifier):

  def __init__(self, nu=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0):
    self.nu = nu
    self.kernel = kernel
    self.degree = degree
    self.gamma = gamma
    self.coef0 = coef0

  def fit(self, X, y):
    self.X_ = util.cast_to_float32(X)
    kernel_matrix = pairwise_kernels(
        util.cast_to_float32(X), metric=self.kernel, filter_params=True,
        gamma=self.gamma, degree=self.degree, coef0=self.coef0)
    super(L2KernelClassifier, self).fit(kernel_matrix, y)

  def predict(self, X):
    kernel_matrix = pairwise_kernels(
        util.cast_to_float32(X), self.X_, metric=self.kernel,
        filter_params=True, gamma=self.gamma, degree=self.degree,
        coef0=self.coef0)
    return super(L2KernelClassifier, self).predict(kernel_matrix)


class L2FredholmClassifier(BaseL2KernelClassifier):

  def __init__(self, nu=1.0, in_kernel=['rbf'], out_kernel='rbf', gamma=0.0):
    self.nu = nu
    self.in_kernel = in_kernel
    self.out_kernel = out_kernel
    self.gamma = gamma

  def fit(self, X, y, unlabeled_data=None):
    if issparse(X):
      self.X_ = vstack((util.cast_to_float32(X), util.cast_to_float32(unlabeled_data)), format='csr')
    else:
      self.X_ = np.concatenate((util.cast_to_float32(X), util.cast_to_float32(unlabeled_data)))
    labeled = range(X.shape[0])
    self.labeled_ = labeled
    kernel_matrix = self.fredholm_kernel(self.X_[labeled])
    super(L2FredholmClassifier, self).fit(kernel_matrix, y)

  def predict(self, X):
    kernel_matrix = self.fredholm_kernel(X, Y=self.X_[self.labeled_])
    return super(L2FredholmClassifier, self).predict(kernel_matrix)

  def fredholm_kernel(self, X, Y=None):
    in_kernel_matrix_uu = self.compute_in_kernel()
    out_kernel_matrix_xu = pairwise_kernels(
        X, self.X_, metric=self.out_kernel, filter_params=True,
        gamma=self.gamma)
    if self.out_kernel == "rbf":
      out_kernel_matrix_xu = out_kernel_matrix_xu/np.sum(out_kernel_matrix_xu, axis=1).reshape((X.shape[0],1))
    if Y is None:
      out_kernel_matrix_yu = out_kernel_matrix_xu
    else:
      out_kernel_matrix_yu = pairwise_kernels(
          Y, self.X_, metric=self.out_kernel, filter_params=True,
          gamma=self.gamma)
      if self.out_kernel == "rbf":
        out_kernel_matrix_yu = out_kernel_matrix_yu/np.sum(out_kernel_matrix_yu, axis=1).reshape((Y.shape[0],1))
    return np.dot(
        out_kernel_matrix_xu,
        np.dot(in_kernel_matrix_uu, out_kernel_matrix_yu.T))

  def compute_in_kernel(self):
    if len(self.in_kernel) == 1:
      return pairwise_kernels(self.X_, metric=self.in_kernel[0],
                              filter_params=True, gamma=self.gamma)

    for i, kernel in enumerate(self.in_kernel[:-1]):
      if i == 0:
        in_kernel_matrix = pairwise_kernels(
            self.X_, metric=kernel, filter_params=True, gamma=self.gamma)
        if kernel == "rbf":
          in_kernel_matrix = in_kernel_matrix/np.sum(in_kernel_matrix, axis=1).reshape((self.X_.shape[0],1))
      else:
        K = pairwise_kernels(self.X_, metric=kernel, filter_params=True,
                             gamma=self.gamma)
        if kernel == "rbf":
          K = K / np.sum(K, axis=1).reshape((self.X_.shape[0],1))
        in_kernel_matrix = np.dot(in_kernel_matrix, K)
    inner_kernel_matrix = pairwise_kernels(self.X_, metric=self.in_kernel[-1],
                                           filter_params=True, gamma=self.gamma)
    return np.dot(in_kernel_matrix,
                  np.dot(inner_kernel_matrix, in_kernel_matrix.T))
