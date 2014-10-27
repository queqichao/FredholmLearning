import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import column_or_1d
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.externals import six
from sklearn.base import BaseEstimator, ClassifierMixin
from abc import ABCMeta
import util
from scipy.sparse import issparse, vstack, spdiags


class BaseL2KernelClassifier(six.with_metaclass(ABCMeta, BaseEstimator),
                             ClassifierMixin):

  def __init__(self, nu=1.0):
    self.nu = nu
    self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)

  def get_label_matrix(self, y, class_for_unlabeled):
    if class_for_unlabeled is None:
      return util.cast_to_float32(self._label_binarizer.fit_transform(y))
    else:
      tmp_Y  = util.cast_to_float32(self._label_binarizer.fit_transform(y))
      if class_for_unlabeled not in self._label_binarizer.classes_:
        raise IndexError("The class for unlabeled is not in the input labels.")
      unlabeled_idx, = np.where(self._label_binarizer.classes_ == class_for_unlabeled)
      idx = (tmp_Y[:, unlabeled_idx] == 1).ravel()
      Y = np.zeros((tmp_Y.shape[0], tmp_Y.shape[1]-1), dtype=np.float32)
      count = 0;
      for i in xrange(Y.shape[1]):
        if not i == unlabeled_idx:
          Y[:,count] = tmp_Y[:, i]
          print(Y.shape)
          print(idx.shape)
          Y[idx, count] = 0
          count += 1
      return Y

  def fit(self, kernel_matrix, y, class_for_unlabeled=None):
    Y = self.get_label_matrix(y, class_for_unlabeled)
    self.coef_ = np.empty(
        (Y.shape[1], kernel_matrix.shape[1]), dtype=np.float32)
    num_data = kernel_matrix.shape[0]
    kernel_matrix[np.diag_indices(num_data)] += self.nu
    cond_num = np.linalg.cond(kernel_matrix)
    for i in xrange(Y.shape[1]):
      y_column = Y[:, i]
      if np.isinf(cond_num):
        self.coef_[i] = np.linalg.lstsq(kernel_matrix, y_column)[0]
      else:
        self.coef_[i] = np.linalg.solve(kernel_matrix, y_column)

  def predict(self, kernel_matrix):
    scores = self.decision_function_w_kernel(kernel_matrix)
    return self.predict_w_scores(scores)

  def decision_function_w_kernel(self, kernel_matrix):
    num_centers = kernel_matrix.shape[1]
    if self.coef_.shape[1] != num_centers:
      raise ValueError("Kernel matrix has %d centers per sample; expecting %d"
                       % (num_centers, self.coef_.shape[1]))
    scores = np.dot(kernel_matrix, self.coef_.T)
    return scores.ravel() if scores.shape[1] == 1 else scores

  def classes_(self):
    return self._label_binarizer.classes_

  def predict_w_scores(self, scores):
    if len(scores.shape) == 1:
      indices = (scores > 0).astype(np.int)
    else:
      indices = scores.argmax(axis=1)
    return self.classes_()[indices]

class L2KernelClassifier(BaseL2KernelClassifier):

  def __init__(self, nu=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0):
    self.nu = nu
    self.kernel = kernel
    self.degree = degree
    self.gamma = gamma
    self.coef0 = coef0
    super(L2KernelClassifier, self).__init__(nu=self.nu)

  def fit(self, X, y):
    self.X_ = util.cast_to_float32(X)
    kernel_matrix = pairwise_kernels(
        util.cast_to_float32(X), metric=self.kernel, filter_params=True,
        gamma=self.gamma, degree=self.degree, coef0=self.coef0)
    super(L2KernelClassifier, self).fit(kernel_matrix, y)

  def predict(self, X):
    scores = self.decision_function(X)
    return self.predict_w_scores(scores)

  def decision_function(self, X):
    kernel_matrix = pairwise_kernels(
        util.cast_to_float32(X), self.X_, metric=self.kernel,
        filter_params=True, gamma=self.gamma, degree=self.degree,
        coef0=self.coef0) 
    return super(L2KernelClassifier, self).decision_function_w_kernel(kernel_matrix)

class L2FredholmClassifier(BaseL2KernelClassifier):

  def __init__(self, nu=1.0, in_kernel=['rbf'], out_kernel='rbf', gamma=0.0, normalized=True):
    self.nu = nu
    self.in_kernel = in_kernel
    self.out_kernel = out_kernel
    self.gamma = gamma
    self.normalized = normalized
    super(L2FredholmClassifier, self).__init__(nu=self.nu)

  def fit(self, X, y, unlabeled_data=None):
    if issparse(X):
      self.X_ = vstack((util.cast_to_float32(X), util.cast_to_float32(unlabeled_data)), format='csr')
    else:
      self.X_ = np.concatenate((util.cast_to_float32(X), util.cast_to_float32(unlabeled_data)))
    labeled = xrange(X.shape[0])
    self.labeled_ = labeled
    kernel_matrix = self.fredholm_kernel(self.X_[self.labeled_],
                                         semi_data=self.X_,
                                         in_kernel=self.in_kernel,
                                         out_kernel=self.out_kernel,
                                         gamma=self.gamma,
                                         normalized=self.normalized)
    super(L2FredholmClassifier, self).fit(kernel_matrix, y)
    if self.out_kernel == "linear":
      self.linear_coef = self.compute_linear_coef(self.coef_.T, X, semi_data=self.X_, in_kernel=self.in_kernel, gamma=self.gamma)

  def predict(self, X):
    if self.out_kernel == "linear":
      if issparse(X):
        scores = X*self.linear_coef.T
      else:
        scores = np.dot(X, self.linear_coef.T)
      return super(L2FredholmClassifier, self).predict_w_scores(scores)
    else:
      kernel_matrix = self.fredholm_kernel(X, Y=self.X_[self.labeled_],
                                           semi_data=self.X_,
                                           in_kernel=self.in_kernel,
                                           out_kernel=self.out_kernel,
                                           gamma=self.gamma,
                                           normalized=self.normalized)
      return super(L2FredholmClassifier, self).predict(kernel_matrix)

  @classmethod
  def compute_linear_coef(cls, kernel_coef, X, semi_data=None, in_kernel=["rbf"], gamma=1.0):
    if semi_data is None:
      semi_data = X
    in_kernel_matrix_uu = cls.compute_in_kernel(semi_data, in_kernel, gamma=gamma)
    out_kernel_matrix_ux = pairwise_kernels(semi_data, X, metric="linear")
    tmp_coef = np.dot(in_kernel_matrix_uu, np.dot(out_kernel_matrix_ux, kernel_coef))
    if issparse(semi_data):
      linear_coef = np.zeros(tmp_coef.shape[1], X.shape[1])
      for i in xrange(tmp_coef.shape[1]):
        linear_coef[i] = np.array((spdiags(tmp_coef[:, i], 0, X.shape[0], X.shape[0])*semi_data).sum(axis=0))[0]
    else:
      linear_coef = np.dot(tmp_coef.T, semi_data)
    return linear_coef

  @classmethod
  def fredholm_kernel(cls, X, Y=None, semi_data=None, in_kernel=["rbf"], out_kernel="rbf", gamma=1.0, normalized=True):
    if semi_data is None:
      semi_data = X;
    in_kernel_matrix_uu = cls.compute_in_kernel(semi_data, in_kernel, gamma=gamma)
    out_kernel_matrix_xu = pairwise_kernels(
        X, semi_data, metric=out_kernel, filter_params=True,
        gamma=gamma)
    if out_kernel == "rbf":
      if normalized:
        out_kernel_matrix_xu = out_kernel_matrix_xu/np.sum(out_kernel_matrix_xu, axis=1).reshape((X.shape[0],1))
      else:
        out_kernel_matrix_xu = out_kernel_matrix_xu/semi_data.shape[0]
    if Y is None:
      out_kernel_matrix_yu = out_kernel_matrix_xu
    else:
      out_kernel_matrix_yu = pairwise_kernels(
          Y, semi_data, metric=out_kernel, filter_params=True,
          gamma=gamma)
      if out_kernel == "rbf":
        if normalized:
          out_kernel_matrix_yu = out_kernel_matrix_yu/np.sum(out_kernel_matrix_yu, axis=1).reshape((Y.shape[0],1))
        else:
          out_kernel_matrix_yu = out_kernel_matrix_yu/semi_data.shape[0]
    return np.dot(
        out_kernel_matrix_xu,
        np.dot(in_kernel_matrix_uu, out_kernel_matrix_yu.T))

  @classmethod
  def compute_in_kernel(cls, semi_data, in_kernel, gamma=1.0):
    if len(in_kernel) == 1:
      return pairwise_kernels(semi_data, metric=in_kernel[0],
                              filter_params=True, gamma=gamma)

    for i, kernel in enumerate(in_kernel[:-1]):
      if i == 0:
        in_kernel_matrix = pairwise_kernels(
            semi_data, metric=kernel, filter_params=True, gamma=gamma)
        if kernel == "rbf":
          in_kernel_matrix = in_kernel_matrix/np.sum(in_kernel_matrix, axis=1).reshape((semi_data.shape[0],1))
      else:
        K = pairwise_kernels(semi_data, metric=kernel, filter_params=True,
                             gamma=gamma)
        if kernel == "rbf":
          K = K / np.sum(K, axis=1).reshape((semi_data.shape[0],1))
        in_kernel_matrix = np.dot(in_kernel_matrix, K)
    inner_kernel_matrix = pairwise_kernels(semi_data, metric=in_kernel[-1],
                                           filter_params=True, gamma=gamma)
    return np.dot(in_kernel_matrix,
                  np.dot(inner_kernel_matrix, in_kernel_matrix.T))
