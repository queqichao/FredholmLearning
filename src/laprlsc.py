import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.graph import graph_laplacian
from sklearn.preprocessing import LabelBinarizer

class LapRLSC(BaseEstimator, ClassifierMixin):
  def __init__(self, rbf_gamma=None, normalize_laplacian=False, nu1=1.0, nu2=None):
    self.rbf_gamma = rbf_gamma
    self.normalize_laplacian = normalize_laplacian
    self.nu1 = nu1
    if nu2 is None:
      self.nu2 = nu1
    else:
      self.nu2 = nu2
    self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)

  @profile
  def fit(self, X, y):
    num_data = X.shape[0]
    labeled = y!=-1
    unlabeled = y==-1
    num_labeled = y[labeled].shape[0]
    num_unlabeled = num_data - num_labeled
    Y_labeled = self._label_binarizer.fit_transform(y[labeled])
    self.num_classes_ = Y_labeled.shape[1]
    Y_unlabeled = np.zeros((num_unlabeled, self.num_classes_,))
    Y = np.zeros((num_data, self.num_classes_))
    Y[labeled] = Y_labeled
    Y[unlabeled] = Y_unlabeled
    self.X_ = X

    self.coef_ = np.empty((Y.shape[1], num_data))
    self.rbf_gamma_ = (self.rbf_gamma if self.rbf_gamma is not None else 1.0/X.shape[1])
    kernel_matrix_ = rbf_kernel(X, gamma=self.rbf_gamma_)
    kernel_matrix_ = 0.5*(kernel_matrix_+kernel_matrix_.T)
    laplacian = graph_laplacian(kernel_matrix_, normed=self.normalize_laplacian)
    I = np.identity(num_data)
    J = np.diag(labeled.astype(np.double)) 
    for i in range(Y.shape[1]):
      y_column = Y[:,i]
      if self.nu2 != 0:
        self.coef_[i] = np.linalg.solve(np.dot(J, kernel_matrix_) + self.nu1*I + self.nu2*np.dot(laplacian, kernel_matrix_), y_column)
      else:
        self.coef_[i] = np.linalgo.solve(np.dot(J, kernel_matrix_) + self.nu1*I, y_column)

  def predict(self, X):
    kernel_matrix_ = rbf_kernel(X, self.X_, gamma=self.rbf_gamma_)
    scores = np.dot(kernel_matrix_, self.coef_.T)
    scores = scores.ravel() if scores.shape[1]==1 else scores
    if len(scores.shape) == 1:
      indices = (scores > 0).astype(np.int)
    else:
      indices = scores.argmax(axis=1) 
    return self.classes_()[indices]  

  def classes_(self):
    return self._label_binarizer.classes_
