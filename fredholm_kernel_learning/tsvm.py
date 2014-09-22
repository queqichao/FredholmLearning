import numpy as np
from abc import ABCMeta
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, ClassifierMixin
import svmlight as svm
from scipy.sparse import issparse


class SVMLight(BaseEstimator, ClassifierMixin):

  def __init__(self, C=None, costratio=1.0, kernel='rbf', poly_degree=3,
               rbf_gamma=0.0, coef_lin=0.0, coef_const=0.0):
    self.C = C
    self.costratio = costratio
    self.kernel = kernel
    self.poly_degree = poly_degree
    self.rbf_gamma = rbf_gamma
    self.coef_lin = coef_lin
    self.coef_const = coef_const

  def get_params(self, deep=True):
    return {'C': self.C,
            'costratio': self.costratio,
            'kernel': self.kernel,
            'poly_degree': self.poly_degree,
            'rbf_gamma': self.rbf_gamma,
            'coef_lin': self.coef_lin,
            'coef_const': self.coef_const}

  def set_params(self, **params):
    if 'C' in params:
      self.C = params['C']
    if 'costratio' in params:
      self.costratio = params['costratio']
    if 'kernel' in params:
      self.kernel = params['kernel']
    if 'poly_degree' in params:
      self.poly_degree = params['poly_degree']
    if 'rbf_gamma' in params:
      self.rbf_gamma = params['rbf_gamma']
    if 'coef_lin' in params:
      self.coef_lin = params['coef_lin']
    if 'coef_const' in params:
      self.coef_const = params['coef_const']

    return self

  def fit(self, X, y, unlabeled_data=None):
    num_data = X.shape[0]+unlabeled_data.shape[0]
    num_unlabeled = unlabeled_data.shape[0]
    labeled = range(X.shape[0])
    unlabeled = range(X.shape[0], num_data)
    X = np.concatenate((X, unlabeled_data))
    self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
    Y_labeled = self._label_binarizer.fit_transform(y)
    self.num_classes_ = Y_labeled.shape[1]
    Y_unlabeled = np.zeros(
        (num_unlabeled, self.num_classes_,), dtype=np.float32)
    Y = np.zeros((num_data, self.num_classes_), dtype=np.float32)
    Y[labeled] = Y_labeled
    Y[unlabeled] = Y_unlabeled
    self.model_ = []
    for i in range(self.num_classes_):
      y_column = Y[:, i]
      self.model_.append(
          svm.learn(self.__data2docs(X, y_column),
                    type='classification'.encode()))

  def predict(self, X):
    num_data = X.shape[0]
    scores = np.zeros((num_data, self.num_classes_,), dtype=np.float32)
    for i in range(self.num_classes_):
      scores[:, i] = svm.classify(
          self.model_[i],
          self.__data2docs(X, np.zeros((num_data,), dtype=np.float32)))
    if self.num_classes_ == 1:
      indices = (scores.ravel() > 0).astype(np.int)
    else:
      indices = scores.argmax(axis=1)
    return self.classes_()[indices]

  def classes_(self):
    return self._label_binarizer.classes_

  def __data2docs(self, X, y):
    return [(l, self.__vector2words(x)) for (x, l) in zip(X, y)]

  def __vector2words(self, v):
    if issparse(v):
      return [(i, x) for i, x in zip(v.nonzero()[1], v.data)]
    else:
      return [(i + 1, x) for i, x in enumerate(v) if x != 0.0]
