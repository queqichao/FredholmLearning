from data import dataset
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import RidgeClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_approximation import Nystroem
from scipy.optimize import fmin_l_bfgs_b
from rbf_kernel_approximation import kernel_approximation_help
import struct
import util

class KMeansRegression(BaseEstimator, TransformerMixin):
  def __init__(self, dictionary=None, dictionary_file=None, gamma=None, eta=None, dictionary_parm=None, subsample=None):
    self.gamma = gamma
    self.dictionary_file = dictionary_file
    self.dictionary = dictionary
    self.eta = eta
    self.dictionary_parm = dictionary_parm
    self.subsample = subsample

  def fit(self, X):
    if self.dictionary is None:
      if self.dictionary_file is None:
        self.learn_dictionary(X)
      else:
        self.dictionary = self.dict_from_bin_file(self.dictionary_file)
    d = self.dictionary.shape[0]
    if self.subsample is None:
      data_kernel = rbf_kernel(X, Y=self.dictionary, gamma=2*self.gamma)
      true_kernel = rbf_kernel(X, gamma=self.gamma)
    else:
      p = np.random.permutation(X.shape[0])
      data_kernel = rbf_kernel(X[p[:self.subsample]], Y=self.dictionary, gamma=2*self.gamma)
      true_kernel = rbf_kernel(X[p[:self.subsample]], gamma=self.gamma)
    bounds = [(0, None) for i in range(d)]
    self.eta, f, _ = fmin_l_bfgs_b(_loss_fun, np.zeros((d,)), fprime=_loss_fun_prime, args=(true_kernel, data_kernel), bounds=bounds, maxiter=1000)
    self.eta = self.eta.astype(np.float)
    print('The loss function at minimum is: '+str(f))
    print('Eta is:')
    print(self.eta)

  def transform(self, X, y=None):
    kernel_matrix = rbf_kernel(X, Y=self.dictionary, gamma=2*self.gamma)
    return kernel_matrix * np.sqrt(self.eta)

  def to_bin_file(self, filename):
    fid = open(filename, 'wb')
    fid.write(struct.pack('d', self.gamma))
    fid.write(struct.pack('II', self.dictionary.shape[0], self.dictionary.shape[1]))
    fid.write(self.dictionary.tostring())
    fid.write(self.eta.tostring())
    fid.close()

  @staticmethod
  def from_bin_file(filename):
    fid = open(filename, 'rb')
    gamma, = struct.unpack('d', fid.read(8))
    dict_r, dict_c, = struct.unpack('II', fid.read(8))
    dictionary = np.fromstring(fid.read(8*dict_r*dict_c), dtype=np.float).reshape((dict_r, dict_c))
    eta = np.fromstring(fid.read(8*dict_r), dtype=np.float)
    return KMeansRegression(dictionary, gamma, eta)

  def dict_to_bin_file(self, filename):
    fid = open(filename, 'wb')
    fid.write(struct.pack('II', self.dictionary.shape[0], self.dictionary.shape[1]))
    fid.write(self.dictionary.tostring())
    fid.close()

  @staticmethod
  def dict_from_bin_file(filename):
    fid = open(filename, 'rb')
    dict_r, dict_c, = struct.unpack('II', fid.read(8))
    dictionary = np.fromstring(fid.read(8*dict_r*dict_c), dtype=np.float).reshape((dict_r, dict_c))
    return dictionary

  def learn_dictionary(self, data):
    if "preprocessing" in self.dictionary_parm:
      data = kernel_approximation_help.preprocessing(data, self.dictionary_parm["preprocessing"])
    
    kmeans = kernel_approximation_help.get_kmeans(self.dictionary_parm["cluster_parm"])
    kmeans.fit(data)
    self.dictionary = kmeans.cluster_centers_
    
    if "cut_off" in self.dictionary_parm:
      mx = self.dictionary_parm["cut_off"]
      mn = -self.dictionary_parm["cut_off"]
      self.dictionary = (
          (util.cut_off_values(self.dictionary, mn, mx) - mn)
          / (mx - mn) * 255
      ).astype(np.float)

  def dictionary(self):
    return self.dictionary

  def visualize_dictionary(self, image_size, save_path, cmap, matrix_size):
    dictionary_img = dataset.ImageDataSet.from_array(
          self.dictionary.astype(np.uint8), image_size)
    
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pyplot as plt
    import os
 
    for i in range(np.int(self.dictionary.shape[0]/matrix_size)):
      util.show_images_matrix(dictionary_img.images()[i*matrix_size:(i+1)*matrix_size], plt, cmap)
      plt.savefig(os.path.join(save_path, 'centroids_' + str(i) + '.png'))

class RBFKernelApprClassifier(RidgeClassifier):
  
  def __init__(self, kernel_approximating_method=None, dictionary_file=None, gamma=None, eta=None, dictionary_parm=None, subsample=None, n_components=None, random_state=None, alpha=None, fit_intercept=False):
    self.kernel_approximating_method = kernel_approximating_method
    self.dictionary_file = dictionary_file
    self.gamma = gamma
    self.eta = eta
    self.dictionary_parm = dictionary_parm
    self.subsample = subsample
    self.n_components = n_components
    self.random_state = random_state
    if self.kernel_approximating_method is not None:
      self._init_kernel_approximator()
    self.alpha = alpha
    self.fit_intercept = fit_intercept
    super(RBFKernelApprClassifier, self).__init__(alpha=alpha, fit_intercept=fit_intercept)

  def set_params(self, **params):
    super(RBFKernelApprClassifier, self).set_params(**params)
    self._init_kernel_approximator()
    sub_params = {}
    for name in self.kernel_approximator._get_param_names():
      if name in params:
        sub_params[name] = params[name]
    self.kernel_approximator.set_params(**sub_params)
    return self 

  def _init_kernel_approximator(self):
    if self.kernel_approximating_method is None:
      raise TypeError("kernel approximating method has not been initialized.")
    if self.kernel_approximating_method == "kmeans_appr":
      self.kernel_approximator = KMeansRegression(dictionary_file=self.dictionary_file, gamma=self.gamma, eta=self.eta, dictionary_parm=self.dictionary_parm, subsample=self.subsample)
    elif self.kernel_approximating_method == "random_fourier":
      self.kernel_approximator = RBFSampler(gamma=self.gamma, n_components=self.n_components, random_state=self.random_state)
    elif self.kernel_approximating_method == "nystroem":
      self.kernel_approximator = Nystroem(kernel="rbf", gamma=self.gamma, n_components=self.n_components, random_state=self.random_state)
    else:
      raise TypeError("Not existing kernel approximation method: "+self.kernel_approximating_method+".")
 
  def fit(self, X, y, kernel_training_data=None):
    self.kernel_approximator.fit(kernel_training_data)
    X = self.kernel_approximator.transform(X)
    super(RBFKernelApprClassifier, self).fit(X, y)
  
  def predict(self, X):
    X = self.kernel_approximator.transform(X)
    return super(RBFKernelApprClassifier, self).predict(X)

def _loss_fun(eta, true_kernel, data_kernel):
  feature_map = data_kernel * np.sqrt(eta)
  n = data_kernel.shape[0]
  return (np.linalg.norm(true_kernel - np.dot(feature_map, feature_map.T)) ** 2) / (n ** 2)

def _loss_fun_prime(eta, true_kernel, data_kernel):
  n = data_kernel.shape[0]
  d = data_kernel.shape[1]
  eta_grad = np.zeros(eta.shape)
  feature_map = data_kernel * np.sqrt(eta)
  error = true_kernel - np.dot(feature_map, feature_map.T)
  for i in range(d):
    k_v = data_kernel[:,i]
    eta_grad[i] = np.sum(np.multiply(error, -np.outer(k_v, k_v))) / (n ** 2)
  return eta_grad
