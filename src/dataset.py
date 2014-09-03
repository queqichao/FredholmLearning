import numpy as np
import mnist
from matplotlib import pyplot as plt

class SemiSupervisedDataSet:
  def training_data(self):
    return self._training_data

  def training_labels(self):
    return self._training_labels

  def num_training(self):
    return self._training_data.shape[0]

  def testing_data(self):
    return self._testing_data

  def testing_labels(self):
    return self._testing_labels

  def num_testing(self):
    return self._testing_data.shape[0]

  def unlabeled_data(self):
    return self._unlabeled_data

  def num_unlabeled(self):
    return self._unlabeled_data.shape[0]

  def semi_supervised_data(self):
    return np.concatenate((self._training_data, self._unlabeled_data), axis=0)

  def semi_supervised_labels(self):
    return np.concatenate((self._training_labels, -np.ones((self.num_unlabeled(),))), axis=0)

  def visualize(self, indices, colors=[]):
    internal_colors = ['b','g','r','c','m','y','k','w']
    data = np.concatenate((self._training_data, self._testing_data), axis=0)
    label = np.concatenate((self._training_labels, self._testing_labels),axis=0)
    if len(np.unique(label)) > len(internal_colors) and len(colors)==0:
      raise NameError("Number of classes is more that the number of internal colors. You need to specify the color explicitly.")
    if len(colors)==0:
      colors = internal_colors
    for i, ll in enumerate(np.unique(label)):
      plt.plot([x[indices[0]] for (l,x) in zip(label, data) if l==ll], [x[indices[1]] for (l,x) in zip(label, data) if l==ll], '.', color=colors[i])
    plt.axis('equal')
    plt.show()

class SynthesizedSemiSupervisedDataSet(SemiSupervisedDataSet):
  def __init__(self, dataset_config):
    self._name = dataset_config["name"]
    self._num_training = dataset_config["num_training"]
    self._num_testing = dataset_config["num_testing"]
    self._num_unlabeled = dataset_config["num_unlabeled"]
    self._dim = dataset_config["dim"]
    self._noise_scale = dataset_config["noise_scale"]
    self._training_data, self._training_labels = self._synthesize_data(
      self._num_training, self._dim, self._noise_scale, self._name)
    self._testing_data, self._testing_labels = self._synthesize_data(
      self._num_testing, self._dim, self._noise_scale, self._name)
    self._unlabeled_data,_ = self._synthesize_data(
      self._num_unlabeled, self._dim, self._noise_scale, self._name)

  def _generate_circle_data(cls, num_data, dim, noise_scale):
    X = np.random.uniform(0, 1, num_data)
    data = np.array([[np.cos(2*np.pi*x), np.sin(2*np.pi*x)] for x in X])
    data = np.concatenate(
      ([x+np.random.normal(0, noise_scale, (2,)) for x in data],
       np.random.normal(0, noise_scale, (num_data, dim-2,))), axis=1)
    return data
  
  def _generate_circle_linear_data(cls, num_data, dim, noise_scale):
    data = cls._generate_circle_data(num_data, dim, noise_scale)
    label = np.zeros((num_data,), dtype=float)
    for i in range(num_data):
      if data[i][0] >= 0:
        label[i] = 1
      else:
        label[i] = 0
    return data,label
  
  def _generate_circle_quad_data(cls, num_data, dim, noise_scale):
    data = cls._generate_circle_data(num_data, dim, noise_scale)
    label = np.zeros((num_data,), dtype=float)
    for i in range(num_data):
      if (data[i][0] >= 0 and data[i][1] >= 0) or (data[i][0] < 0 and data[i][1] < 0):
        label[i] = 1
      else:
        label[i] = 0
    return data,label
  
  def _generate_two_line(cls, num_data, dim, noise_scale):
    X1 = np.random.uniform(-1,1, num_data)
    X2 = np.concatenate((np.ones((num_data/2,)), -np.ones((num_data/2,))))+np.random.normal(0, noise_scale, (num_data,))
    X_res = np.random.normal(0, noise_scale, (num_data, dim-2,))
    data = np.concatenate((np.array([[x1, x2] for (x1,x2) in zip(X1,X2)]), X_res), axis=1) 
    return data
  
  def _generate_two_line_w_cluster_assumption(cls, num_data, dim, noise_scale):
    data = cls._generate_two_line(num_data, dim, noise_scale)
    label = np.zeros((num_data,), dtype=float)
    label[data[:,1]>=0] = 1
    label[data[:,1]<0] = 0
    return data, label
  
  def _generate_two_line_wo_cluster_assumption(cls, num_data, dim, noise_scale):
    data = cls._generate_two_line(num_data, dim, noise_scale)
    label = np.zeros((num_data,), dtype=float)
    label[data[:,0]>=0] = 1
    label[data[:,0]<0] = 0
    return data, label
  
  def _synthesize_data(cls, num_data, dim, noise_scale, data_set_name):
    if data_set_name == 'circle-linear':
      data,labels = cls._generate_circle_linear_data(num_data, dim, noise_scale)
    elif data_set_name == 'circle-quad':
      data,labels = cls._generate_circle_quad_data(num_data, dim, noise_scale)
    elif data_set_name == 'twoline-cluster':
      data,labels = cls._generate_two_line_w_cluster_assumption(num_data, dim, noise_scale)
    elif data_set_name == 'twoline-nocluster':
      data,labels = cls._generate_two_line_wo_cluster_assumption(num_data, dim, noise_scale)
    else:
      raise NameError('Not existing data_set_name: '+data_set_name+'.')
    return data,labels

class ExistingSemiSupervisedDataSet(SemiSupervisedDataSet):
  def __init__(self, dataset):
    if dataset["name"] == 'mnist_train':
      self._load_mnist_train(dataset)

  def _load_mnist_train(self, dataset):
    data,labels = mnist.read(dataset["digits"], path=dataset["path"])
    data = np.array([x.flatten() for x in data])
    num_data = data.shape[0]
    if dataset["permutation"]:
      if "seed" in dataset:
        np.random.seed(dataset["seed"])
      p = np.random.permutation(num_data)
      data = data[p]
      labels = labels[p]
    data = data*1.0/255
    labels = labels*1.0
    if dataset["noise_scale"] > 0:
      data = ExistingSemiSupervisedDataSet._add_noise(data, dataset["noise_scale"])
    self._num_training = dataset["num_training"]
    self._num_unlabeled = dataset["num_unlabeled"]
    self._training_data = data[0:self._num_training]
    self._training_labels = labels[0:self._num_training]
    self._unlabeled_data = data[self._num_training:self._num_training+self._num_unlabeled]
    self._testing_data = data[self._num_training:num_data]
    self._testing_labels = labels[self._num_training:num_data]
   
  def _add_noise(cls, X, noise_scale):
    num_data = X.shape[0]
    dim = X.shape[1]
    return X+np.random.normal(0, noise_scale, (num_data, dim,))
