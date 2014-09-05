import numpy as np
import mnist
import news_group
from matplotlib import pyplot as plt

class SemiSupervisedDataSet:
  def training_data(self):
    return self._data[self._training_indices]

  def training_labels(self):
    return self._labels[self._training_indices]

  def num_training(self):
    return self._num_training

  def testing_data(self):
    return self._data[self._testing_indices]

  def testing_labels(self):
    return self._labels[self._testing_indices]

  def num_testing(self):
    return self._num_testing

  def unlabeled_data(self):
    return self._data[self._unlabeled_indices]

  def num_unlabeled(self):
    return self._num_unlabeled

  @profile
  def semi_supervised_data(self):
    return np.concatenate((self.training_data(), self.unlabeled_data()), axis=0)

  def semi_supervised_labels(self):
    return np.concatenate((self.training_labels(), -np.ones((self.num_unlabeled(),))), axis=0)

  def visualize(self, indices, colors=[]):
    internal_colors = ['b','g','r','c','m','y','k','w']
    data = np.concatenate((self.training_data(), self.testing_data()), axis=0)
    label = np.concatenate((self.training_labels(), self.testing_labels()),axis=0)
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
    num_data = self._num_training+self._num_unlabeled+self._num_testing
    self._data, self._labels = self._synthesize_data(num_data,
                                       self._dim,
                                       self._noise_scale,
                                       self._name)
    self._training_indices = np.array(range(self._num_training))
    self._unlabeled_indices = np.array(range(self._num_training, self._num_training+self._num_unlabeled))
    self._testing_indices = np.array(range(self._num_training+self._num_unlabeled, num_data))
    
  def _generate_circle_data(self, num_data, dim, noise_scale):
    X = np.random.uniform(0, 1, num_data)
    data = np.array([[np.cos(2*np.pi*x), np.sin(2*np.pi*x)] for x in X])
    data = np.concatenate(
      ([x+np.random.normal(0, noise_scale, (2,)) for x in data],
       np.random.normal(0, noise_scale, (num_data, dim-2,))), axis=1)
    return data
  
  def _generate_circle_linear_data(self, num_data, dim, noise_scale):
    data = self._generate_circle_data(num_data, dim, noise_scale)
    label = np.zeros((num_data,), dtype=float)
    for i in range(num_data):
      if data[i][0] >= 0:
        label[i] = 1
      else:
        label[i] = 0
    return data,label
  
  def _generate_circle_quad_data(self, num_data, dim, noise_scale):
    data = self._generate_circle_data(num_data, dim, noise_scale)
    label = np.zeros((num_data,), dtype=float)
    for i in range(num_data):
      if (data[i][0] >= 0 and data[i][1] >= 0) or (data[i][0] < 0 and data[i][1] < 0):
        label[i] = 1
      else:
        label[i] = 0
    return data,label
  
  def _generate_two_line(self, num_data, dim, noise_scale):
    X1 = np.random.uniform(-1,1, num_data)
    X2 = np.concatenate((np.ones((num_data/2,)), -np.ones((num_data/2,))))+np.random.normal(0, noise_scale, (num_data,))
    X_res = np.random.normal(0, noise_scale, (num_data, dim-2,))
    data = np.concatenate((np.array([[x1, x2] for (x1,x2) in zip(X1,X2)]), X_res), axis=1) 
    return data
  
  def _generate_two_line_w_cluster_assumption(self, num_data, dim, noise_scale):
    data = self._generate_two_line(num_data, dim, noise_scale)
    label = np.zeros((num_data,), dtype=float)
    label[data[:,1]>=0] = 1
    label[data[:,1]<0] = 0
    return data, label
  
  def _generate_two_line_wo_cluster_assumption(self, num_data, dim, noise_scale):
    data = self._generate_two_line(num_data, dim, noise_scale)
    label = np.zeros((num_data,), dtype=float)
    label[data[:,0]>=0] = 1
    label[data[:,0]<0] = 0
    return data, label
  
  def _synthesize_data(self, num_data, dim, noise_scale, data_set_name):
    if data_set_name == 'circle-linear':
      data,labels = self._generate_circle_linear_data(num_data, dim, noise_scale)
    elif data_set_name == 'circle-quad':
      data,labels = self._generate_circle_quad_data(num_data, dim, noise_scale)
    elif data_set_name == 'twoline-cluster':
      data,labels = self._generate_two_line_w_cluster_assumption(num_data, dim, noise_scale)
    elif data_set_name == 'twoline-nocluster':
      data,labels = self._generate_two_line_wo_cluster_assumption(num_data, dim, noise_scale)
    else:
      raise NameError('Not existing data_set_name: '+data_set_name+'.')
    return data,labels

class ExistingSemiSupervisedDataSet(SemiSupervisedDataSet):
  @profile
  def __init__(self, dataset):
    self._num_training = dataset["num_training"]
    self._num_unlabeled = dataset["num_unlabeled"]
    if dataset["name"] == 'mnist':
      self._data, self._labels = mnist.read(dataset)
    elif dataset["name"] == '20news_group':
      self._data, self._labels = news_group.read(dataset)

    if dataset["noise_scale"] > 0:
      self._add_noise(self._data, dataset["noise_scale"])
    self._split_dataset()

  def _split_dataset(self):
    num_data = self._data.shape[0]
    self._training_indices = np.array(range(0, self._num_training))
    self._unlabeled_indices = np.array(range(self._num_training, self._num_training+self._num_unlabeled))
    self._testing_indices = np.array(range(self._num_training,num_data))
    self._num_testing = self._testing_indices.shape[0]

  def _permutation(self, data, labels, dataset):
    num_data = data.shape[0]
    if "seed" in dataset:
      np.random.seed(dataset["seed"])
    p = np.random.permutation(num_data)
    data = data[p]
    labels = labels[p]
    return data, labels
 
  def _add_noise(self, X, noise_scale):
    num_data = X.shape[0]
    dim = X.shape[1]
    X += np.random.normal(0, noise_scale, (num_data, dim,))
