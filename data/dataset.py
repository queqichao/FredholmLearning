import numpy as np
from data import mnist
from data import news_group
from scipy.sparse import issparse
import util
from sklearn.feature_extraction.image import PatchExtractor
import struct


class SupervisedDataSet(object):

  def __init__(self, data, labels, num_training):
    self._data = data
    num_data, self._dim = data.shape
    self._labels = labels
    self._num_training = num_training
    self._num_testing = num_data-num_training
    self._training_indices = np.array(range(self._num_training))
    self._testing_indices = np.array(range(self._num_training, num_data))

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

  def visualize(self, indices, colors=[]):
    internal_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    data = np.concatenate((self.training_data(), self.testing_data()), axis=0)
    label = np.concatenate(
        (self.training_labels(), self.testing_labels()), axis=0)
    if len(np.unique(label)) > len(internal_colors) and len(colors) == 0:
      raise NameError(
          "Number of classes is more that the number of internal colors. "
          "You need to specify the color explicitly.")
    if len(colors) == 0:
      colors = internal_colors
    from matplotlib import pyplot as plt
    for i, ll in enumerate(np.unique(label)):
      plt.plot([x[indices[0]] for (l, x) in zip(label, data) if l == ll],
               [x[indices[1]] for (l, x) in zip(label, data) if l == ll],
               '.', color=colors[i])
    plt.axis('equal')
    plt.show()


class SemiSupervisedDataSet(SupervisedDataSet):

  def __init__(self, data, labels, num_training, num_unlabeled):
    self._data = data
    self._labels = labels
    num_data, self._dim = data.shape
    self._num_training = num_training
    self._num_unlabeled = num_unlabeled
    self._num_testing = num_data-num_training-num_unlabeled
    self._training_indices = np.array(range(self._num_training))
    self._unlabeled_indices = np.array(
        range(self._num_training, self._num_training + self._num_unlabeled))
    self._testing_indices = np.array(
        range(self._num_training + self._num_unlabeled, num_data))

  def unlabeled_data(self):
    return self._data[self._unlabeled_indices]

  def num_unlabeled(self):
    return self._num_unlabeled

  def semi_supervised_data(self):
    return self._data[np.concatenate(
        (self._training_indices, self._unlabeled_indices),
        axis=0)]

  def semi_supervised_labels(self):
    return np.concatenate(
        (self.training_labels(),
         -np.ones((self.num_unlabeled(),), dtype=np.float32)), axis=0)


class SynthesizedSemiSupervisedDataSet(SemiSupervisedDataSet):

  def __init__(self, dataset_config):
    self._name = dataset_config["name"]
    self._num_training = dataset_config["num_training"]
    self._num_testing = dataset_config["num_testing"]
    self._num_unlabeled = dataset_config["num_unlabeled"]
    dim = dataset_config["dim"]
    self._noise_scale = dataset_config["noise_scale"]
    num_data = self._num_training + self._num_unlabeled + self._num_testing
    data, labels = self._synthesize_data(num_data,
                                         dim,
                                         self._noise_scale,
                                         self._name)
    super(SynthesizedSemiSupervisedDataSet, self).__init__(data, labels, self._num_training, self._num_unlabeled)

  @classmethod
  def _generate_circle_data(cls, num_data, dim, noise_scale):
    X = np.random.uniform(0, 1, num_data)
    data = np.array(
        [[np.cos(2 * np.pi * x), np.sin(2 * np.pi * x)] for x in X])
    data = np.concatenate(
        ([x + np.random.normal(0, noise_scale, (2,)) for x in data],
         np.random.normal(0, noise_scale, (num_data, dim - 2,))), axis=1)
    return data

  @classmethod
  def _generate_circle_linear_data(cls, num_data, dim, noise_scale):
    data = cls._generate_circle_data(num_data, dim, noise_scale)
    label = np.zeros((num_data,), dtype=float)
    for i in range(num_data):
      if data[i][0] >= 0:
        label[i] = 1
      else:
        label[i] = 0
    return data, label

  @classmethod
  def _generate_circle_quad_data(cls, num_data, dim, noise_scale):
    data = cls._generate_circle_data(num_data, dim, noise_scale)
    label = np.zeros((num_data,), dtype=float)
    for i in range(num_data):
      if (data[i][0] >= 0 and data[i][1] >= 0) or \
              (data[i][0] < 0 and data[i][1] < 0):
        label[i] = 1
      else:
        label[i] = 0
    return data, label

  @classmethod
  def _generate_two_line(cls, num_data, dim, noise_scale):
    X1 = np.random.uniform(-1, 1, num_data)
    X2 = np.concatenate((np.ones((num_data / 2,)), -np.ones((num_data / 2,)))
                        ) + np.random.normal(0, noise_scale, (num_data,))
    X_res = np.random.normal(0, noise_scale, (num_data, dim - 2,))
    data = np.concatenate(
        (np.array([[x1, x2] for (x1, x2) in zip(X1, X2)]), X_res), axis=1)
    return data

  @classmethod
  def _generate_two_line_w_cluster_assumption(
          cls, num_data, dim, noise_scale):
    data = cls._generate_two_line(num_data, dim, noise_scale)
    label = np.zeros((num_data,), dtype=float)
    label[data[:, 1] >= 0] = 1
    label[data[:, 1] < 0] = 0
    return data, label

  @classmethod
  def _generate_two_line_wo_cluster_assumption(
          cls, num_data, dim, noise_scale):
    data = cls._generate_two_line(num_data, dim, noise_scale)
    label = np.zeros((num_data,), dtype=float)
    label[data[:, 0] >= 0] = 1
    label[data[:, 0] < 0] = 0
    return data, label

  @classmethod
  def _synthesize_data(cls, num_data, dim, noise_scale, data_set_name):
    if data_set_name == 'circle-linear':
      data, labels = cls._generate_circle_linear_data(
          num_data, dim, noise_scale)
    elif data_set_name == 'circle-quad':
      data, labels = cls._generate_circle_quad_data(
          num_data, dim, noise_scale)
    elif data_set_name == 'twoline-cluster':
      data, labels = cls._generate_two_line_w_cluster_assumption(
          num_data, dim, noise_scale)
    elif data_set_name == 'twoline-nocluster':
      data, labels = cls._generate_two_line_wo_cluster_assumption(
          num_data, dim, noise_scale)
    else:
      raise NameError('Not existing data_set_name: ' + data_set_name + '.')
    return data, labels


class ExistingSemiSupervisedDataSet(SemiSupervisedDataSet):

  def __init__(self, dataset):
    num_training = dataset["num_training"]
    num_unlabeled = dataset["num_unlabeled"]
    if dataset["name"] == 'mnist':
      data, labels = mnist.read(dataset)
    elif dataset["name"] == '20news_group':
      data, labels = news_group.read(dataset)
    super(ExistingSemiSupervisedDataSet, self).__init__(data, labels, num_training, num_unlabeled)
    if dataset["noise_scale"] > 0:
      self._add_noise(dataset["noise_scale"])

  def _add_noise(self, noise_scale):
    if issparse(self._data):
      self._data.data += util.cast_to_float32(
          np.random.normal(0, noise_scale, (len(self._data.data),)))
    else:
      num_data, dim = self._data.shape
      self._data += util.cast_to_float32(
          np.random.normal(0, noise_scale, (num_data, dim,)))


class ImageDataSet:

  def __init__(self, images):
    if not images.dtype == np.uint8:
      raise TypeError("The dtype of image must be numpy.uint8")
    self._images = images
    self._num_images, self._i_h, self._i_w = self._images.shape[:3]
    if len(self._images.shape) >= 4:
      self._n_channels = self._images.shape[3]
    else:
      self._n_channels = 1

  def is_greyscale(self):
    return self._n_channels == 1

  def images(self):
    return self._images

  def num_images(self):
    return self._num_images

  def extract_patches(self, patch_size, max_patches=None, random_state=None):
    patch_extractor = PatchExtractor(patch_size=patch_size, max_patches=np.int(
        max_patches / self.num_images()), random_state=random_state)
    return patch_extractor.transform(self._images).astype(np.uint8)

  def to_array(self):
    return self._images.reshape(
        self._num_images, self._i_h * self._i_w * self._n_channels)

  @classmethod
  def from_array(cls, data, image_size):
    vec_len = image_size[0] * image_size[1]
    num_data = data.shape[0]
    i_h = image_size[0]
    i_w = image_size[1]
    if len(image_size) == 3:
      vec_len *= image_size[2]

    if not data.shape[1] == vec_len:
      raise NameError("Vector lenght in data must match the image sizes.")

    if len(image_size) == 3:
      n_channels = image_size[2]
      return cls(data.reshape((num_data, i_h, i_w, n_channels)))
    else:
      return cls(data.reshape((num_data, i_h, i_w)))

  def to_bin_file(self, file_name):
    fid = open(file_name, 'wb')
    fid.write(struct.pack(
        'IIII', self._num_images, self._i_h, self._i_w, self._n_channels))
    fid.write(self._images.tostring())
    fid.close()

  @classmethod
  def from_bin_file(cls, file_name):
    fid = open(file_name, 'rb')
    num_images, i_h, i_w, n_channels = struct.unpack('IIII', fid.read(16))
    s = fid.read()
    fid.close()
    if n_channels == 1:
      return cls(np.fromstring(s, dtype=np.uint8)
          .reshape((num_images, i_h, i_w)))
    else:
      return cls(np.fromstring(s, dtype=np.uint8)
          .reshape((num_images, i_h, i_w, n_channels)))
