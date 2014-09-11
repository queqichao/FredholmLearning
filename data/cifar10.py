import numpy as np
import os
from sklearn.feature_extraction.image import PatchExtractor


def read_images(dataset):
  """
  Python function for import the CIFAR-10 data set as images.
  """
  if "dataset" not in dataset:
    raise NameError("dataset need to specified.")

  if dataset["dataset"] == "training":
    data = np.zeros((50000, 3073,), dtype=np.uint8)
    for i in range(5):
      fname_data = os.path.join(
          dataset["path"], 'data_batch_' + str(i + 1) + '.bin')
      with open(fname_data, 'rb') as fdata:
        data[i * 10000:(i + 1) * 10000] = np.fromfile(
            fdata, dtype=np.uint8).reshape(10000, 3073)
  elif dataset["dataset"] == "testing":
    fname_data = os.path.join(dataset["path"], 'test_batch.bin')
    with open(fname_data, 'rb') as fdata:
      data = np.fromfile(fdata, dtype=np.uint8).reshape(10000, 3073)
  else:
    raise ValueError("dataset must be 'testing' or 'training'")

  lbl = data[:, 0]
  img = np.transpose(
      data[:, 1:].reshape((data.shape[0], 3, 32, 32)), (0, 2, 3, 1))
  if "seed" in dataset:
    np.random.seed(dataset["seed"])

  if dataset["permutation"]:
    p = np.random.permutation(len(lbl))
    return np.array([img[p[k]]
                     for k in range(len(lbl))
                     if lbl[p[k]] in dataset["labels"]]), \
        np.array([lbl[p[k]]
                  for k in range(len(lbl))
                  if lbl[p[k]] in dataset["labels"]])
  else:
    return np.array([img[k]
                     for k in range(len(lbl))
                     if lbl[k] in dataset["labels"]], dtype=np.float32), \
        np.array([lbl[k]
                  for k in range(len(lbl))
                  if lbl[k] in dataset["labels"]], dtype=np.float32)


def read(dataset):
  """
  Python function for importing the CIFAR-10 data set as data points.
  """

  if "dataset" not in dataset:
    raise NameError("dataset need to specified.")

  if dataset["dataset"] == "training":
    data = np.zeros((50000, 3073,), dtype=np.uint8)
    for i in range(5):
      fname_data = os.path.join(
          dataset["path"], 'data_batch_' + str(i + 1) + '.bin')
      with open(fname_data, 'rb') as fdata:
        data[i * 10000:(i + 1) * 10000] = np.fromfile(
            fdata, dtype=np.uint8).reshape(10000, 3073)
  elif dataset["dataset"] == "testing":
    fname_data = os.path.join(dataset["path"], 'test_batch.bin')
    with open(fname_data, 'rb') as fdata:
      data = np.fromfile(fdata, dtype=np.uint8).reshape(10000, 3073)
  else:
    raise ValueError("dataset must be 'testing' or 'training'")

  lbl = data[:, 0]
  img = data[:, 1:]
  img = img.astype(np.float32) / 255
  lbl = lbl.astype(np.float32)
  if "seed" in dataset:
    np.random.seed(dataset["seed"])

  if dataset["permutation"]:
    p = np.random.permutation(len(lbl))
    return np.array([img[p[k]]
                     for k in range(len(lbl))
                     if lbl[p[k]] in dataset["labels"]]), \
        np.array([lbl[p[k]]
                  for k in range(len(lbl))
                  if lbl[p[k]] in dataset["labels"]])
  else:
    return np.array([img[k]
                     for k in range(len(lbl))
                     if lbl[k] in dataset["labels"]], dtype=np.float32), \
        np.array([lbl[k]
                  for k in range(len(lbl))
                  if lbl[k] in dataset["labels"]], dtype=np.float32)
