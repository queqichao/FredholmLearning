import numpy as np
from scipy.sparse import csr_matrix
import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD


def read(dataset):
  TRAINING_DOC_NUM = 11269
  TESTING_DOC_NUM = 7505
  VOC_SIZE = 61188
  if "dataset" not in dataset:
    raise NameError("dataset needs to be specified.")

  if dataset["dataset"] == "all":
    training_data_tuples = get_data_tuples(
        os.path.join(dataset["path"], 'train.data'))
    training_labels = get_labels(os.path.join(dataset["path"], 'train.label'))
    testing_data_tuples = get_data_tuples(
        os.path.join(dataset["path"], 'test.data'))
    testing_labels = get_labels(os.path.join(dataset["path"], 'test.label'))
    testing_data_tuples[:, 0] += TRAINING_DOC_NUM
    data_tuples = np.concatenate(
        (training_data_tuples, testing_data_tuples), axis=0)
    labels = np.concatenate((training_labels, testing_labels), axis=0)
    DOC_NUM = TRAINING_DOC_NUM + TESTING_DOC_NUM
  elif dataset["dataset"] == "training":
    data_tuples = get_data_tuples(os.path.join(dataset["path"], 'train.data'))
    labels = get_labels(os.path.join(dataset["path"], 'train.label'))
    DOC_NUM = TRAINING_DOC_NUM
  elif dataset["dataset"] == "testing":
    data_tuples = get_data_tuples(os.path.join(dataset["path"], 'test.data'))
    labels = get_labels(os.path.join(dataset["path"], 'test.label'))
    DOC_NUM = TESTING_DOC_NUM
  else:
    raise ValueError(
        "dataset must be 'all' or 'testing' or 'training', not " +
        dataset["dataset"])
  labels *= 1.0
  num_data = sum(1 for k in range(DOC_NUM) if labels[k] in dataset["labels"])

  wc_sparse = csr_matrix((
      [data_tuples[i][2] * 1.0
       for i in range(len(data_tuples))
       if labels[data_tuples[i][0] - 1] in dataset["labels"]],
      ([data_tuples[i][0] - 1
        for i in range(len(data_tuples))
        if labels[data_tuples[i][0] - 1] in dataset["labels"]],
       [data_tuples[i][1] - 1
        for i in range(len(data_tuples))
        if labels[data_tuples[i][0] - 1] in dataset["labels"]])),
      dtype=np.float32)
  nz_rows = np.unique(wc_sparse.nonzero()[0])
  wc_sparse = wc_sparse[nz_rows]
  labels = labels[nz_rows]
  tfidf_transformer = TfidfTransformer()
  tfidf = tfidf_transformer.fit_transform(wc_sparse)
  if dataset["SVD"]:
    svd = TruncatedSVD(n_components=dataset["SVD_components"])

  if "seed" in dataset:
    np.random.seed(dataset["seed"])

  if dataset["permutation"]:
    p = np.random.permutation(num_data)
    if dataset["SVD"]:
      data = svd.fit_transform(tfidf[p])
    else:
      data = tfidf[p]
    labels = labels[p]
  else:
    if dataset["SVD"]:
      data = svd.fit_transform(tfidf)
    else:
      data = tfidf
  return data, labels


def get_data_tuples(data_path):
  return np.array([[int(i) for i in x.split(' ')]
                   for x in open(data_path).read().split('\n')[:-1]])


def get_labels(labels_path):
  return np.array([int(x) for x in open(labels_path).read().split('\n')[:-1]])
