import numpy as np
from scipy.sparse import csr_matrix
import os
from sklearn.feature_extraction.text import TfidfTransformer


def read(groups, dataset='all', path='.'):
    TRAINING_DOC_NUM = 11269
    TESTING_DOC_NUM = 7505
    VOC_SIZE = 61188
    if dataset == "all":
      training_data_tuples = get_data_tuples(os.path.join(path, 'train.data'))
      training_labels = get_labels(os.path.join(path, 'train.label'))
      testing_data_tuples = get_data_tuples(os.path.join(path, 'test.data'))
      testing_labels = get_labels(os.path.join(path, 'test.label'))
      testing_data_tuples[:,0] += TRAINING_DOC_NUM
      data_tuples = np.concatenate((training_data_tuples,testing_data_tuples), axis=0)
      labels = np.concatenate((training_labels, testing_labels), axis=0)
      DOC_NUM = TRAINING_DOC_NUM+TESTING_DOC_NUM
    elif dataset == "training":
      data_tuples = get_data_tuples(os.path.join(path, 'train.data'))
      labels = get_labels(os.path.join(path, 'train.label'))
      DOC_NUM = TRAINING_DOC_NUM
    elif dataset == "testing":
      data_tuples = get_data_tuples(os.path.join(path, 'test.data'))
      labels = get_labels(os.path.join(path, 'test.label'))
      DOC_NUM = TESTING_DOC_NUM
    else:
      raise ValueError("dataset must be 'all' or 'testing' or 'training', not " + dataset + "")
    wc_sparse = csr_matrix(([x[2] for x in data_tuples],
                            ([x[0]-1 for x in data_tuples], [x[1]-1 for x in data_tuples])),
                           shape=(DOC_NUM, VOC_SIZE))
    indices = [k for k in range(DOC_NUM) if labels[k] in groups]
    tfidf_transformer = TfidfTransformer()
    tfidf = np.asarray(tfidf_transformer.fit_transform(wc_sparse[indices]).todense())
    return tfidf, labels[indices]

def get_data_tuples(data_path):
  return np.array([[int(i) for i in x.split(' ')] for x in open(data_path).read().split('\n')[:-1]])
 
def get_labels(labels_path):
  return np.array([int(x) for x in open(labels_path).read().split('\n')[:-1]])
