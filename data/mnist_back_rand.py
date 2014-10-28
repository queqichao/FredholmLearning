import numpy as np

def load_mnist_back_rand(path):
  data = np.array([[np.float(i) for i in x.split('   ')[1:]]
                   for x in open(path).read().split('\n')[:-1]])
  labels = data[:, -1]
  data = data[:, :-1]
  return data, labels

def read(dataset):
  """
  Python function for importing the MNIST Background Random data set.
  """
  if "dataset" not in dataset:
    raise NameError("dataset need to specified.")

  if dataset["dataset"] == "training":
    fname = os.path.join(dataset["path"], 'mnist_background_train.amat')
  elif dataset["dataset"] == "testing":
    fname = os.path.join(dataset["path"], 'mnist_background_test.amat')
  else:
    raise ValueError("dataset must be 'testing' or 'training'")

  # Load everything in some numpy arrays
  data, labels = load_mnist_back_rand(fname)

  if "seed" in dataset:
    np.random.seed(dataset["seed"])

  if dataset["permutation"]:
    p = np.random.permutation(data.shape[0])
    return data[p], labels[p]
  else:
    return data, labels
