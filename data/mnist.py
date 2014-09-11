import os
import struct
import numpy as np

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read_images(digits, dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    indices = [k for k in range(img.shape[0]) if lbl[k] in digits]

    return img[indices], lbl[indices]

def read(dataset):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if not "dataset" in dataset:
      raise NameError("dataset need to specified.")

    if dataset["dataset"] == "training":
        fname_img = os.path.join(dataset["path"], 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(dataset["path"], 'train-labels-idx1-ubyte')
    elif dataset["dataset"] == "testing":
        fname_img = os.path.join(dataset["path"], 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(dataset["path"], 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows*cols)
    img = img.astype(np.float32)/255
    lbl = lbl.astype(np.float32)
    if "seed" in dataset:
      np.random.seed(dataset["seed"])
    
    if dataset["permutation"]:
      p = np.random.permutation(len(lbl))
      return np.array([img[p[k]] for k in range(len(lbl)) if lbl[p[k]] in dataset["labels"]]),np.array([lbl[p[k]] for k in range(len(lbl)) if lbl[p[k]] in dataset["labels"]])
    else:
      return np.array([img[k] for k in range(len(lbl)) if lbl[k] in dataset["labels"]], dtype=np.float32),np.array([lbl[k] for k in range(len(lbl)) if lbl[k] in dataset["labels"]], dtype=np.float32)

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()
