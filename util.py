import numpy as np
from scipy.sparse import issparse

def cast_to_float32(X):
  if issparse(X):
    X.data = np.float32(X.data)
  else:
    X = np.float32(X)
  return X


def contrast_normalization(X, bias=3, copy=True):
  means = np.mean(X, axis=1)
  stds = np.std(X, axis=1) + bias
  if copy:
    return [(x - m) / s for (x, m, s) in zip(X, means, stds)]
  else:
    for i in range(len(X)):
      X[i] = (X[i] - means[i]) / stds[i]
    return X


def show_images_matrix(images, ax):
  num_images = images.shape[0]
  ROW = np.int(np.sqrt(num_images))
  COL = np.int(num_images / ROW)

  if not ROW * COL == num_images:
    raise NameError("Input images cannot be displayed in a squre matrix.")
  i_h, i_w = images[0].shape[:2]
  if len(images[0].shape) == 3:
    is_greyscale = False
    n_channels = images[0].shape[2]
  else:
    is_greyscale = True
  if is_greyscale:
    image_matrix = np.zeros((ROW * (i_h + 1), COL * (i_w + 1)), dtype=np.uint8)
  else:
    image_matrix = np.zeros(
        (ROW * (i_h + 1), COL * (i_w + 1), n_channels), dtype=np.uint8)

  for i in range(num_images):
    r = np.int(i / COL)
    c = i % COL
    if is_greyscale:
      image_matrix[
          r * (i_h + 1):(r + 1) * (i_h + 1) - 1,
          c * (i_w + 1):(c + 1) * (i_w + 1) - 1] = images[i]
    else:
      image_matrix[
          r * (i_h + 1):(r + 1) * (i_h + 1) - 1,
          c * (i_w + 1):(c + 1) * (i_w + 1) - 1, :] = images[i]

  if is_greyscale:
    imgplot = ax.imshow(image_matrix, cmap=mpl.cm.Greys)
  else:
    imgplot = ax.imshow(image_matrix)
  imgplot.set_interpolation('nearest')


def cut_off_values(X, mn, mx):
  indices = X < mn
  X[indices] = mn
  indices = X > mx
  X[indices] = mx
  return X
