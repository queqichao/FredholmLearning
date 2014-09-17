from data import cifar10
from data import mnist
import json
import argparse
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import numpy as np
from data import dataset
import util

parser = argparse.ArgumentParser()
parser.add_argument('config_file', help='Config file.', type=str)
args = parser.parse_args()

config_file = args.config_file

config = json.loads(open(args.config_file).read())
data_config = config["dataset"]
if data_config["name"] == "mnist":
  images, _ = mnist.read_images(data_config)
elif data_config["name"] == "cifar10":
  images, _ = cifar10.read_images(data_config)

image_data = dataset.ImageDataSet(images.astype(np.uint8))
if data_config["sample_patches"]:
  patches = dataset.ImageDataSet(
      image_data.extract_patches(
          (data_config["patch_h"], data_config["patch_w"]),
          max_patches=data_config["max_patches"]))
else:
  patches = image_data
data = patches.to_array().astype(np.float)
if data_config["contrast_normalization"]:
  data = util.contrast_normalization(data, bias=3, copy=False)

if data_config["whiten"]:
  M = np.mean(data, axis=0)
  U,s,V = np.linalg.svd(data-M, full_matrices=False)
  var = (s ** 2) / data.shape[0]
  data = np.dot(data-M, np.dot(V.T, np.dot(np.diag(1/(var+0.1)), V)))

if data_config["cluster_parm"]["method"] == "mini-batch":
  kmeans = MiniBatchKMeans(n_clusters=data_config["cluster_parm"]["n_clusters"],
                           max_iter=data_config["cluster_parm"]["max_iter"],
                           batch_size=data_config["cluster_parm"]["batch_size"])
else:
  kmeans = KMeans(n_clusters=data_config["cluster_parm"]["n_clusters"],
                  max_iter=data_config["cluster_parm"]["max_iter"])
kmeans.fit(data)
centroids = kmeans.cluster_centers_

mx = data_config["cut_off"]
mn = -data_config["cut_off"]
centroids_img = (
    (util.cut_off_values(centroids, mn, mx) - mn)
    / (mx - mn) * 255
).astype(np.uint8)

if data_config["is_greyscale"]:
  centroids_img = dataset.ImageDataSet.from_array(
      centroids_img, (data_config["patch_h"], data_config["patch_w"]))
else:
  centroids_img = dataset.ImageDataSet.from_array(
      centroids_img, (data_config["patch_h"], data_config["patch_w"], 3))
centroids_img.to_bin_file(data_config['name']+'_centroids_img.bin')

import matplotlib as mpl
mpl.use('Agg')  
from matplotlib import pyplot as plt

for i in range(np.int(data_config["cluster_parm"]["n_clusters"]/100)):
  util.show_images_matrix(centroids_img.images()[i*100:(i+1)*100], plt)
  plt.savefig(data_config['name'] + '_centroids_' + str(i) + '.png')
