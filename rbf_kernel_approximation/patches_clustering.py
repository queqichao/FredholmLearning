from data import cifar10
from data import mnist
import json
import argparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
from data import dataset
import util

parser = argparse.ArgumentParser()
parser.add_argument('config_file', help='Config file', type=str)
args = parser.parse_args()

config = json.loads(open(args.config_file).read())
data_config = config["dataset"]
if data_config["name"] == "mnist":
  images, _ = mnist.read_images(data_config)
elif data_config["name"] == "cifar10":
  images, _ = cifar10.read_images(data_config)

image_data = dataset.ImageDataSet(images)
patches = dataset.ImageDataSet(
    image_data.extract_patches(
        (data_config["patch_h"], data_config["patch_w"]),
        max_patches=data_config["max_patches"]))
data = patches.to_array().astype(np.float)
data = util.contrast_normalization(data, bias=10, copy=False)
pca = PCA(n_components=data_config["pca_components"], whiten=True)
pca.fit(data)
X = pca.transform(data)

kmeans = KMeans(n_clusters=data_config["n_clusters"])
kmeans.fit(X)
centroids = kmeans.cluster_centers_

if data_config["inv_whitening"]:
  centroid_patches = pca.inverse_transform(centroids)
else:
  centroid_patches = np.dot(centroids, pca.components_) + pca.mean_

mx = 1.5
mn = -1.5
new_data = (
    (util.cut_off_values(centroid_patches, mn, mx) - mn)
    / (mx - mn) * 255
).astype(np.uint8)

if data_config["is_greyscale"]:
  new_patches = dataset.ImageDataSet.from_array(
      new_data, (data_config["patch_h"], data_config["patch_w"]))
else:
  new_patches = dataset.ImageDataSet.from_array(
      new_data, (data_config["patch_h"], data_config["patch_w"], 3))
util.show_images_matrix(
    new_patches.images()[:100],
    save_path=data_config['name'] + '_centroids.png')
