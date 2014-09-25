from data import cifar10
from data import mnist
import json
import argparse
from sklearn.decomposition import PCA
import numpy as np
from data import dataset
from sklearn.linear_model import RidgeClassifier
import os
from fredholm_kernel_learning import L2KernelClassifier
from fredholm_kernel_learning import classifier_help

parser = argparse.ArgumentParser()
parser.add_argument('config_file', help='Config file.', type=str)
args = parser.parse_args()

config_file = args.config_file

config = json.loads(open(args.config_file).read())
data_config = config["dataset"]
if data_config["name"] == "mnist":
  images, labels = mnist.read_images(data_config)
elif data_config["name"] == "cifar10":
  images, labels = cifar10.read_images(data_config)

image_data = dataset.ImageDataSet(images.astype(np.uint8))

classifiers = config["classifiers"]
results = [0, 0, 0, 0]
data = image_data.to_array().astype(np.float)
data = dataset.SupervisedDataSet(data, labels, config["num_training"])

classifiers[0]["n_jobs"] = 1
results[0] = classifier_help.evaluation_classifier(data, classifiers[0], True, 5, fit_params={"kernel_training_data": image_data.to_array().astype(np.float)})

classifiers[1]["n_jobs"] = 1
results[1] = classifier_help.evaluation_classifier(data, classifiers[0], True, 5, fit_params={"kernel_training_data": image_data.to_array().astype(np.float)})

classifiers[2]["n_jobs"] = 1
results[2] = classifier_help.evaluation_classifier(data, classifiers[0], True, 5, fit_params={"kernel_training_data": image_data.to_array().astype(np.float)})

classifiers[3]["n_jobs"] = 1
results[3] = classifier_help.evaluation_classifier(data, classifiers[1], True, 5)

result_str = ''
for j, result in enumerate(results):
  result_str += classifiers[j]["name"] + ': ' + str(result) + '\n'

result_str += "\n\nThe result json is:\n"
result_str += json.dumps(config) + '\n'
classifier_help.send_results(
    result_str,
    getpass.getuser() + '@' + socket.gethostname() + '.cse.ohio-state.edu',
    'que@cse.ohio-state.edu')
