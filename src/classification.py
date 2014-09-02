import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.cross_validation import KFold
from sklearn.base import clone
import argparse
import util
import json

parser = argparse.ArgumentParser()
parser.add_argument('config_file', help='Config file', type=str)
parser.add_argument('--repeat', help='Whether or not to repeat the experiment', type=int, default=1)
parser.add_argument('--plot_data', help='Whether or not to plot the data', action='store_true')
parser.add_argument('--cv_config_file', help='Config file after cv', type=str, default="")
parser.add_argument('--n_jobs', help='Number of jobs running in parallel', type=int, default=2)
args = parser.parse_args()

config_file = args.config_file
config = json.loads(open(config_file).read())
cv_config_file = args.cv_config_file
plot_data = args.plot_data
repeat = args.repeat
n_jobs = args.n_jobs

num_training = config["num_training"]
num_unlabled = config["num_unlabeled"]
num_testing = config["num_testing"]
dim = config["dim"]
noise_scale = config["noise_scale"]
data_set_name = config["data_set_name"]
classifiers = config["classifiers"]
cross_validation = config["cross_validation"]
n_folds = config["n_folds"]

results = [[] for x in range(len(classifiers))]

for i in range(repeat):
  training_data, training_labels = util.generate_data(num_training, dim, noise_scale, data_set_name)
  unlabeled_data,_ = util.generate_data(num_unlabled, dim, noise_scale, data_set_name)
  testing_data, testing_labels = util.generate_data(num_testing, dim, noise_scale, data_set_name) 

  if plot_data:
    util.plot_data(np.concatenate((training_data, testing_data), axis=0), np.concatenate((training_labels, testing_labels),axis=0))

  for j, classifier in enumerate(classifiers):
    classifier["n_jobs"] = n_jobs
    if cross_validation:
      c = util.get_cv_classifier(classifier, n_folds)
    else:
      c = util.get_classifier(classifier)
    if classifier["semi-supervised"]:
      c.fit(np.concatenate((training_data, unlabeled_data), axis=0), np.concatenate((training_labels, -np.ones((num_unlabled,))), axis=0))
    else:
      c.fit(training_data, training_labels)
    if cross_validation:
      print(classifier["name"]+": "+str(c.best_params_))
      classifier["params"] = c.best_params_
    testing_pred_labels = c.predict(testing_data)
    results[j].append(len([0 for i in range(num_testing) if testing_labels[i]==testing_pred_labels[i]])*1.0/num_testing)

if cv_config_file != "":
  open(cv_config_file, 'w').write(json.dumps(config))

for j, result in enumerate(results):
  print(classifiers[j]["name"]+': '+', '.join([str(x) for x in result])+"\nmean: "+str(np.mean(result))+"; std:"+str(np.std(result)))
