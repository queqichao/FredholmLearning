import numpy as np
from data import SynthesizedSemiSupervisedDataSet
from data import ExistingSemiSupervisedDataSet
import argparse
from fredholm_kernel_learning import classifier_help
import util
import json
import getpass
import socket

parser = argparse.ArgumentParser()
parser.add_argument('config_file', help='Config file', type=str)
parser.add_argument(
    '--repeat', help='Whether or not to repeat the experiment', type=int,
    default=1)
parser.add_argument(
    '--plot_data', help='Whether or not to plot the data', action='store_true')
parser.add_argument(
    '--cv_config_file', help='Config file after cv', type=str, default="")
parser.add_argument(
    '--n_jobs', help='Number of jobs running in parallel', type=int, default=2)
args = parser.parse_args()

config_file = args.config_file
config = util.convert(json.loads(open(config_file).read()))
cv_config_file = args.cv_config_file
plot_data = args.plot_data
repeat = args.repeat
n_jobs = args.n_jobs

dataset_config = config["dataset"]
classifiers = config["classifiers"]
cross_validation = config["cross_validation"]
n_folds = config["n_folds"]

results = [[] for x in xrange(len(classifiers))]

for i in xrange(repeat):
  if dataset_config["type"] == 'synthesized':
    dataset = SynthesizedSemiSupervisedDataSet(dataset_config)
  elif dataset_config["type"] == 'existing':
    dataset = ExistingSemiSupervisedDataSet(dataset_config)
  else:
    raise NameError(
        "Data set type: " + dataset_config["type"] + " does not exist.")

  if plot_data:
    dataset.visualize([0, 1])

  for j, classifier in enumerate(classifiers):
    classifier["n_jobs"] = n_jobs
    if cross_validation:
      results[j].append(classifier_help.evaluation_classifier(
          dataset, classifier, cross_validation=cross_validation, n_folds=n_folds, fit_params = {}))
    else:
      results[j].append(
          classifier_help.evaluation_classifier(dataset, classifier))
  print('result at '+str(i)+'-th repetition:')
  print([r[-1] for r in results])

if cv_config_file != "":
  open(cv_config_file, 'w').write(json.dumps(config))

result_str = ''
for j, result in enumerate(results):
  result_str += classifiers[j]["name"] + ': ' + ', '.join([str(x) for x in result]) + "\nmean: " + str(np.mean(result)) + "; std:" + str(np.std(result)) + '\n'

result_str += "\n\nThe result json is:\n"
result_str += json.dumps(config) + '\n'
classifier_help.send_results(
    result_str,
    getpass.getuser() + '@' + socket.gethostname() + '.cse.ohio-state.edu',
    'que@cse.ohio-state.edu')
