from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics.scorer import check_scoring
from fredholm_kernel_learning import L2KernelClassifier
from fredholm_kernel_learning import L2FredholmClassifier
from fredholm_kernel_learning import SVMLight
from fredholm_kernel_learning import LapRLSC
from rbf_kernel_approximation import RBFKernelApprClassifier
import smtplib
from email.mime.text import MIMEText


def get_classifier(classifier):
  if classifier["name"] == 'linear-ridge':
    c = RidgeClassifier()
  elif classifier["name"] == 'SVC':
    c = SVC()
  elif classifier["name"] == "l2-SVC":
    c = L2KernelClassifier()
  elif classifier["name"] == "fredholm":
    c = L2FredholmClassifier()
  elif classifier["name"] == "TSVM":
    c = SVMLight()
  elif classifier["name"] == "Lap-RLSC":
    c = LapRLSC()
  elif classifier["name"] == "rbf_kernel_appr":
    c = RBFKernelApprClassifier()
  else:
    raise NameError('Not existing classifier: ' + classifier["name"] + '.')
  c.set_params(**classifier["params"])
  return c


def get_cv_classifier(classifier, cv):
  if classifier["name"] == 'linear-ridge':
    c = RidgeClassifier()
  elif classifier["name"] == 'SVC':
    c = SVC()
  elif classifier["name"] == "l2-SVC":
    c = L2KernelClassifier()
  elif classifier["name"] == "fredholm":
    c = L2FredholmClassifier()
  elif classifier["name"] == "TSVM":
    c = SVMLight()
  elif classifier["name"] == "Lap-RLSC":
    c = LapRLSC()
  elif classifier["name"] == "rbf_kernel_appr":
    c = RBFKernelApprClassifier()
  else:
    raise NameError('Not existing classifier: ' + classifier["name"] + '.')
  return GridSearchCV(c, classifier["params_grid"], scoring=check_scoring(c),
                      fit_params={}, n_jobs=classifier["n_jobs"], cv=cv)

def send_results(msg_string, from_addr, to_addr):
  msg = MIMEText(msg_string)
  msg['Subject'] = 'Results'
  msg['From'] = from_addr
  msg['To'] = to_addr
  s = smtplib.SMTP('localhost')
  s.sendmail(from_addr, [to_addr], msg.as_string())
  s.quit()


def evaluation_classifier(dataset, classifier, cross_validation=False,
                          n_folds=None, fit_params={}):
  if classifier["semi-supervised"]:
    fit_params["unlabeled_data"] = dataset.unlabeled_data()
  if cross_validation:
    if n_folds is None:
      raise NameError("n_folds should be specified if using cross validation")
    c = get_cv_classifier(classifier, n_folds)
    c.fit_params = fit_params
    c.fit(dataset.training_data(), dataset.training_labels())
    print(classifier["name"] + ": " + str(c.best_params_))
    classifier["params"] = c.best_params_
  else:
    c = get_classifier(classifier)
    c.fit(dataset.training_data(), dataset.training_labels(), **fit_params)
  testing_pred_labels = c.predict(dataset.testing_data())
  return len([0 for i in range(dataset.num_testing())
              if dataset.testing_labels()[i] == testing_pred_labels[i]]
             ) * 1.0 / dataset.num_testing()
