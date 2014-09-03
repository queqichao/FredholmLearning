from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics.scorer import check_scoring
from l2_kernel_classifier import L2KernelClassifier
from l2_kernel_classifier import L2FredholmClassifier
from tsvm import SVMLight
from laprlsc import LapRLSC
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
  else:
    raise NameError('Not existing classifier: '+classifier["name"]+'.')
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
  else:
    raise NameError('Not existing classifier: '+classifier["name"]+'.') 
  return GridSearchCV(c, classifier["params_grid"], scoring=check_scoring(c), fit_params={}, n_jobs=classifier["n_jobs"], cv=cv)

def send_results(msg_string, from_addr, to_addr):
  msg = MIMEText(msg_string)
  msg['Subject'] = 'Results'
  msg['From'] = from_addr
  msg['To'] = to_addr
  s = smtplib.SMTP('localhost')
  s.sendmail(from_addr, [to_addr], msg.as_string())
  s.quit()
  
