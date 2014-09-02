import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics.scorer import check_scoring
from l2_kernel_classifier import L2KernelClassifier
from l2_kernel_classifier import L2FredholmClassifier
from tsvm import SVMLight
from laprlsc import LapRLSC

def generate_circle_data(num_data, dim, noise_scale):
  X = np.random.uniform(0, 1, num_data)
  data = np.array([[np.cos(2*np.pi*x), np.sin(2*np.pi*x)] for x in X])
  data = np.concatenate(
    ([x+np.random.normal(0, noise_scale, (2,)) for x in data],
     np.random.normal(0, noise_scale, (num_data, dim-2,))), axis=1)
  return data

def generate_circle_linear_data(num_data, dim, noise_scale):
  data = generate_circle_data(num_data, dim, noise_scale)
  label = np.zeros((num_data,), dtype=float)
  for i in range(num_data):
    if data[i][0] >= 0:
      label[i] = 1
    else:
      label[i] = 0
  return data,label

def generate_circle_quad_data(num_data, dim, noise_scale):
  data = generate_circle_data(num_data, dim, noise_scale)
  label = np.zeros((num_data,), dtype=float)
  for i in range(num_data):
    if (data[i][0] >= 0 and data[i][1] >= 0) or (data[i][0] < 0 and data[i][1] < 0):
      label[i] = 1
    else:
      label[i] = 0
  return data,label

def generate_two_line(num_data, dim, noise_scale):
  X1 = np.random.uniform(-1,1, num_data)
  X2 = np.concatenate((np.ones((num_data/2,)), -np.ones((num_data/2,))))+np.random.normal(0, noise_scale, (num_data,))
  X_res = np.random.normal(0, noise_scale, (num_data, dim-2,))
  data = np.concatenate((np.array([[x1, x2] for (x1,x2) in zip(X1,X2)]), X_res), axis=1) 
  return data

def generate_two_line_w_cluster_assumption(num_data, dim, noise_scale):
  data = generate_two_line(num_data, dim, noise_scale)
  label = np.zeros((num_data,), dtype=float)
  label[data[:,1]>=0] = 1
  label[data[:,1]<0] = 0
  return data, label

def generate_two_line_wo_cluster_assumption(num_data, dim, noise_scale):
  data = generate_two_line(num_data, dim, noise_scale)
  label = np.zeros((num_data,), dtype=float)
  label[data[:,0]>=0] = 1
  label[data[:,0]<0] = 0
  return data, label

def plot_data(data, label, colors=[]):
  internal_colors = ['b','g','r','c','m','y','k','w']
  if len(np.unique(label)) > len(internal_colors) and len(colors)==0:
    raise NameError("Number of classes is more that the number of internal colors. You need to specify the color explicitly.")
  if len(colors)==0:
    colors = internal_colors
  for i, ll in enumerate(np.unique(label)):
    plt.plot([x[0] for (l,x) in zip(label, data) if l==ll], [x[1] for (l,x) in zip(label, data) if l==ll], '.', color=colors[i])
  plt.axis('equal')
  plt.show()

def generate_data(num_data, dim, noise_scale, data_set_name):
  if data_set_name == 'circle-linear':
    data,labels = generate_circle_linear_data(num_data, dim, noise_scale)
  elif data_set_name == 'circle-quad':
    data,labels = generate_circle_quad_data(num_data, dim, noise_scale)
  elif data_set_name == 'twoline-cluster':
    data,labels = generate_two_line_w_cluster_assumption(num_data, dim, noise_scale)
  elif data_set_name == 'twoline-nocluster':
    data,labels = generate_two_line_wo_cluster_assumption(num_data, dim, noise_scale)
  else:
    raise NameError('Not existing data_set_name: '+data_set_name+'.')
  
  return data,labels

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
