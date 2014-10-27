# Learning With Fredholm Kernel

### System Requirement:
  python2.7, Numpy, scikit-learn

### Installation
  1. Install [numpy](http://www.numpy.org) and [scikit-learn](http://http://scikit-learn.org/stable/install.html).
  2. Fork this rep. And using git to download the source code.

     ```
       git clone https://github.com/YOUR_USERNAME/FredholmLearning
     ```

  3. (Optional) Enter `fredholm_kernel_learning/pysvmlight` and install the SvmLight for python.

     ```
       cd fredholm_kernel_learning/pysvmlight

       python2.7 setup.py build

       python2.7 setup.py install
     ``` 

### Usage
  For every experiment, a json file need to provided as examples in `config/`. Use script `classification.py` to run the experiment.

### Example Of Usage
  Generate a synthesize data: a circle corrupted with Gaussian noise, partitioned into two classes.

  ```
    python2.7 classification.py config/circle-quad.json --repeat 1 --n_jobs 1
  ```

[1] "Learning with Fredholm Kernels",
    Qichao Que, Mikhail Belkin, Yusu Wang, 
    Advances in Neural Information Processing Systems (NIPS), 2014.

