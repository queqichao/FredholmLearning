from .dataset import SupervisedDataSet
from .dataset import SemiSupervisedDataSet
from .dataset import SynthesizedSemiSupervisedDataSet
from .dataset import ExistingSemiSupervisedDataSet, ImageDataSet
__all__ = ['cifar10', 'mnist', 'news_group', 'SupervisedDataSet',
           'SemiSupervisedDataSet', 'SynthesizedSemiSupervisedDataSet',
           'ExistingSemiSupervisedDataSet']
