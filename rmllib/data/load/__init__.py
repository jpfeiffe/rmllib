'''
Builds various datasets, primarily intended for loading actual datasets but sometimes
will augment iid data with generated edges
'''

from .base import Dataset
from .boston import BostonMedians
