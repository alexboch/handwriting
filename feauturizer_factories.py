from feature_points_set_base import *
from feature_vectors_set import *

def default_ft_factory(points):
    return FeatureVectorsSet(points)

def vectors_ft_factory(points):
    return FeatureVectorsSet(points)