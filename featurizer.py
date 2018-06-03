from abc import ABC
from abc import  abstractstaticmethod

class Featurizer(ABC):

    @abstractstaticmethod
    def get_features(self,points):
        pass