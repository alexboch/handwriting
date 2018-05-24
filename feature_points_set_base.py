import abc
import numpy as np

class FeaturePointsSetBase(abc.ABC):


    def __init__(self,points=None):
        if points is not None:
            self.CreateFeatures(points)
        pass



    @abc.abstractmethod
    def GetNumFeatures(self):
        pass

    @abc.abstractmethod
    def _createFeatures(self,points):
        pass

    def CreateFeatures(self, word_points):
        """
        :param word_points:Массив точек слова, точка--кортеж (x,y) типа float
        :return:
        """
        self.features=self._createFeatures(word_points)

    pass

    def SaveToFile(self):

        pass


    def GetFeatures(self):
        return self.features


