import abc
import numpy as np

class FeaturePointsSetBase(abc.ABC):


    def __init__(self,points):
        self.CreateFeatures(points)
        pass


    @abc.abstractmethod
    def GetFeatureVector(self,point):

        pass

    @abc.abstractmethod
    def GetNumFeatures(self):
        pass

    @abc.abstractmethod
    def _createFeatures(self,points):
        pass

    def CreateFeatures(self,points):
        """
        :param points:Массив точек слова, точка--кортеж (x,y) типа float
        :return:
        """
        self.features=self._createFeatures(points)

    pass

    def SaveToFile(self):

        pass


    def GetFeatures(self):
        return self.features


