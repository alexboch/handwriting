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

    @abc.abstractmethod
    def MapToVectorLabels(self,points,pointwise_labels):
        """
        :param pointwise_labels:Метки такие, что одной точке соответствует одна метка
        :return:
        """
        pass

    @abc.abstractmethod
    def MapToPointLabels(self,points,pointwise_labels):
        """
        :param pointwise_labels:
        :return:
        """
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


