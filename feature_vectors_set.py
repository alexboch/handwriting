from feature_points_set_base import *
from sklearn.preprocessing import normalize
import utils


class FeatureVectorsSetFull(FeaturePointsSetBase):

    def GetNumFeatures(self):
        return 5




    def MapToPointLabels(self, points, vector_labels):
        return NotImplemented


    def MapToVectorLabels(self,points,pointwise_labels):
        mapped_labels=[]
        l=len(points)
        for i in range(l-1):#Векторов на 1 меньше, чем точек
            if i!=l-2:#Считаем, что метка последнего вектора равна метке последней точки
                label=pointwise_labels[i]
            else:
                label=pointwise_labels[i+1]
            mapped_labels.append(label)
        return mapped_labels



    def _createFeatures(self,points):
        features_list=[]
        bounds=utils.get_bounds(points)
        min_x=bounds.left
        avg_y=np.average(np.sum(points,1))
        vectors=utils.get_vectors_from_points(points,normalize_vectors=False)#Будет len(points)-1 векторов
        normalized_vectors=normalize(vectors)
        for i in range(len(points)-1):#Посчитать относительные координаты
            pt=points[i]
            rel_x=pt[0]-min_x#x Относительно начала
            rel_y=pt[1]-avg_y#y относительно среднего
            x_dir=normalized_vectors[i][0]#Направление по x
            y_dir=normalized_vectors[i][1]#Направление по y
            length=np.linalg.norm(vectors[i])#Длина вектора
            features_list.append((rel_x,rel_y,x_dir,y_dir,length))
        return features_list
