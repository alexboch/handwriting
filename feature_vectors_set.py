from feature_points_set_base import *
import utils

class FeatureVectorsSet(FeaturePointsSetBase):


    def GetNumFeatures(self):
        return 5

    def _createFeatures(self,points):
        features_list=[]
        bounds=utils.get_bounds(points)
        min_x=bounds.left
        avg_y=np.average(np.sum(points,1))
        vectors=utils.get_vectors_from_points(points,normalize_vectors=False)#Будет len(points)-1 векторов
        for i in range(len(points)-1):#Посчитать относительные координаты
            pt=points[i]
            rel_x=pt[0]-min_x#x Относительно начала
            rel_y=pt[1]-avg_y#y относительно среднего
            x_dir=vectors[i][0]#Направление по x
            y_dir=vectors[i][1]#Направление по y
            length=np.linalg.norm(vectors[i])#Длина вектора
            features_list.append((rel_x,rel_y,x_dir,y_dir,length))
        return features_list
