from PointsAndRectangles import *
import numpy as np

class FeatureVector:
    NUM_FEATURES=6

    def __init__(self,rel_x:float,rel_y:float,direction_x:float,direction_y:float,length:int,num_intersections:int):
        self.rel_x=rel_x
        self.rel_y=rel_y
        self.direction_y=direction_y
        self.length=length
        self.num_intersections=num_intersections
        pass

    def to_array(self):
        arr=np.zeros(shape=(self.NUM_FEATURES))


        return