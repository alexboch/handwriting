from constants import *

def num_filter(points,labels):
    result_labels=[]
    result_points=[]
    for i in range(len(labels)):
        if labels[i][CHAR_KEY].isdigit():#Оставлять только цифры в строковом представлении
            result_labels.append(labels[i])
            result_points.append(points[i])
    return result_points,result_labels


