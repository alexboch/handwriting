import tensorflow as tf
import pandas as pd
import numpy as np
import os
import constants
from operator import methodcaller
from sklearn.preprocessing import normalize



class LabelsAlphabet:
    """
    Содержит методы преобразования между числовыми метками и буквенными
    """
    def __init__(self,characters=None):
        self.int_to_char_dict={}#Словарь для преобразования из числовой метки в буквенную
        self.char_to_int_dict={}#Словарь для преобразования из буквенной метки в числовую
        for i in np.arange(len(characters)):
            self.int_to_char_dict[i]=characters[i]
            self.char_to_int_dict[characters[i]]=i


    def label_to_int(self,char_label):
        return self.char_to_int_dict[char_label]


    def int_label_to_char(self,int_label):
        return self.int_to_char_dict[int_label]

    def decode_numeric_labels(self,int_labels):
        return [self.int_label_to_char(x) for x in int_labels]

    def encode_char_labels(self,char_labels):
        return [self.label_to_int(x) for x in char_labels]

class WordData:#TODO:Добавить координаты точек
    """
    Класс слова, содержащий список точек и меток
    """
    
    def __init__(self):
        self.point_list=[]
        self.labels_list=[]
        self.text=""
    
        
class DataHelper:
    
    #Словарь с точками слов и метками
    words_dict={}
    def __init__(self,labels_alphabet,labels_map_function=None):
        self.labels_alphabet=labels_alphabet
        self.labels_map_function=labels_map_function
        return

    LastCode=1105#Код буквы ё, последней в UTF-8
    FirstCode=1040#Код буквы


    def get_words_list(self):
        wl=list(self.words_dict.values())
        flattened_list=[item for sublist in wl for item in sublist]
        return flattened_list

    @staticmethod
    def get_vectors_from_points(points):
        vectors=[]#список векторов
        if len(points)==1:
            points.append(points[0])#Чтобы получился вектор из одной точки
        for i in np.arange(len(points)-1):
            p1=points[i]#текущая точка
            p2=points[i+1]#следующая точка
            v=[p2[0]-p1[0],p2[1]-p1[1]]
            vectors.append(v)
        vectors=normalize(vectors)  # нормализовать векторы
        return vectors

    @staticmethod
    def filter_nans(points,labels):
        result_labels=[]
        result_points=[]
        for i in np.arange(len(labels)):
            if labels[i]!=None:
                result_labels.append(labels[i])
                result_points.append(points[i])
        return result_points,result_labels

    def load_lds(self,filename):
        """
        Добавляет в словарь точки слов из файла
        """
        raw_data=pd.read_json(filename,encoding='utf8')
        transposed_data=raw_data.T
        labeled_words=[]
        words_data=[]
        #Создать ключи словаря из слов всех текстов
        
        #words_dict=dict.fromkeys(np.hstack(transposed_data['TextWords'].ravel()))
        
        for text in transposed_data.iterrows():#Цикл по всем текстам текущего файла
            pl=text[1][['PointLists','Labels']].dropna()#выбрать  списки точек и соответствующие им списки меток
            #words_dict=dict.fromkeys(text_words)
            num_words=len(text[1].TextWords)
            tmp_words_data=[]
            for i in np.arange(num_words):
                wd=WordData()
                tmp_words_data.append(wd)
            #tmp_words_data=[WordData() for i in np.arange(num_words)]#временный список данных слов для текущего текста
            #

            is_labeled=False
            for i in np.arange(len(pl.PointLists)):#Цикл по всем спискам точек
                #points_list=pl.PointLists[i]
                #labels_list=pl.Labels[i]
                points_list,labels_list=DataHelper.filter_nans(pl.PointLists[i],pl.Labels[i])
                if len(points_list)>0:#Если есть хоть одна метка
                    if self.labels_map_function is not None:
                        labels_list=self.labels_map_function(labels_list)
                    points_list=list(map(methodcaller("split",","),points_list))#разделить координаты на x и y
                    #map(lambda p: p.split(","),points_list)
                    points_list=list(map(lambda x:list(map(float, x)),points_list))#превратить координаты в числа
                    vectors=DataHelper.get_vectors_from_points(points_list)
                    for j in np.arange(len(vectors)):#Цикл по всем точкам списка
                        vector=vectors[j]
                        label=labels_list[j]
                        if label is not None:#Если не нулевая метка
                            is_labeled=True
                            char_label=label['Item1']
                            #integer_label=self.label_to_int(char_label)#Букву в число
                            integer_label=self.labels_alphabet.label_to_int(char_label)
                            word_index=label['Item2']#индекс слова в списке
                            tmp_words_data[word_index].point_list.append(vector)#сохранить данные слова по ключу
                            tmp_words_data[word_index].labels_list.append(integer_label)
                            print(word_index)
                            assert(text[1].TextWords[word_index]!='')
                            tmp_words_data[word_index].text=text[1].TextWords[word_index]#Задать строку текста
                        
            if is_labeled:#сохранить данные, только если в тексте есть метки
                words_data.extend(filter(lambda w:w.text!='', tmp_words_data))
        for wd in words_data:#пройти по всем словам и сохранить данные в словарь слов по всем текстам
            if wd.text not in self.words_dict:#если в словаре нет данных для такого словаште
                assert(wd.text!='')
                self.words_dict[wd.text]=[]
            #wd.labels_list=DataLoader.insert_blank_labels(wd.labels_list)
            #if self.labels_map_function is not None:
            #    wd.labels_dict=self.labels_map_function(wd.labels_list)
            self.words_dict[wd.text].append(wd)#сохранить данные для этого слова               
    pass

    #Загрузить все размеченные тексты из папки
    def load_labeled_texts(self,directory):
        lds_files=[]#Список имен файлов
        for file in os.listdir(directory):
            if file.endswith(".lds"):
                fullpath=os.path.join(directory,file)
                self.load_lds(fullpath)
    pass

#dl=DataLoader();
#data=dl.load_lds('Data//labeledTexts.lds')
#dl.load_labeled_texts('Data')