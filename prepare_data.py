import tensorflow as tf
import pandas as pd
import numpy as np
import os
from operator import methodcaller

class WordData:
    """
    Класс слова, содержащий список точек и меток
    """
    
    def __init__(self):
        self.point_list=[]
        self.labels_list=[]
        self.text=""
    
        
class DataLoader:
    
    #Словарь с точками слов и метками
    words_dict={}
    def __init__(self):
        return
    
    NOISE_LABEL='&'#Метка для шума
    CONNECTION_LABEL='$'#Метка для соединения
    LAST_CODE=1105#Код буквы ё, последней в UTF-8
    FIRST_CODE=1040#Код буквы

    @staticmethod
    def label_to_int(char_label):
        if char_label==DataLoader.CONNECTION_LABEL:
            return DataLoader.LAST_CODE-DataLoader.FIRST_CODE+1
        if char_label==DataLoader.NOISE_LABEL:
            return DataLoader.LAST_CODE-DataLoader.FIRST_CODE+2
        return ord(char_label)-DataLoader.FIRST_CODE#Код буквы а=1072
    
    @staticmethod
    def int_label_to_char(int_label):
        if int_label==DataLoader.LAST_CODE+1:
            return DataLoader.CONNECTION_LABEL
        if int_label==DataLoader.LAST_CODE+2:
            return DataLoader.NOISE_LABEL
        return chr(int_label + DataLoader.FIRST_CODE)
    
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
            pl=text[1][['PointLists','Labels']]#выбрать  списки точек и соответствующие им списки меток
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
                points_list=pl.PointLists[i]
                labels_list=pl.Labels[i]
                points_list=list(map(methodcaller("split",","),points_list))#разделить координаты на x и y
                #map(lambda p: p.split(","),points_list)
                points_list=list(map(lambda x:list(map(float, x)),points_list))#превратить координаты в числа
                for j in np.arange(len(points_list)):#Цикл по всем точкам списка
                    point=points_list[j]
                    label=labels_list[j]
                    if label is not None:#Если не нулевая метка
                        is_labeled=True
                        char_label=label['Item1']
                        integer_label=self.label_to_int(char_label)#Букву в число
                        word_index=label['Item2']#индекс слова в списке
                        tmp_words_data[word_index].point_list.append(point)#сохранить данные слова по ключу
                        tmp_words_data[word_index].labels_list.append(integer_label)
                        print(word_index)
                        assert(text[1].TextWords[word_index]!='')
                        tmp_words_data[word_index].text=text[1].TextWords[word_index]#Задать строку текста
                        
            if is_labeled:#сохранить данные, только если в тексте есть метки
                words_data.extend(filter(lambda w:w.text!='', tmp_words_data))
        for wd in words_data:#пройти по всем словам и сохранить данные в словарь слов по всем текстам
            if wd.text not in self.words_dict:#если в словаре нет данных для такого слова
                assert(wd.text!='')
                self.words_dict[wd.text]=[]
            
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