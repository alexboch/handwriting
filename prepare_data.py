import tensorflow as tf
import pandas as pd
import numpy as np
import os


class WordData:
    """
    Класс слова, содержащий список точек и меток
    """
    
    
    def __init__(self,point_list=[],labels_list=[],text=""):
        self.point_list=point_list
        self.labels_list=labels_list
        self.text=text
class DataLoader:
    
    #Словарь с точками слов и метками
    words_dict={}
    def __init__(self):
        return
    
   
    
    def label_to_int(self,char_label):
        return ord(char_label)-1072#Код буквы А=1072
    
    def int_label_to_char(self,int_label):
        return chr(int_label + 1072)
    
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
        for key_word in self.words_dict:
            self.words_dict[key_word]=[]
        for text in transposed_data.iterrows():
            pl=text[1][['PointLists','Labels']]
            #words_dict=dict.fromkeys(text_words)
            num_words=len(text[1].TextWords)
            tmp_words_data=[WordData() for i in np.arange(num_words)]#временный список данных слов для текущего текста
            is_labeled=False
            for i in np.arange(len(pl.PointLists)):
                print(i)
                points_list=pl.PointLists[i]
                labels_list=pl.Labels[i]
                for j in np.arange(len(points_list)):
                    point=points_list[j]
                    label=labels_list[j]
                    if label is not None:#Если не нулевая метка
                        is_labeled=True
                        char_label=label['Item1']
                        integer_label=self.label_to_int(char_label)#Букву в число
                        word_index=label['Item2']#индекс слова в списке
                        tmp_words_data[word_index].point_list.append(point)#сохранить данные слова по ключу
                        tmp_words_data[word_index].labels_list.append(integer_label)
                        tmp_words_data[word_index].text=text[1].TextWords[word_index]
                        
            if is_labeled:#сохранить данные, только если в тексте есть метки
                words_data.extend(tmp_words_data)
        for wd in words_data:#пройти по всем словам и сохранить данные в словарь слов по всем текстам
            if wd.text not in self.words_dict:#если в словаре нет данных для такого слова
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

dl=DataLoader();
#data=dl.load_lds('Data//labeledTexts.lds')
dl.load_labeled_texts('Data')