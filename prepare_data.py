import tensorflow as tf
import pandas as pd
import numpy as np
import os
from PointsAndRectangles import *
import constants
from operator import methodcaller
from sklearn.preprocessing import normalize
from feature_points_set_base import *
import pickle
import argparse
import configurator as conf


class LabelsAlphabet:
    """
    Содержит методы преобразования между числовыми метками и буквенными
    """
    def __init__(self,characters):
        self.int_to_char_dict={}#Словарь для преобразования из числовой метки в буквенную
        self.char_to_int_dict={}#Словарь для преобразования из буквенной метки в числовую
        self.num_chars=len(characters)
        for i in np.arange(self.num_chars):
            self.int_to_char_dict[i]=characters[i]
            self.char_to_int_dict[characters[i]]=i




    def one_hot(self,labels):
        one_hot_labels=[]
        for label in labels:
            vector=[0.0]*self.num_chars
            vector[label]=1.0
            one_hot_labels.append(vector)
        return one_hot_labels

    def label_to_int(self,char_label):
        return self.char_to_int_dict[char_label]


    def int_label_to_char(self,int_label):
        return self.int_to_char_dict[int_label]

    def decode_numeric_labels(self,int_labels):
        return [self.int_label_to_char(x) for x in int_labels]

    def encode_char_labels(self,char_labels):
        return [self.label_to_int(x) for x in char_labels]

    def get_length(self):
        return len(self.char_to_int_dict)

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
    def __init__(self, labels_alphabet:LabelsAlphabet, featurizer:FeaturePointsSetBase, labels_map_function=None):
        self.labels_alphabet=labels_alphabet
        self.featurizer=featurizer
        self.labels_map_function=labels_map_function
        return

    LastCode=1105#Код буквы ё, последней в UTF-8
    FirstCode=1040#Код буквы


    def get_words_list(self):
        wl=list(self.words_dict.values())
        flattened_list=[item for sublist in wl for item in sublist]
        return flattened_list

    @staticmethod
    def get_data_vectors(points):
        vectors=[]

        return vectors

    @staticmethod
    def get_vectors_from_points(points):
        vectors=[]#список векторов
        if len(points)==1:
            #points.append(points[0])#Чтобы получился вектор из одной точки
            vectors.append([0,0])
        #for i in np.arange(len(points)-1):
        #    p1=points[i]#текущая точка
        #    p2=points[i+1]#следующая точка
        #    v=[p2[0]-p1[0],p2[1]-p1[1]]
        #    vectors.append(v)
        else:
            p1=points[0]
            for i in np.arange(1,len(points)):
                p2=points[i]
                v = [p2[0] - p1[0], p2[1] - p1[1]]
                vectors.append(v)
                p1=points[i]
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


    def clear(self):
        """Очищает загруженные слова"""


    def read_data(self,filaname):
        """
        Читает из файла обработанные данные
        :param filaname:
        :return:
        """
        try:
            with open(filaname,mode='rb') as f:
                self.words_dict=pickle.load(f)
        except Exception as ex:
            print(f"Error loading words dictionary from file:{ex}")

    def save_data(self,filename):
        """
        Сохраняет обработанные данные слов в файл
        :param filename:
        :return:
        """
        try:
            with open(filename,mode='wb') as f:
                pickle.dump(self.words_dict,f)
        except Exception as ex:
            print(f"Error saving words dictionary to file:{ex}")

    def load_lds(self,filename,merged_labels=False):
        """
        Добавляет в словарь точки слов из файла
        """
        raw_data=pd.read_json(filename,encoding='utf8')
        transposed_data=raw_data.T
        labeled_words=[]
        words_data=[]
        #Создать ключи словаря из слов всех текстов
        for text in transposed_data.iterrows():#Цикл по всем текстам текущего файла
            pl=text[1][['PointLists','Labels']].dropna()#выбрать  списки точек и соответствующие им списки меток
            num_words=len(text[1].TextWords)
            tmp_words_data=[]
            for i in np.arange(num_words):
                wd=WordData()
                tmp_words_data.append(wd)
            is_labeled=False
            try:
                for i in np.arange(len(pl.PointLists)):#Цикл по всем спискам точек
                    points_list,labels_list=DataHelper.filter_nans(pl.PointLists[i],pl.Labels[i])
                    if len(points_list)>0:#Если есть хоть одна метка
                        if self.labels_map_function is not None:
                            labels_list=self.labels_map_function(labels_list)
                        points_list=list(map(methodcaller("split",","),points_list))#разделить координаты на x и y
                        points_list=list(map(lambda x:list(map(float, x)),points_list))#превратить координаты в числа
                        for j in np.arange(len(points_list)):#Цикл по всем точкам списка TODO:Исправить, чтобы не терялась последняя метка
                            vector=points_list[j]

                            label=labels_list[j]#Метка задана для каждой точки
                            if label is not None:#Если не нулевая метка
                                is_labeled=True
                                word_index = label['Item2']  # индекс слова в списке
                                tmp_words_data[word_index].point_list.append(vector)#сохранить данные слова по ключу
                                if not merged_labels:
                                    char_label=label['Item1']
                                    try:
                                        integer_label=self.labels_alphabet.label_to_int(char_label)
                                    except KeyError as kerr:

                                        print(f"Key error:{kerr}")
                                        print(f"Mapper:{self.labels_map_function}")
                                        print(f"Labels:{labels_list}")
                                        print(f"Alphabet:{self.labels_alphabet}")
                                        print(f"Alphabet symbols dictionary:{self.labels_alphabet.char_to_int_dict}")
                                        exit(1)
                                    tmp_words_data[word_index].labels_list.append(integer_label)
                                    assert(text[1].TextWords[word_index]!='')
                                tmp_words_data[word_index].text=text[1].TextWords[word_index]#Задать строку текста
            except IndexError as index_exception:
                print(index_exception)
            for wd in tmp_words_data:
                if wd.point_list is not None and len(wd.point_list)>0:
                    wd.labels_list=self.featurizer.MapToVectorLabels(wd.point_list,wd.labels_list)
                    self.featurizer.CreateFeatures(wd.point_list)  # Вычислить признаки точек
                    wd.point_list=self.featurizer.GetFeatures()
            if is_labeled:#сохранить данные, только если в тексте есть метки
                words_data.extend(filter(lambda w:w.text!='', tmp_words_data))
        for wd in words_data:#пройти по всем словам и сохранить данные в словарь слов по всем текстам
            if wd.text not in self.words_dict:#если в словаре нет данных для такого словаште
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

if __name__=="__main__":
    parser=argparse.ArgumentParser(description='Reads data from lds file and saves processed data')
    usage='Usage:prepare_data [--input [file 1,file2,...]]|[--input_dir input_dir] --config[BORDERS|LETTERS|LETTERS_MERGED|CONNECTIONS|FRAGMENTS]' \
          '  --dir output_dir. ' \
          'Example: py prepare_data.py --input myfile.lds myfile2.lds --config BORDERS --dir C://output_dir'
    parser.add_argument('--input','-i',nargs='*', help='input file',required=False,default='')
    parser.add_argument('--output','-o',type=str,help='output file')
    parser.add_argument('--dir','-d',type=str,help='input directory')
    parser.add_argument('--config','-c',type=str,help='train configuration',default='BORDERS')
    args=parser.parse_args()
    print(f"Arguments:{args}")
    print(args.input)
    print(args.dir)
    has_input=args.input is not None and type(args.input) is list and len(args.input)>0
    has_dir=args.dir is not None and args.dir!=''
    has_config=args.config is not None and args.config!=''
    if (not (has_input ^ has_dir)) or (not has_config):#Должна быть указана одна из опций и только одна
        print('Incorrect arguments!')
        print(usage)
    else:
        config_enum=conf.TrainConfig[args.config]
        alphabet=conf.get_alphabet(config_enum)
        labels_mapper=conf.get_labels_mapper(config_enum)
        featurizer=conf.get_featurizer(config_enum)
        dh:DataHelper=DataHelper(alphabet,featurizer,labels_mapper)
        if has_input:#Если задано одно или несколько имен файлов
            files=args.input
            for file in files:
                dh.load_lds(file)
                print(f"Text from file {file} loaded")
        else:
            if has_dir:
                dh.load_labeled_texts(args.dir)


