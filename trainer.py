import prepare_data as pd
import tensorflow as tf
import numpy as np
from utils import levenshtein
from lstm import *
from datetime import datetime

class Trainer:
    """
    Запускает процесс загрузки данных и тренировки нейросети
    """
    def __init__(self,network:LSTMDecoder,data_loader:DataHelper,num_epochs,model_name,data_directory,output_training=False):
        self.network=network
        self.num_epochs=num_epochs
        self.model_name=model_name
        self.data_directory=data_directory
        self.data_loader=data_loader
        self.output_training=output_training
        self.model_dir_path=f"{data_directory}{os.sep}{model_name}"

        pass

    def load_data(self):
        return self.data_loader.load_labeled_texts(self.data_directory)

    def train_network(self,data=None):
        if data is None:
            data=self.data_loader.get_words_list()
        model_dir_path_with_datetime=self.model_dir_path+datetime.now().strftime('%d-%m-%Y-%I_%M_%S')
        self.network.train(data,self.num_epochs,self.output_training,self.model_name,model_dir_path_with_datetime)
        pass

    def set_learning_rate(self,learning_rate):
        self.network.learning_rate=learning_rate

    def get_labels_for_list(self, words_list):
        """

        :param words_list:Список слов
        :return: Список меток в виде чисел
        """
        points=[word.point_list for word in words_list]
        return self.network.label(points,model_dir='D:\Projects\Python\handwriting\Data\BORDERS\\',model_name='BORDERS')

    def run_training(self, test=False):
        self.load_data()
        training_data=self.data_loader.get_words_list()
        #points = [word.point_list for word in training_data]
        points=[]
        for word in training_data:
            points.append(word.point_list)
        target_labels = [word.labels_list for word in training_data]
        #self.network.label(points, path_to_model='D:\Projects\Python\handwriting\Data\BORDERS\BORDERS',model_dir='D:\Projects\Python\handwriting\Data\BORDERS\\')
        data_len=len(training_data)
        test= test and data_len > 1
        if test:
            np.random.shuffle(training_data)
            # Разделить на валидационное и тренировочное множества в отношении 20 на 80
            test_len=int(data_len * 0.2)#20% на тест
            train_len= data_len - test_len#80 на обучение
            test_data= training_data[:test_len]
            training_data= training_data[test_len:]

        train_stat=self.train_network(training_data)

        if test:
            #decoded_labels_list=self.get_labels_for_list(validation_data)
            #target_labels_list=[word.labels_list for word in validation_data]

            #validation_error=levenshtein(decoded_labels,target_labels)/(len(target_labels)+len(decoded_labels))#нормализованное расстояние редактирования
            validation_error=0.0

            """for i in range(len(decoded_labels_list)):
                decoded_labels=decoded_labels_list[i]
                target_labels=target_labels_list[i]
                validation_error+=levenshtein(decoded_labels,target_labels)/(len(target_labels)+len(decoded_labels))#нормализованное расстояние редактирования
            validation_error/=len(decoded_labels_list)#Среднее расстояние редактирования по всем спискам точек"""

            print(f"Validation error:{validation_error}")
        pass