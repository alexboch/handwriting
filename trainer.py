import prepare_data as pd
import tensorflow as tf
import numpy as np
from utils import levenshtein
from lstm import *

class Trainer:
    """
    Запускает процесс загрузки данных и тренировки нейросети
    """
    def __init__(self,network,data_loader,num_epochs,model_name,data_directory,output_training=False):
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
        self.network.train(data,self.num_epochs,self.output_training,self.model_name)
        pass


    def get_labels_for_list(self, words_list):
        """

        :param words_list:Список слов
        :return: Список меток в виде чисел
        """

        return [self.network.label([word.point_list],symbolic=True) for word in words_list]

    def run_training(self,validate=False):
        self.load_data()
        training_data=self.data_loader.get_words_list()

        data_len=len(training_data)
        validate=validate and data_len>1
        if validate:
            np.random.shuffle(training_data)
            # Разделить на валидационное и тренировочное множества в отношении 20 на 80
            valid_len=int(data_len*0.2)#20% на валидацию
            train_len=data_len-valid_len#80 на обучение
            validation_data=training_data[:valid_len]
            training_data=training_data[valid_len:]
        train_stat=self.train_network(training_data)

        if validate:
            decoded_labels=self.get_labels_for_list(validation_data)
            target_labels=[self.data_loader.labels_map_function(word) for word in validation_data]
            validation_error=levenshtein(decoded_labels,target_labels)/(len(target_labels)+len(decoded_labels))#нормализованное расстояние редактирования
        pass