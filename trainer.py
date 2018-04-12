import prepare_data as pd
import tensorflow as tf
import numpy as np
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
        pass

    def load_data(self):
        return self.data_loader.load_labeled_texts(self.data_directory)

    def train_network(self,data=None):
        if data is None:
            data=self.data_loader.get_words_list()
        self.network.train(data,self.num_epochs,self.output_training,self.model_name)
        pass


    def run_training(self):
        self.load_data()

        self.train_network(self.data_loader.get_words_list())

        pass