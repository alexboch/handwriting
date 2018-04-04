import prepare_data as pd
import tensorflow as tf
import numpy as np
from lstm import *

class Trainer:
    """
    Запускает процесс загрузки данных и тренировки нейросети
    """
    def __init__(self,network,num_epochs,model_name):
        self.network=network
        self.num_epochs=num_epochs
        self.model_name=model_name
        
        pass

    def load_data(self):
        pass

    def train_network(self,data):
        self.network.train(data,)
        pass



    def run_training(self):
        data=self.load_data()
        self.train_network(data)

        pass