import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

class LSTMDecoder:
    """
    Класс для создания, обучения и получения разметки от LSTM-нейросети
    """
    
    def __init__(self,num_units,num_layers,learning_rate):
        """
        Конструктор, в нем задаются размеры слоев и создается клетка сети
        """
        self.create_network(num_units,num_layers,learning_rate)
        pass
    
    def train(self,data):
        
        pass
        
    
    def create_network(self,num_units,num_layers,learning_rate):
        self.num_units=num_units
        self.num_layers=num_layers
        self.learning_rate=learning_rate
        cells=[]
        for _ in range(num_layers):#Создать клетки для слоев
            cell=tf.contrib.rnn.LSTMCell(num_units)
            cells.append(cell)
            
        cell=tf.contrib.rnn.MultiRNNCell(cells)
        pass
