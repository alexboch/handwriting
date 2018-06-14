from enum import Enum

import tensorflow as tf

import data_filters as dc
import labels_converter as lc
import prepare_data as prepdata
from feature_vectors_set import *
from mappers import *

"""
В этом модуле задается соответствие между параметрами и конфигурациями
"""

class TrainConfig(Enum):
    BORDERS=1#Границы букв, метка для каждого входного вектора
    LETTERS=2#Метка буквы для каждого входного вектора
    LETTERS_MERGED=3#Метки букв, одинаковые буквы сливаются(распознавание слов без сегментации)
    CONNECTIONS=4
    FRAGMENTS=5#Разделение на фрагменты

class CellType(Enum):
    LSTM=1
    GRU=2
    BIDIR_LSTM=3

class NetworkConfig:
    def __init__(self,num_units,num_layers,learning_rate,batch_size=1):
        self.num_units=num_units
        self.num_layers=num_layers
        self.learning_rate=learning_rate
        self.batch_size=batch_size

def get_labels_filter(train_config):
    if train_config==TrainConfig.FRAGMENTS:
        return dc.num_filter
    return None

def get_num_epochs(train_config):
    """
    Возвращает кол-во эпох обучения
    :param train_config:
    :return:
    """
    if train_config==TrainConfig.BORDERS:
        return 5
    else:
        if train_config==TrainConfig.LETTERS_MERGED:
            return 500
        else:
            if train_config==TrainConfig.LETTERS:
                return 25
            else:
                return 1000

def get_network_config(train_config):
    if train_config==TrainConfig.BORDERS:
        return NetworkConfig(num_units=512,num_layers=2,learning_rate=0.01)
    else:

        if train_config==TrainConfig.LETTERS_MERGED:
            return NetworkConfig(num_units=400,num_layers=1,learning_rate=1e-6)
        else:
            if train_config==TrainConfig.LETTERS:
                return NetworkConfig(512,2,1e-3)
            else:
                if train_config==TrainConfig.FRAGMENTS:
                    return NetworkConfig(num_units=512, num_layers=2, learning_rate=0.01)
                else:
                    return NetworkConfig(500,1,1e-8)

def get_model_name(train_config):
    return train_config.name

def get_featurizer(train_config):
    if train_config==TrainConfig.BORDERS:
        return FeatureVectorsSetFull()
    else:
        return FeatureVectorsSetFull()


def get_data_directory(train_config):
    return "Data"

def lstm_cell_factory(num_units):
    cell=tf.contrib.rnn.LSTMCell(num_units,state_is_tuple=True)

def get_cell_factory(train_config):
    """Функция, создающая клетку нейронки"""


def get_alphabet(train_config):

    if train_config is TrainConfig.BORDERS or train_config is TrainConfig.CONNECTIONS:
        chars=[constants.CONNECTION_LABEL,constants.NOISE_LABEL]
        return prepdata.LettersAlphabet(chars)
    else:
        if train_config is TrainConfig.LETTERS or train_config is TrainConfig.LETTERS_MERGED:
            chars=[chr(x+1040) for x in range(65)]#Русский алфавит в UTF-8
            chars.append('Ё')
            chars.append('ё')
            chars.append(constants.CONNECTION_LABEL)
            chars.append(constants.NOISE_LABEL)
            return prepdata.LettersAlphabet(chars)
        else:
            if train_config is TrainConfig.FRAGMENTS:
                return prepdata.NumbersAlphabet(128)
            else:
                chars = [constants.CONNECTION_LABEL, constants.NOISE_LABEL]
                return prepdata.LettersAlphabet(chars)
    #return prepdata.LettersAlphabet(chars)

def get_labels_mapper(train_config):
    if train_config==TrainConfig.BORDERS:
        return framewise_mapper
    else:
        if train_config==TrainConfig.CONNECTIONS:
            return connections_only_mapper
        else:
            if train_config==TrainConfig.FRAGMENTS:
                return fragments_mapper

def get_labels_converter(train_config):
    if train_config==TrainConfig.FRAGMENTS:
        return lc.BinLabelsConverter(7)
    return None