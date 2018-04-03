from enum import Enum
import prepare_data as prepdata
import numpy as np
import constants

class TrainConfig(Enum):
    BORDERS=1#Границы букв, метка для каждого входного вектора
    LETTERS=2#Метка буквы для каждого входного вектора
    LETTERS_MERGED=3#Метки букв, одинаковые буквы сливаются(распознавание слов без сегментации)
    CONNECTIONS=4

def make_label(symbol,index):
    return {constants.CHAR_KEY:symbol,constants.INDEX_KEY:index}


def connections_only_mapper(labels_list):
    """

    :param labels_list:
    :return:
    """
    result_list=[]
    for label in labels_list:
        new_label=label
        if label is not None:
            new_char_label=label['Item1'] if label['Item1']==constants.CONNECTION_LABEL else constants.NOISE_LABEL
            new_label['Item1']=new_char_label
        result_list.append(new_label)
    return result_list

def framewise_mapper(labels_list):
    """
    Ставит метку границы там, где меняется символ, для остальных меток--'не-граница'
    :param labels_list:
    :return:
    """
    result_list=[]
    labels_count=len(labels_list)

    if labels_count>0:
        for i in np.arange(1,labels_count):
            current_label = labels_list[i]
            prev_label=labels_list[i-1]
            if prev_label[constants.CHAR_KEY]==current_label[constants.CHAR_KEY]:#Если не граница
                new_label=make_label(constants.NOISE_LABEL,prev_label[constants.INDEX_KEY])
            else:
                new_label=make_label(constants.CONNECTION_LABEL,prev_label[constants.INDEX_KEY])
            result_list.append(new_label)
    return result_list

def get_alphabet(train_config):

    if train_config==TrainConfig.BORDERS:
        chars=[constants.CONNECTION_LABEL,constants.NOISE_LABEL]
    else:
        if train_config==TrainConfig.LETTERS or train_config==TrainConfig.LETTERS_MERGED:
            chars=[chr(x+1040) for x in range(65)]
    return prepdata.LabelsAlphabet(chars)

def get_labels_mapper(train_config):
    if train_config==TrainConfig.BORDERS:
        return framewise_mapper
    else:
        if train_config==TrainConfig.CONNECTIONS:
            return connections_only_mapper

