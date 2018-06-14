import numpy

import constants
import numpy as np

def framewise_mapper(labels_list):
    """
    Ставит метку границы там, где меняется символ, для остальных меток--'не-граница'
    :param labels_list:
    :return:
    """
    result_list=[]
    labels_count=len(labels_list)

    if labels_count>0:
        first_label=make_label(constants.CONNECTION_LABEL,labels_list[0][constants.INDEX_KEY])
        result_list.append(first_label)
        for i in np.arange(1,labels_count):
            current_label = labels_list[i]
            prev_label=labels_list[i-1]
            if prev_label[constants.CHAR_KEY]==current_label[constants.CHAR_KEY] and i!=labels_count-1:#Если не граница
                new_label=make_label(constants.NOISE_LABEL,prev_label[constants.INDEX_KEY])
            else:
                new_label=make_label(constants.CONNECTION_LABEL,prev_label[constants.INDEX_KEY])
            result_list.append(new_label)
    return result_list


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


frag_map={'4':'27','23':'24','12':'31','42':'31','39':'3','27':'40','41':'46','1':'34',
          '19':'17','21':'17',
          '16':'17','20':'22','47':'6','26':'49','38':'54','28':'35','55':'56'}
def fragments_mapper(labels_list):
    result_list = []
    for label in labels_list:
        new_label = label
        if label is not None:

            if frag_map.get(label['Item1']) is not None:#Если есть в словаре, то заменить
                new_label['Item1']=frag_map[label['Item1']]
        result_list.append(new_label)
    return result_list


def make_label(symbol,index):
    return {constants.CHAR_KEY:symbol,constants.INDEX_KEY:index}