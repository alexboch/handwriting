import tensorflow as tf
import numpy as np
import prepare_data as prepdata
from decoder_factories import *
import constants
import lstm

all_chars=[chr(x+1040) for x in range(65)]#Все символы русского алфавита
all_chars.append(constants.CONNECTION_LABEL)#Метка соединения
all_chars.append(constants.NOISE_LABEL)#Метка шума
full_alphabet=prepdata.LabelsAlphabet(all_chars)
chars=[constants.CONNECTION_LABEL,constants.NOISE_LABEL]
connections_only_alphabet=prepdata.LabelsAlphabet(chars)
def train_output_func(decoded):
    values = np.asarray(decoded.values)
    train_decoded = []
    for x in values:
        train_decoded.append(full_alphabet.int_label_to_char(x))
    print("Decoding:", train_decoded)
    pass
#TODO:Переместить вывод в класс Alphabet
def connections_only_output_func(decoded):
    values = np.asarray(decoded.values)
    train_decoded = []
    for x in values:
        train_decoded.append(connections_only_alphabet.int_label_to_char(x))
    print("Decoding:", train_decoded)
    pass

def make_label(symbol,index):
    return {constants.CHAR_KEY:symbol,constants.INDEX_KEY:index}

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

tf.reset_default_graph()


# Загрузка данных
#dh = prepdata.DataHelper(full_alphabet)
dh=prepdata.DataHelper(connections_only_alphabet)#Только соединения/не соединения
#dh.labels_map_function=connections_only_mapper
dh.labels_map_function=framewise_mapper
dh.load_labeled_texts('SmallData');
#dh.load_labeled_texts('Data')
#dh.load_labeled_texts('Data');
# нейросеть
num_classes=69#Строчные и заглавные буквы + соединение + шум + пустая метка

#factory=ConnectionsOnlyDecoderFactory()
factory=FullAlphabetDecoderFactory()
ld=factory.CreateDecoder()
#ld.learning_rate=0.1
#ld.num_units=75
#ld = lstm.LSTMDecoder(num_units=300, num_layers=1, num_features=2, num_classes=num_classes, learning_rate=1e-5, batch_size=1)
train_word=prepdata.WordData()
train_word.point_list.extend([(1, 1), (0, 0), (-1, 1),(0,0)])

train_word.labels_list.extend(['а', 'и', 'а', 'и'])
#test_word.labels_list=connections_only_alphabet.encode_char_labels(test_word.labels_list)
#ld.train([test_word],1000,connections_only_output_func)
train_word.labels_list=full_alphabet.encode_char_labels(train_word.labels_list)
ld.train([train_word], 150, train_output_func)
#ld.train([dh.words_dict['аб'][0]],10000,connections_only_output_func)
#ld.train(dh.get_words_list(), 50,connections_only_output_func)
#ld.train(dh.get_words_list(),50,train_output_func)
#labels,probs=ld.label([dh.words_dict['аб'][0].point_list],"Models/model.ckpt")

test_word=prepdata.WordData()
test_word.point_list.extend([(0, 0), (-1, 1)])
test_word.labels_list.extend(['и', 'а'])
test_word.labels_list=full_alphabet.encode_char_labels(test_word.labels_list)
labels,probs=ld.label([test_word.point_list],"Models/model.ckpt")
#labels,probs=ld.label([dh.words_dict['аб'][0].point_list],"Models/model.ckpt")
char_labels=full_alphabet.decode_numeric_labels(labels)
#char_labels=connections_only_alphabet.decode_numeric_labels(labels)
print("Labels:",char_labels)