import tensorflow as tf
import numpy as np
import prepare_data as prepdata
from decoder_factories import *
import constants
import lstm

tf.reset_default_graph()
all_chars=[chr(x+1040) for x in range(65)]#Все символы русского алфавита
all_chars.append(constants.CONNECTION_LABEL)#Метка соединения
all_chars.append(constants.NOISE_LABEL)#Метка шума
full_alphabet=prepdata.LabelsAlphabet(all_chars)
# Загрузка данных
dh = prepdata.DataHelper(full_alphabet)
dh.load_labeled_texts('SmallData');
#dh.load_labeled_texts('Data');
# нейросеть
num_classes=69#Строчные и заглавные буквы + соединение + шум + пустая метка

factory=FullAlphabetDecoderFactory()
ld=factory.CreateDecoder()
ld.learning_rate=0.1
ld.num_units=75
#ld = lstm.LSTMDecoder(num_units=300, num_layers=1, num_features=2, num_classes=num_classes, learning_rate=1e-5, batch_size=1)
ld.train([dh.words_dict['аб'][0]],10000)
#ld.train(dh.get_words_list(), 10000)
#labels,probs=ld.label([dl.words_dict['аб'][0].point_list])