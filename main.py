import tensorflow as tf
import numpy as np
import prepare_data as prepdata
import lstm

tf.reset_default_graph()
# Загрузка данных
dl = prepdata.DataLoader()
dl.load_labeled_texts('Data');
# нейросеть
#TODO:Понять, что такое "пустая метка", и нужна ли метка шума
num_classes=69#Строчные и заглавные буквы + соединение + шум + пустая метка
ld = lstm.LSTMDecoder(num_units=75, num_layers=1, input_size=2, num_classes=num_classes, learning_rate=0.0001, batch_size=1)
ld.train([dl.words_dict['люстра'][0]],100)
#ld.train(dl.get_words_list(),1000)
labels,probs=ld.label([dl.words_dict['люстра'][0].point_list])