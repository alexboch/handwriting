import tensorflow as tf
import numpy as np
import prepare_data as prepdata
import lstm

tf.reset_default_graph()
# Загрузка данных
dl = prepdata.DataLoader()
dl.load_labeled_texts('Data');
# нейросеть
ld = lstm.LSTMDecoder(num_units=2, num_layers=1, input_size=2, output_size=1, learning_rate=1, batch_size=1)
ld.train(dl.words_dict['люстра'])
