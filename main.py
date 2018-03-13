import tensorflow as tf
import numpy as np
import prepare_data as prepdata
import lstm

tf.reset_default_graph()
#Загрузка данных
dl=prepdata.DataLoader()
dl.load_labeled_texts('Data');
#нейросеть
ld=lstm.LSTMDecoder(2,1,2,2,1,1)