import numpy as np
import tensorflow as tf
import csv
import time
import os
from prepare_data import DataHelper
from utils import *
import tensorflow.contrib.layers as layers

map_fn = tf.map_fn


class LSTMDecoder:
    """
    Класс для создания, обучения и получения разметки от LSTM-нейросети TODO:Сделать другие варианты клеток, кроме LSTM
    """

    def __init__(self, num_units, num_layers, num_features, num_classes, learning_rate, batch_size,alphabet):
        """
        Конструктор, в нем задаются размеры слоев и создается клетка сети
        """
        self.create_network(num_units, num_layers, num_features,
                            num_classes, learning_rate, batch_size)
        self.alphabet=alphabet
        pass






    def label(self, points, path_to_model=None,model_dir=None, symbolic=False):
        """

        :param points: Последовательность из последовательностей точек
        :param path_to_model:полный путь к метаграфу
        :param model_dir: путь к папке с сохраненной моделью
        :param symbolic: Преобразовывать ли числовые метки в символы
        :return:
        """
        if path_to_model is None:
            if self._checkpoint_path is None:
                raise Exception("Model not found! Train model or supply valid path")
            path_to_model=self._checkpoint_path

        path_to_model= path_to_model + ".meta"
        #session = tf.Session()
        with tf.Session() as session:
        #saver = tf.train.Saver()
            saver=tf.train.import_meta_graph(path_to_model)
            saver.restore(session, tf.train.latest_checkpoint(model_dir))#Загрузить натренированную модель
            #session.run(tf.global_variables_initializer())
            length = [len(points_list) for points_list in points]
            decoded_list=[]
            for points_list in points:
                input_arr=np.asarray(points_list)
                input_arr=np.expand_dims(input_arr,0)
                decoded, log_prob,probs = session.run([self.decoded_integers, self.log_prob,self.probs],
                                                feed_dict={self.inputs: input_arr, self.seq_len: [len(points_list)]})
                arr_decoded = np.asarray(decoded.values)
                if symbolic:
                    arr_decoded=self.alphabet.decode_numeric_labels(arr_decoded)#Числовые метки в символьные
                decoded_list.append(arr_decoded)
        return np.asarray(decoded_list)

    TINY = 1e-6  # to avoid NaNs in logs

    def train(self, words, num_epochs=100, output_training=False, model_name="model.ckpt",model_dir_path=f"Models{os.sep}model.ckpt",):
        """
        words--Список слов, содержащих точки и метки
        """
        print("starting training,epochs:",num_epochs,"learning rate:",self.learning_rate)
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        num_batches = len(words) / self.batch_size
        num_words = len(words)
        output_period=max(num_epochs/100.0*5,1)#Выводить каждый раз, когда прошло 5% обучения
        epoch_errors=[]
        start_time=time.time()
        prev_epoch_time=start_time
        for i in np.arange(num_epochs):
            #print("Epoch number ", str(i))
            np.random.shuffle(words)
            epoch_error = 0  # Средняя ошибка по всем батчам в данной эпохе
            edit_dist=0#По расстоянию редактирования
            can_output=output_training and (i%output_period==0 or i==num_epochs-1)
            for j in np.arange(0, num_words, self.batch_size):  # Цикл по всем словам, берем по batch_size слов
                j1 = j
                j2 = j1 + self.batch_size
                batch_words = words[j:j2]  # слова для создания батча
                batch_inputs = []
                batch_labels = []
                seq_length = []  # Вектор длин каждой последовательности
                # получить список точек и меток
                for w in batch_words:
                    batch_inputs.append(w.point_list)
                    seq_length.append(len(w.point_list))  # Присоединяем длину последовательности точек
                    batch_labels.append(w.labels_list)
                inputs_arr = np.asarray(batch_inputs)
                targets_array = np.asarray(batch_labels)
                targets_array = sparse_tuple_from(targets_array)
                #targets_sparse_tensor=tf.SparseTensor(targets_array[0],targets_array[1],targets_array[2])
                s, loss, logits,probs, ler, decoded_integers,targets,_ = session.run([self.cost, self.loss, self.logits, self.probs, self.ler, self.decoded_integers, self.targets, self.train_fn],
                                                                             feed_dict={self.inputs: inputs_arr, self.targets: targets_array,
                                                                 self.seq_len: seq_length})
                #indices=np.asarray(decoded_integers.indices)
                if can_output:
                    target_values = np.asarray(decoded_integers.values)
                    char_decoded=self.alphabet.decode_numeric_labels(target_values)
                    print("Decoded chars:",char_decoded)
                    print('batch loss:', loss)
                    print('edit distance error:', ler)

                edit_dist+=ler
                epoch_error += s
            epoch_error /= num_words
            edit_dist/=num_words

            elapsed_time=time.time()-start_time

            if i%output_period==0 or i==num_epochs-1:
                if output_training:
                    print("Epoch number:",i)
                    print("Epoch error:", epoch_error)
                    print(f"Edit distance:{edit_dist}")
                    print(f"Time:{elapsed_time}")
                epoch_errors.append({"Epoch":i,"Error":epoch_error,"Edit distance":edit_dist,"Time":elapsed_time})

        for epoch_error in epoch_errors:
            for k,v in epoch_error.items():
                epoch_error[k]=str(v).replace('.',',')#Заменить на запятую, чтобы excel понимал
        #Вывод в csv-файл
        headers=['Epoch','Error','Edit distance',"Time"]

        self._csv_path=os.path.join(model_dir_path,f"{model_name}.csv")
        if not os.path.exists(model_dir_path):
            os.makedirs(model_dir_path)#Создать папку модели, если она не существует
        with open(self._csv_path, 'w+',newline='') as csvfile:
            sep=getListSeparator()#Получить разделитель для текущих настроек локали
            csvwriter=csv.DictWriter(csvfile,fieldnames=headers,delimiter=sep)
            csvwriter.writeheader()
            for i in np.arange(len(epoch_errors)):
                data=epoch_errors[i]
                csvwriter.writerow(data)#Записать строку в csv
        self._checkpoint_path=os.path.join(model_dir_path, model_name)#Путь к сохраненному графу
        saver = tf.train.Saver()
        saver.save(session,self._checkpoint_path)
        session.close()
        return epoch_errors

    def lstm_cell(self):
        cell = tf.contrib.rnn.LSTMCell(self.num_units, state_is_tuple=True)
        return cell


    def create_multi_bilstm(self,num_layers,inputs):
        cells_fw=[]
        cells_bw=[]
        for n in range(num_layers):
            cell_fw=self.lstm_cell()
            cell_bw=self.lstm_cell()
            cells_fw.append(cell_fw)
            cells_bw.append(cell_bw)
        return tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw,cells_bw=cells_bw,inputs=inputs, dtype=tf.float32)


    def create_network(self, num_units, num_layers, num_features, num_classes, learning_rate, batch_size=10):
        self.num_units = num_units
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.inputs = tf.placeholder(tf.float32, [None, None, num_features], name='inputs')
        self.num_classes = num_classes
        shape = tf.shape(self.inputs)
        batch_s, max_timesteps = shape[0], shape[1]
        # self.targets = tf.placeholder(tf.float32, (None, None, num_classes))#
        # Here we use sparse_placeholder that will generate a
        # SparseTensor required by ctc_loss op.

        self.targets = tf.sparse_placeholder(tf.int32)

        self.cells_stack = tf.contrib.rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.num_layers)],
                                                       state_is_tuple=True)

        # for i in range(self.num)

        self.W = tf.Variable(
            tf.truncated_normal([self.num_units, self.num_classes], stddev=0.1),name='W')  # Начальная матрица весов
        self.b = tf.Variable(tf.constant(0., shape=[self.num_classes]),name='b')
        # Given inputs (time, batch, input_size) outputs a tuple
        #  - outputs: (time, batch, output_size)  [do not mistake with OUTPUT_SIZE]
        #  - states:  (time, batch, hidden_size)
        self.batch_size = batch_size
        # 1d array of size [batch_size]
        self.seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
        # rnn_outputs--Тензор размерности [batch_size,max_time,cell.output_size],max_time--кол-во точек слова,cell.output_size=2(x,y координаты)
        # rnn_state--последнее состояние

        # self.rnn_outputs, self.rnn_state_fw,_ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self.cells_stack, self.inputs, self.seq_len,
        #                                                        dtype=tf.float32)
        self.rnn_outputs, self.rnn_state_fw = tf.nn.dynamic_rnn(self.cells_stack, self.inputs, self.seq_len,
                                                                dtype=tf.float32)
        #self.rnn_outputs,self.output_state_fw,self.output_state_bw=self.create_multi_bilstm(self.num_layers,self.inputs)
        # Reshaping to apply the same weights over the timesteps
        self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.num_units])
        self.logits = tf.matmul(self.rnn_outputs, self.W) + self.b
        # Reshaping back to the original shape
        self.logits = tf.reshape(self.logits, [self.batch_size, -1, self.num_classes])

        # Time major

        self.logits = tf.transpose(self.logits, (1, 0, 2))#Shape: [seq_length,batch_size,num_classes]
        self.probs=tf.nn.softmax(self.logits)#вероятность для [t,batch_num,class_num]
        #self.loss = tf.nn.ctc_loss(self.targets, self.logits, self.seq_len,preprocess_collapse_repeated=False,ctc_merge_repeated=False)
        self.loss = tf.nn.ctc_loss(self.targets, self.logits, self.seq_len,preprocess_collapse_repeated=False,ctc_merge_repeated=False)
        self.cost = tf.reduce_mean(self.loss)
        self.train_fn = tf.train.MomentumOptimizer(self.learning_rate,
                                                   0.99).minimize(self.cost)
        #self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.logits, self.seq_len,merge_repeated=False)
        self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.logits, self.seq_len,merge_repeated=False)
        self.decoded_integers=tf.cast(self.decoded[0], tf.int32)
        self.ler = tf.reduce_mean(tf.edit_distance(self.decoded_integers, self.targets))
        pass
