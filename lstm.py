import numpy as np
import tensorflow as tf
from prepare_data import DataHelper
from utils import *
import tensorflow.contrib.layers as layers

map_fn = tf.map_fn


class LSTMDecoder:
    """
    Класс для создания, обучения и получения разметки от LSTM-нейросети
    """

    def __init__(self, num_units, num_layers, num_features, num_classes, learning_rate, batch_size):
        """
        Конструктор, в нем задаются размеры слоев и создается клетка сети
        """
        self.create_network(num_units, num_layers, num_features,
                            num_classes, learning_rate, batch_size)
        pass

    def label(self, points,path_to_model):
        """
        Принимает список точек и возвращает последовательность меток
        """
        saver=tf.train.Saver()

        session = tf.Session()
        saver.restore(session,path_to_model)#Загрузить натренированную модель
        #session.run(tf.global_variables_initializer())
        length = [len(points[0])]
        points = np.asarray(points)
        decoded, log_prob = session.run([self.decoded[0], self.log_prob],
                                        feed_dict={self.inputs: points, self.seq_len: length})
        arr_decoded = np.asarray(decoded[1])

        session.close()
        #str_decoded = [(x) for x in arr_decoded]
        #print('Decoded string:', str_decoded)
        return (arr_decoded, log_prob)

    TINY = 1e-6  # to avoid NaNs in logs

    def train(self, words, num_epochs=100,train_output_func=None,model_name="model.ckpt"):
        """
        words--Список слов, содержащих точки и метки
        """
        # init_op = tf.initialize_all_variables()
        saver=tf.train.Saver()
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        num_batches = len(words) / self.batch_size
        num_words = len(words)
        for i in np.arange(num_epochs):
            print("Epoch number ", str(i))
            np.random.shuffle(words)
            epoch_error = 0  # Средняя ошибка по всем батчам в данной эпохе
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
                targets_sparse_tensor=tf.SparseTensor(targets_array[0],targets_array[1],targets_array[2])
                s, loss, logits, ler, cast_seq,targets,_ = session.run([self.cost, self.loss, self.logits, self.ler,self.cast_seq,self.targets, self.train_fn],
                                                      feed_dict={self.inputs: inputs_arr, self.targets: targets_array,
                                                                 self.seq_len: seq_length})
                indices=np.asarray(cast_seq.indices)
                if train_output_func is not None:
                    train_output_func(cast_seq)
                """values=np.asarray(cast_seq.values)
                train_decoded=[]
                for x in values:
                    train_decoded.append(DataHelper.int_label_to_char(x))
                print("Decoding:",train_decoded)"""
                #target_indices=np.asarray(targets_array.indices)
                #target_values=np.asarray(targets_array.values)
                print('batch loss:', loss)
                print('edit distance error:', ler)
                # print('logits:',logits)
                epoch_error += s
                # print("Batch cost:",s)
                # session.run(self.train_fn,feed_dict={self.inputs:batch_inputs, self.outputs:batch_labels})
            epoch_error /= num_words
            print("Epoch error:", epoch_error)
        saver.save(session,"Models/"+model_name)
        session.close()

        pass

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
            tf.truncated_normal([self.num_units, self.num_classes], stddev=0.1))  # Начальная матрица весов
        self.b = tf.Variable(tf.constant(0., shape=[self.num_classes]))
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
        self.logits = tf.transpose(self.logits, (1, 0, 2))
        #self.loss = tf.nn.ctc_loss(self.targets, self.logits, self.seq_len,preprocess_collapse_repeated=False,ctc_merge_repeated=False)
        self.loss = tf.nn.ctc_loss(self.targets, self.logits, self.seq_len,preprocess_collapse_repeated=True,ctc_merge_repeated=True)
        self.cost = tf.reduce_mean(self.loss)
        self.train_fn = tf.train.MomentumOptimizer(self.learning_rate,
                                                   0.99).minimize(self.cost)
        #self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.logits, self.seq_len,merge_repeated=False)
        self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.logits, self.seq_len,merge_repeated=True)
        self.cast_seq=tf.cast(self.decoded[0],tf.int32)
        self.ler = tf.reduce_mean(tf.edit_distance(self.cast_seq, self.targets))
        """
        #inputs shape:[max_time,batch_size,depth]
        # project output from rnn output size to OUTPUT_SIZE. Sometimes it is worth adding
        # an extra layer here.
        #layers_fully_connected--Добавляет полносвязный слой
        final_projection = lambda x: layers.fully_connected(x, num_outputs=output_size, activation_fn=tf.nn.sigmoid)
        # apply projection to every timestep.
        predicted_outputs = map_fn(final_projection, rnn_outputs)
        
        
        
        # compute elementwise cross entropy.
        error = -(self.outputs * tf.log(predicted_outputs + self.TINY) + (1.0 - self.outputs) * tf.log(1.0 - predicted_outputs + self.TINY))
        error = tf.reduce_mean(error)
        
        # optimize
        self.train_fn = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(error)
        #self.label_fn = 
        # assuming that absolute difference between output and correct answer is 0.5
        # or less we can round it to the correct output.
        self.accuracy = tf.reduce_mean(tf.cast(tf.abs(self.outputs - predicted_outputs) < 0.5, tf.float32))
        """
        pass
