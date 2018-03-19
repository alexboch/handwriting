import numpy as np
import tensorflow as tf
from utils import *
import tensorflow.contrib.layers as layers
map_fn = tf.map_fn

class LSTMDecoder:
    """
    Класс для создания, обучения и получения разметки от LSTM-нейросети
    """
    
    def __init__(self,num_units,num_layers,input_size,num_classes,learning_rate,batch_size):
        """
        Конструктор, в нем задаются размеры слоев и создается клетка сети
        """
        self.create_network(num_units,num_layers,input_size,
                            num_classes,learning_rate,batch_size)
        pass
        
    
    
    def label(self,data):
        """
        Принимает список точек и возвращает последовательность меток 
        """
        
        pass
    
    
    TINY  = 1e-6    # to avoid NaNs in logs
    def train(self,words,num_epochs=100):
        """
        words--Список слов, содержащих точки и метки
        """
        #init_op = tf.initialize_all_variables()
        session=tf.Session()
        session.run(tf.global_variables_initializer())
        num_batches=len(words)/self.batch_size
        num_words=len(words)
        for i in np.arange(num_epochs):
            print("Epoch number ",str(i))
            np.random.shuffle(words)
            epoch_error=0#Средняя ошибка по всем батчам в данной эпохе
            for j in np.arange(0,num_words,self.batch_size):#Цикл по всем словам, берем по batch_size слов
                j1=j
                j2=j1+self.batch_size
                batch_words=words[j:j2]#слова для создания батча
                batch_inputs=[]
                batch_labels=[]
                seq_length = []  # Вектор длин каждой последовательности
                #получить список точек и меток
                for w in batch_words:
                    batch_inputs.append(w.point_list)
                    seq_length.append(len(w.point_list))#Присоединяем длину последовательности точек
                    batch_labels.append(w.labels_list)
                inputs_arr=np.asarray(batch_inputs)
                targets_array=np.asarray(batch_labels)
                #targets_array=np.expand_dims(targets_array, 2)#чтобы создать из числа-метки массив из одного числа
                #targets_array=targets_array.astype(float)
                targets_array=sparse_tuple_from(targets_array)
                s,_=session.run([self.cost,self.train_fn], feed_dict={self.inputs:inputs_arr, self.targets:targets_array,self.seq_len:seq_length})
                epoch_error+=s
                print("Batch cost:",s)
                #session.run(self.train_fn,feed_dict={self.inputs:batch_inputs, self.outputs:batch_labels})
            epoch_error /= num_words
            print("Epoch error:",epoch_error)
        session.close()
        pass
        
    
     
    def create_network(self, num_units, num_layers, input_size, num_classes, learning_rate, batch_size=10):
        self.num_units=num_units
        self.num_layers=num_layers
        self.learning_rate=learning_rate
        self.inputs=tf.placeholder(tf.float32,[None,None,input_size],name='inputs')
        shape = tf.shape(self.inputs)
        batch_s, max_timesteps = shape[0], shape[1]
        #self.targets = tf.placeholder(tf.float32, (None, None, num_classes))#
        # Here we use sparse_placeholder that will generate a
        # SparseTensor required by ctc_loss op.

        self.targets = tf.sparse_placeholder(tf.int32)
        #dropout=tf.placeholder(tf.float32,name='dropout')
        cell=tf.contrib.rnn.LSTMCell(num_units,state_is_tuple=True)
        self.cells_stack=tf.contrib.rnn.MultiRNNCell([cell] * num_units, state_is_tuple=True)
        self.W=tf.Variable(tf.truncated_normal([num_units, num_classes], stddev=0.1))#Начальная матрица весов
        b=tf.Variable(tf.constant(0., shape=[num_classes]))
        # Given inputs (time, batch, input_size) outputs a tuple
        #  - outputs: (time, batch, output_size)  [do not mistake with OUTPUT_SIZE]
        #  - states:  (time, batch, hidden_size)
        self.batch_size=batch_size

        # 1d array of size [batch_size]
        self.seq_len = tf.placeholder(tf.int32, [None])
        #seq_len=np.ndarray(shape=[batch_size],dtype=int)

        # If cell.state_size is an integer, this must be a Tensor of appropriate type and shape [batch_size, cell.state_size].
        #If cell.state_size is a tuple,
        #this should be a tuple of tensors having shapes [batch_size, s] for s in cell.state_size
        initial_state=self.cells_stack.zero_state(self.batch_size, dtype=tf.float32)
        #rnn_outputs--Тензор размерности [batch_size,max_time,cell.output_size],max_time--кол-во точек слова,cell.output_size=2(x,y координаты)
        #rnn_state--последнее состояние
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(self.cells_stack, self.inputs,self.seq_len, dtype=tf.float32)
        # Reshaping to apply the same weights over the timesteps
        rnn_outputs = tf.reshape(rnn_outputs, [-1, num_units])
        logits=tf.matmul(rnn_outputs,self.W)+b
        # Reshaping back to the original shape
        logits = tf.reshape(logits, [self.batch_size, -1, num_classes])

        # Time major
        logits = tf.transpose(logits, (1, 0, 2))
        loss=tf.nn.ctc_loss(self.targets, logits, self.seq_len, ctc_merge_repeated=False)
        self.cost = tf.reduce_mean(loss)
        self.train_fn = tf.train.MomentumOptimizer(learning_rate,
                                           0.9).minimize(self.cost)

        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, self.seq_len)

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
