import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
map_fn = tf.map_fn

class LSTMDecoder:
    """
    Класс для создания, обучения и получения разметки от LSTM-нейросети
    """
    
    def __init__(self,num_units,num_layers,input_size,output_size,learning_rate,batch_size):
        """
        Конструктор, в нем задаются размеры слоев и создается клетка сети
        """
        self.create_network(num_units,num_layers,input_size,output_size,learning_rate,batch_size)
        pass
        
    
    
    def label(self,data):
        """
        Принимает список точек и возвращает последовательность меток 
        """
        pass
    
    
    TINY  = 1e-6    # to avoid NaNs in logs
    def train(self,words,num_epochs=1000):
        """
        words--Список слов, содержащих точки и метки
        """
        init_op = tf.initialize_all_variables()
        session=tf.Session()
        session.run(init_op)
        num_batches=len(words)/self.batch_size
        num_words=len(words)
        for i in np.arange(num_epochs):
            print("Epoch number ",str(i))
            for j in np.arange(0,num_words,self.batch_size):
                j1=j
                j2=j1+self.batch_size
                batch_words=words[j:j2]#слова для создания батча
                batch_inputs=[]
                batch_labels=[]
                #получить список точек и меток
                for w in batch_words:
                    batch_inputs.append(w.point_list)
                    batch_labels.append(w.labels_list)
                inputs_arr=np.asarray(batch_inputs)
                outputs_arr=np.asarray(batch_labels)
                outputs_arr=np.expand_dims(outputs_arr,2)
                outputs_arr=outputs_arr.astype(float)
                session.run(self.train_fn,feed_dict={self.inputs:inputs_arr,self.outputs:outputs_arr})
                #session.run(self.train_fn,feed_dict={self.inputs:batch_inputs, self.outputs:batch_labels})
        pass
        
    
     
    def create_network(self,num_units,num_layers,input_size,output_size,learning_rate,batch_size=10):
        self.num_units=num_units
        self.num_layers=num_layers
        self.learning_rate=learning_rate
        self.inputs=tf.placeholder(tf.float32,(None,None,input_size)) # (time, batch, in)
        self.outputs = tf.placeholder(tf.float32, (None, None, output_size))# (time, batch, out)
        cells=[]#список клеток
        dropout=tf.placeholder(tf.float32)
        for _ in range(num_layers):#Создать клетки для слоев
            cell=tf.contrib.rnn.BasicLSTMCell(num_units,state_is_tuple=True)
            #cell=tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=1.0-dropout)
            cells.append(cell)
        self.cell=tf.contrib.rnn.MultiRNNCell(cells)#Создаем клетку из нескольких
        
        # Given inputs (time, batch, input_size) outputs a tuple
        #  - outputs: (time, batch, output_size)  [do not mistake with OUTPUT_SIZE]
        #  - states:  (time, batch, hidden_size)
        self.batch_size=batch_size
        # If cell.state_size is an integer, this must be a Tensor of appropriate type and shape [batch_size, cell.state_size]. 
        #If cell.state_size is a tuple,
        #this should be a tuple of tensors having shapes [batch_size, s] for s in cell.state_size
        initial_state=self.cell.zero_state(batch_size,dtype=tf.float32)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(self.cell, self.inputs, initial_state=initial_state)
        #inputs shape:[max_time,batch_size,depth]
        # project output from rnn output size to OUTPUT_SIZE. Sometimes it is worth adding
        # an extra layer here.
        final_projection = lambda x: layers.fully_connected(x, num_outputs=output_size, activation_fn=tf.nn.sigmoid)
        # apply projection to every timestep.
        predicted_outputs = map_fn(final_projection, rnn_outputs)
        # compute elementwise cross entropy.
        error = -(self.outputs * tf.log(predicted_outputs + self.TINY) + (1.0 - self.outputs) * tf.log(1.0 - predicted_outputs + self.TINY))
        error = tf.reduce_mean(error)
        # optimize
        self.train_fn = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(error)

        # assuming that absolute difference between output and correct answer is 0.5
        # or less we can round it to the correct output.
        self.accuracy = tf.reduce_mean(tf.cast(tf.abs(self.outputs - predicted_outputs) < 0.5, tf.float32))
        pass
