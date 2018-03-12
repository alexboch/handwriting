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
        
    TINY  = 1e-6    # to avoid NaNs in logs
    def train(self,inputs,labels,num_epochs=1000,iterations_per_epoch=100):
        
        #for epoch in np.arange(num_epochs):
         #   for _ in np.arange(iterations_per_epoch):
                
        pass
        
    
    
    def create_network(self,num_units,num_layers,input_size,output_size,learning_rate,batch_size=100):
        self.num_units=num_units
        self.num_layers=num_layers
        self.learning_rate=learning_rate
        self.inputs=tf.placeholder(tf.float32,(None,None,input_size)) # (time, batch, in)
        self.outputs = tf.placeholder(tf.float32, (None, None, output_size))# (time, batch, out)
        cells=[]#список клеток
        dropout=tf.placeholder(tf.float32)
        for _ in range(num_layers):#Создать клетки для слоев
            cell=tf.contrib.rnn.BasicLSTMCell(num_units,state_is_tuple=True)
            cell=tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=1.0-dropout)
            cells.append(cell)
        self.cell=tf.contrib.rnn.MultiRNNCell(cells)#Создаем клетку из нескольких
        
        # Given inputs (time, batch, input_size) outputs a tuple
        #  - outputs: (time, batch, output_size)  [do not mistake with OUTPUT_SIZE]
        #  - states:  (time, batch, hidden_size)
        self.batch_size=batch_size
        # If cell.state_size is an integer, this must be a Tensor of appropriate type and shape [batch_size, cell.state_size]. 
        #If cell.state_size is a tuple,
        #this should be a tuple of tensors having shapes [batch_size, s] for s in cell.state_size
        cell_state=tf.placeholder(tf.float32, [batch_size, num_units])
        hidden_state=tf.placeholder(tf.float32,[batch_size, num_units])
        initial_state=tf.nn.rnn_cell.LSTMStateTuple(cell_state,hidden_state)
        #initial_state=np.zeros((cell.state_size.c,cell.state_size.h))
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
