import tensorflow as tf
import numpy as np

num_features=1
num_units=50
num_classes=2

def lstm_cell():
    return tf.contrib.rnn.LSTMCell(num_units,state_Is_tuple=True)



inputs=tf.placeholder(tf.float32,[None,None,num_features])
W=tf.Variable(tf.truncated_normal([num_units, num_classes], stddev=0.1), name = 'W')
b=tf.Variable(tf.constant(0.,shape=[num_classes]))
seq_len=tf.placeholder(tf.int32,[None],name='seq_len')
rnn_outputs,_=tf.nn.dynamic_rnn(lstm_cell(),inputs,seq_len)
rnn_outputs=tf.reshape(rnn_outputs,[-1,num_units])
logits=tf.matmul(rnn_outputs,W)+b
logits=tf.reshape(logits,[1,-1,num_classes])
probs=tf.nn.softmax(logits)
logits=tf.transpose(logits,(1,0,2))
