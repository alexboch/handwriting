import tensorflow as tf
import numpy as np

sess=tf.InteractiveSession()
#a=tf.nn.softmax(tf.constant([[[0,2]]],dtype=tf.float32))
a=tf.constant([[[0,1],[0,2]]],dtype=tf.float32)
sm=tf.nn.softmax(a)

print( f"Unscaled logits:{a.eval()}")
print(f"Softmax:{sm.eval()}")
b=tf.constant([[1],[1]])
w=tf.constant([[1.0],[1.0]])
loss=tf.contrib.seq2seq.sequence_loss(a,b,weights=w)
#tf.nn.softmax_cross_entropy_with_logits()
le=loss.eval()
print(le)

#a = tf.placeholder(tf.float32, shape =[None, 1])
#b = tf.placeholder(tf.float32, shape = [None, 1])
#a=tf.nn.softmax(tf.constant([[0,2]],dtype=tf.float32))
#a=tf.constant([[[0,2]]],dtype=tf.float32)
b=tf.constant([[0,1],[0,1]],dtype=tf.float32)
sess.run(tf.global_variables_initializer())
c = tf.nn.softmax_cross_entropy_with_logits(
    logits=a, labels=b
).eval()
print(c)