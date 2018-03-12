import tensorflow as tf
import numpy as np
import lstm

tf.reset_default_graph()
ld=lstm.LSTMDecoder(2,1,2,2,1,1)