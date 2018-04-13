import tensorflow as tf
import numpy as np

def print_ops():
    graph = tf.get_default_graph()
    ops = graph.get_operations()
    print("Total ops:", len(ops))
    for op in ops:
        print(op.name, op.type)

state_size = 5
vector_size = 4

with tf.Graph().as_default():
    # This does not add anything to the graph
    cell = tf.contrib.rnn.BasicLSTMCell(state_size)
    # Typically for training we want batches and unroll the network several (timesteps) times
    batch_size = 3
    timesteps = 2
    x_input = tf.placeholder(tf.float32, [batch_size, timesteps, vector_size], name='x_input')
    cell_state = tf.placeholder(tf.float32, [batch_size, state_size], name='cell_state')
    hidden_state = tf.placeholder(tf.float32, [batch_size, state_size], name='hidden_state')
    initial_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)
    # TIMESTEPS elements in list of tensors BATCH_SIZE x VECTOR_SIZE
    rnn_inputs = tf.unstack(x_input, num=timesteps, axis=1)
    # Here RNNs weights and kernel are added to graph - rnn/basic_lstm_cell/kernel and rnn/basic_lstm_cell/bias
    outputs, current_state = tf.nn.static_rnn(cell, rnn_inputs, initial_state, dtype=tf.float32)
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        # This initializes weights and biases in rnn cell
        sess.run(init_op)
        # Save the variables to disk.
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)
        print_ops()

        x_count = batch_size * timesteps * vector_size
        unrolled_output_array, unrolled_state_tuple = sess.run([outputs, current_state], feed_dict={
            x_input: np.linspace(1,x_count,x_count).reshape ([batch_size, timesteps, vector_size]),
            cell_state: np.zeros([batch_size, state_size]),
            hidden_state: np.zeros([batch_size, state_size])
        })
        print("Output before save:", unrolled_output_array)
        print("State before save:", unrolled_state_tuple)

with tf.Graph().as_default():
    cell = tf.contrib.rnn.BasicLSTMCell(state_size)
    # When using trained model typically we want to apply it at each timestep, "one input at a time"
    batch_size = 1
    timesteps = 1
    x_input = tf.placeholder(tf.float32, [batch_size, timesteps, vector_size], name='x_input')
    cell_state = tf.placeholder(tf.float32, [batch_size, state_size], name='cell_state')
    hidden_state = tf.placeholder(tf.float32, [batch_size, state_size], name='hidden_state')
    initial_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)
    # list of 1 element with tensor 1 x VECTOR_SIZE
    rnn_inputs = tf.unstack(x_input, num=timesteps, axis=1)
    outputs, current_state = tf.nn.static_rnn(cell, rnn_inputs, initial_state, dtype=tf.float32)
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables including RNN weights - rnn/basic_lstm_cell/kernel and rnn/basic_lstm_cell/bias
        saver.restore(sess, "/tmp/model.ckpt")
        print("Model restored.")
        print_ops()

        x_count = batch_size * timesteps * vector_size # = 4
        output_array, _next_state = sess.run([outputs, current_state], feed_dict={
            x_input: np.linspace(1, x_count, x_count).reshape([batch_size, timesteps, vector_size]),
            cell_state:  np.zeros((batch_size, state_size)),
            hidden_state:  np.zeros((batch_size, state_size))
        })
        _current_cell_state, _current_hidden_state = _next_state
        print("Output after save:", output_array)
np.testing.assert_array_almost_equal(output_array[0][0], unrolled_output_array[0][0])