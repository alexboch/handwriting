from unittest import TestCase
from graph_helper import GraphHelper
import tensorflow as tf
import numpy as np
import sys


class TestGraphHelper(TestCase):

    def generate_input(self, num):
        sequence = np.random.randint(2, size=num)
        nonzero = np.count_nonzero(sequence, 0)
        if nonzero > num / 2:
            target = [1]
        else:
            target = [0]
        return sequence, target

    def test_save_restore(self):
        model_dir = "Models/test2/"
        model_name = 'test_model'
        n = 3
        num_epochs = 2000
        num_inputs = 8
        learning_rate = 1e-1
        inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        test_targets = [(lambda x: [1] if np.count_nonzero(x) > n / 2 else [0])(x) for x in
                        inputs]  # 1, если больше половины единиц, иначе 0

        targets = tf.placeholder(tf.float32, [None, 1])
        # Создать тестовую модель
        x = tf.placeholder(tf.float32, [None, n], name='x')
        W = tf.Variable(tf.truncated_normal([n, 1]), name='W')
        b = tf.Variable(tf.constant(0., shape=[1]), name='b')
        y = tf.nn.sigmoid(tf.matmul(x, W) + b, name='y')
        loss = tf.losses.absolute_difference(targets, y)
        train_func = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # тренировка модели
        for i in np.arange(num_epochs):
            _, loss_result = sess.run([train_func, loss], feed_dict={x: inputs, targets: test_targets})
            # print("epoch:",i)
            # print("loss:",loss_result)

        # проверка, что веса изменились:
        for i in np.arange(100):
            np.random.shuffle(inputs)
            test_outputs = sess.run(y, feed_dict={x: inputs})
            test_targets = [(lambda x: [1] if np.count_nonzero(x) > n / 2 else [0])(x) for x in
                            inputs]  # 1, если больше половины единиц, иначе 0
            th_outputs = [(lambda x: [1] if x[0] > 0.5 else [0])(x) for x in test_outputs]
            assert (th_outputs == test_targets)
        # Сохранение ckpt:
        saver = tf.train.Saver()
        saver.save(sess, model_dir + model_name);
        # saver.save(sess,model_dir)
        sess.close()
        # frozen_graph=GraphHelper.freeze_graph(model_dir,'y')#Заморозка графа

        filename = 'graph.pb'
        # tf.train.write_graph(sess.graph_def, model_dir,filename,as_text=False)

        # loaded_graph=GraphHelper.load_graph(model_dir+filename)
        """
        assert (loaded_graph is not None)
        for op in loaded_graph.get_operations():
            print(op.name)
"""
        try:
            # x=loaded_graph.get_tensor_by_name('prefix/x:0')

            # y=loaded_graph.get_tensor_by_name('prefix/y:0')

            with tf.Session() as sess:
                tf.train.import_meta_graph(model_dir + model_name + ".meta")
                # sess.run(tf.global_variables_initializer())
                saver.restore(sess, tf.train.latest_checkpoint(model_dir))
                for i in np.arange(100):
                    np.random.shuffle(inputs)
                    test_outputs = sess.run(y, feed_dict={x: inputs})
                    test_targets = [(lambda x: [1] if np.count_nonzero(x) > n / 2 else [0])(x) for x in
                                    inputs]  # 1, если больше половины единиц, иначе 0
                    th_outputs = [(lambda x: [1] if x[0] > 0.5 else [0])(x) for x in test_outputs]
                    assert (th_outputs == test_targets)
        except Exception as ex:
            print(ex)
            self.fail(ex)

    pass


    def test_freeze_graph(self):
        GraphHelper.freeze_graph('Data\\BORDERS10-05-2018-01_17_02\\','probs')
        graph=GraphHelper.load_graph('Data\\BORDERS10-05-2018-01_17_02\\frozen_model.pb')
        self.fail()
