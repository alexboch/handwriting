from unittest import TestCase
from trainer_builder import TrainerBuilder
from configurator import TrainConfig
import configurator as conf
import tensorflow as tf
import utils

utils.enable_kb_interrupt()
tf.reset_default_graph()
#tb = TrainerBuilder(TrainConfig.BORDERS)
tb=TrainerBuilder(TrainConfig.BORDERS)
#tb.set_learning_rate(1e-2)

#tb.set_learning_rate(1e-5)

tr = tb.build_trainer()
#tr.num_epochs=1
#tr.num_epochs=1
#tr.load_data()
#tr.train_network([tr.data_loader.get_words_list()[0]])
#tr.get_labels_for_list([tr.data_loader.get_words_list()[0]])
tr.run_training(test=False)