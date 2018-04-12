from unittest import TestCase
from trainer_builder import TrainerBuilder
from configurator import TrainConfig
import configurator as conf
import tensorflow as tf


tf.reset_default_graph()
tb = TrainerBuilder(TrainConfig.LETTERS_MERGED)
tr = tb.trainer
#tr.num_epochs=1
#tr.load_data()
#tr.train_network([tr.data_loader.get_words_list()[0]])
tr.run_training()