from unittest import TestCase
from trainer_builder import TrainerBuilder
from configurator import TrainConfig
import configurator as conf
import tensorflow as tf

class TestTrainerBuilder(TestCase):

    def test_parameters(self):
        """Проверяет задание параметров конфигураций"""
        for config in TrainConfig:
            tf.reset_default_graph()
            tb=TrainerBuilder(config)
            tr=tb.trainer
            assert tr.num_epochs==conf.get_num_epochs(config)
            assert tr.model_name==conf.get_model_name(config)
            assert tr.network is not None


    pass
